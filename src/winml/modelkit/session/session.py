# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - Core ONNX Runtime session manager."""

from __future__ import annotations

import gc
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import onnxruntime as ort

from ..core.onnx_utils import get_io_config
from ..onnx import is_compiled_onnx
from .ep_registry import WinMLEPRegistry
from .monitor.ep_monitor import EPMonitor, NullEPMonitor
from .stats import PerfStats


if TYPE_CHECKING:
    from collections.abc import Generator

    import onnx

    from ..compiler.configs import EPConfig


logger = logging.getLogger(__name__)


@contextmanager
def _suppress_native_output(log_path: str | Path | None = None):
    """Redirect native stdout to a log file (or devnull).

    QNN SDK compiler writes progress to stdout via native C++ code that
    Python logging/warnings cannot intercept. Only redirects stdout —
    stderr is left untouched so Rich displays and Python logging work.
    """
    if log_path is not None:
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    else:
        fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    os.dup2(fd, 1)
    os.close(fd)
    try:
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(old_stdout)


class SessionState(Enum):
    """WinMLSession states."""

    INITIALIZED = "INITIALIZED"
    COMPILED = "COMPILED"
    INFERRING = "INFERRING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PerfContext:
    """Yielded by ``WinMLSession.perf()``.

    Aggregates perf statistics and the optional attached EP monitor.
    Frozen: mutation is not a supported pattern — update the underlying
    objects instead.
    """

    stats: PerfStats
    monitor: EPMonitor  # NullEPMonitor when no monitor was passed


# Device to ORT policy mapping (no EP names - let ORT select provider)
DEVICE_POLICY_MAP = {
    "npu": ort.OrtExecutionProviderDevicePolicy.PREFER_NPU,
    "gpu": ort.OrtExecutionProviderDevicePolicy.PREFER_GPU,
    "cpu": ort.OrtExecutionProviderDevicePolicy.PREFER_CPU,
    "auto": ort.OrtExecutionProviderDevicePolicy.PREFER_NPU,  # Default to NPU
}


class WinMLSessionError(Exception):
    """Base exception for WinMLSession."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.message = message
        self.context = context or {}
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {ctx_str}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class CompilationError(WinMLSessionError):
    """Compilation failed."""


class DeviceNotAvailableError(WinMLSessionError):
    """Requested device not available."""


class InferenceError(WinMLSessionError):
    """Inference failed."""


class NotCompiledError(WinMLSessionError):
    """Session not compiled."""


class WinMLSession:
    """ONNX Runtime session manager with WinML EP integration.

    Features:
    - Policy-based device selection (PREFER_NPU, PREFER_GPU, PREFER_CPU)
    - EPContext persistence (JIT-compiled model cache)
    - One session = One EP (immutable binding)

    Note:
        WinMLSession does NOT use explicit EP provider names. Instead, it uses
        ORT's OrtExecutionProviderDevicePolicy to let the runtime automatically
        select the best available provider.

    Usage:
        session = WinMLSession("model.onnx", device="npu")
        outputs = session.run({"input": tensor})
    """

    # Class-level flag for one-time EP initialization
    _eps_initialized: bool = False

    # EP short name -> ORT full provider name (for add_provider_for_devices matching)
    _EP_NAME_MAP: ClassVar[dict[str, str]] = {
        "qnn": "QNNExecutionProvider",
        "dml": "DmlExecutionProvider",
        "migraphx": "MIGraphXExecutionProvider",
        "tensorrt": "NvTensorRTRTXExecutionProvider",
        "vitisai": "VitisAIExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }

    @classmethod
    def _init_winml_eps_once(cls) -> None:
        """Initialize WinML EP registry once at class level."""
        if cls._eps_initialized:
            return

        try:
            registry = WinMLEPRegistry.get_instance()
            if registry.winml_available:
                registered = registry.register_to_ort()
                logger.info("WinML EPs registered: %s", registered)
        except Exception as e:
            logger.debug("WinML EP init skipped: %s", e)
        finally:
            cls._eps_initialized = True

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "auto",
        ep_config: EPConfig | None = None,
        *,
        ep: str | None = None,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        """Initialize WinMLSession.

        Args:
            onnx_path: Path to ONNX model
            device: Target device policy ("auto", "npu", "gpu", "cpu").
                Note: This specifies a policy (PREFER_NPU, PREFER_GPU, PREFER_CPU),
                not a specific execution provider name. ORT selects the best
                available provider for the requested policy.
            ep_config
                persist_jit: Persist JIT-compiled EPContext model
                provider_options: EP-specific options dict
            ep: Explicit EP short name (e.g., "migraphx", "tensorrt").
                When set, bypasses policy-based selection and uses
                add_provider_for_devices to force the specific EP.
            session_options: ORT SessionOptions. If None, creates default with
                policy based on device parameter.
        """
        WinMLSession._init_winml_eps_once()

        self._onnx_path = Path(onnx_path)
        if not self._onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # HF Pipeline may pass torch.device; coerce to string for downstream .lower() calls
        self._device = str(device) if not isinstance(device, str) else device
        self._ep = ep.lower() if ep else None
        self._persist_jit = ep_config.enable_ep_context if ep_config else False
        self._embed_context = ep_config.embed_context if ep_config else False
        self._provider_options = ep_config.provider_options if ep_config else {}
        # Monitor-contributed session config entries (populated by session.perf(monitor=...))
        self._active_session_option_entries: dict[str, str] = {}

        # Create session_options with device policy
        if session_options is None:
            session_options = ort.SessionOptions()
        self._session_options = session_options

        # State management
        self._state = SessionState.INITIALIZED
        self._last_error: Exception | None = None

        # Single session (one session = one EP)
        self._session: ort.InferenceSession | None = None

        # Cached I/O metadata (lazy-loaded)
        self._io_config: dict | None = None

        # Performance tracking (enabled via perf() context manager)
        self._perf_stats: PerfStats | None = None

        logger.info("WinMLSession initialized: %s", onnx_path)

    def compile(self) -> None:
        """Compile model for target device using ModelCompiler API.

        Only compiles once per session (idempotent).
        Device is immutable - set at __init__ time.
        """
        # If already compiled, ignore (idempotent)
        if self._session is not None:
            if self._is_verbose():
                logger.info("Already compiled for %s", self._device)
            return

        target_device = self._device

        # Resolve auto device
        if target_device == "auto":
            target_device = self._detect_best_device()
            self._device = target_device  # Update instance device

        if self._is_verbose():
            logger.info("Compiling for device: %s", target_device)

        # Determine model path (original or EPContext)
        ctx_path = self._onnx_path.parent / f"{self._onnx_path.stem}_{target_device}_ctx.onnx"
        model_path = self._onnx_path

        # Check for existing fresh EPContext
        if (
            self._persist_jit
            and ctx_path.exists()
            and ctx_path.stat().st_mtime >= self._onnx_path.stat().st_mtime
        ):
            model_path = ctx_path
            logger.info("Using cached EPContext: %s", ctx_path)

        # Compile if needed (persist_jit=True and no cache)
        # Native QNN SDK compiler writes progress to stdout/stderr;
        # redirect to log file to keep the console clean.
        compile_log = self._onnx_path.parent / "compile.log"

        if self._persist_jit and model_path == self._onnx_path:
            # Skip ModelCompiler if input model is already compiled (EPContext)
            if is_compiled_onnx(self._onnx_path):
                logger.info("Model already compiled (EPContext), skipping ModelCompiler")
            else:
                try:
                    sess_options = self._build_session_options(target_device)

                    model_compiler = ort.ModelCompiler(
                        sess_options,
                        str(self._onnx_path),
                        embed_compiled_data_into_model=self._embed_context,
                    )
                    with _suppress_native_output(compile_log):
                        model_compiler.compile_to_file(str(ctx_path))

                    # Use compiled model if it was created
                    if ctx_path.exists():
                        model_path = ctx_path
                        logger.info("Compiled to EPContext: %s", ctx_path)

                except Exception as e:
                    # Some EPs don't support compilation - fall back to original
                    logger.warning("ModelCompiler failed, using original: %s", e)

        try:
            # Create InferenceSession
            sess_options = self._build_session_options(target_device)
            with _suppress_native_output(compile_log):
                session = ort.InferenceSession(str(model_path), sess_options=sess_options)

            # Log which providers were selected by ORT (based on policy)
            actual_providers = session.get_providers()
            logger.info(
                "Session created with policy %s, providers: %s",
                target_device,
                actual_providers,
            )

        except Exception as e:
            self._state = SessionState.ERROR
            self._last_error = e
            raise CompilationError(
                message=f"Failed to compile for {target_device}",
                context={
                    "device": target_device,
                    "onnx_path": str(self._onnx_path),
                    "error": str(e),
                },
                suggestion=self._get_compile_suggestion(target_device, e),
            ) from e

        # Store session
        self._session = session
        self._state = SessionState.COMPILED

        # Resolve device label from the primary provider ORT actually selected
        if self._device == "auto" and actual_providers:
            from ..sysinfo.device import get_ep_device_map

            ep_map = get_ep_device_map()
            resolved = ep_map.get(actual_providers[0])
            if resolved and "/" not in resolved:
                self._device = resolved

    def run(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Run inference.

        Auto-compiles if not compiled. Validates inputs.

        Args:
            inputs: Input tensors (torch.Tensor or numpy arrays)

        Returns:
            Dictionary of output name -> numpy array

        Raises:
            ValueError: If inputs is empty or None
            InferenceError: If inference fails
        """
        # Validate inputs early
        if not inputs:
            raise ValueError("inputs cannot be empty")

        # Ensure compiled (auto-compile on first run)
        if self._session is None:
            self.compile()

        if self._state == SessionState.ERROR:
            raise InferenceError(
                message="Session in error state",
                context={"last_error": str(self._last_error)},
                suggestion="Call reset() and try again",
            )

        self._state = SessionState.INFERRING
        try:
            # Validate inputs (raises ValueError if missing)
            self._validate_inputs(inputs)

            # Prepare inputs (convert to numpy, enforce dtype)
            ort_inputs = self._prepare_inputs(inputs, self._session)

            # Run inference (with optional perf tracking)
            output_names = [o.name for o in self._session.get_outputs()]
            if self._perf_stats:
                outputs = self._perf_stats.record(
                    lambda: self._session.run(output_names, ort_inputs)
                )
            else:
                outputs = self._session.run(output_names, ort_inputs)

            # Build result dict
            return dict(zip(output_names, outputs, strict=True))

        except Exception as e:
            self._state = SessionState.ERROR
            self._last_error = e
            raise InferenceError(
                message="Inference failed",
                context={"error": str(e)},
            ) from e

        finally:
            if self._state == SessionState.INFERRING:
                self._state = SessionState.COMPILED

    def reset(self) -> None:
        """Reset session to INITIALIZED state.

        Clears compiled session and error state.
        """
        self._session = None
        self._state = SessionState.INITIALIZED
        self._last_error = None
        logger.info("Session reset")

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        try:
            self._session = None
        except Exception:
            pass  # Suppress errors during interpreter shutdown

    def _is_verbose(self) -> bool:
        """Check if verbose logging is enabled via environment variable."""
        return os.getenv("WINMLSESSION_VERBOSE", "").lower() in ("1", "true", "yes")

    def _build_session_options(self, device: str) -> ort.SessionOptions:
        """Build ORT SessionOptions from instance session_options and device.

        When ``self._ep`` is set, uses ``add_provider_for_devices`` to
        explicitly bind a specific EP (e.g., MIGraphX, TensorRT). Otherwise
        falls back to policy-based selection via DEVICE_POLICY_MAP.

        Note: Returns a **fresh** SessionOptions when using explicit EP to
        avoid "already registered" errors from repeated calls.
        """
        # Explicit EP targeting: create fresh opts to avoid double-registration
        if self._ep and self._ep != "cpu":
            target_name = self._EP_NAME_MAP.get(self._ep)
            if target_name:
                matched = self._find_ep_device(target_name)
                if matched:
                    opts = ort.SessionOptions()
                    opts.add_provider_for_devices([matched], self._provider_options)
                    logger.info(
                        "Explicit EP: %s (%s)",
                        self._ep,
                        target_name,
                    )
                    # Apply monitor-contributed session config entries
                    for key, value in self._active_session_option_entries.items():
                        opts.add_session_config_entry(key, value)
                    return opts
                logger.warning(
                    "EP '%s' (%s) not found in available devices; falling back to policy",
                    self._ep,
                    target_name,
                )

        # Policy-based selection (default path)
        opts = self._session_options
        policy = DEVICE_POLICY_MAP.get(
            device.lower(), ort.OrtExecutionProviderDevicePolicy.PREFER_NPU
        )
        opts.set_provider_selection_policy(policy)
        # Apply monitor-contributed session config entries
        for key, value in self._active_session_option_entries.items():
            opts.add_session_config_entry(key, value)

        return opts

    @staticmethod
    def _find_ep_device(ep_name: str) -> Any:
        """Find an OrtEpDevice matching the given EP name.

        Returns:
            The first matching OrtEpDevice, or None if not found.
        """
        for ep_dev in ort.get_ep_devices():
            if ep_dev.ep_name == ep_name:
                return ep_dev
        return None

    def _validate_inputs(self, inputs: dict[str, Any]) -> None:
        """Validate inputs against model expectations.

        Raises ValueError for missing required inputs.
        Logs warnings for unexpected inputs.
        """
        expected_inputs = set(self.io_config["input_names"])
        provided_inputs = set(inputs.keys())

        # Check for missing inputs (strict - raise error)
        missing = expected_inputs - provided_inputs
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")

        # Check for unexpected inputs (soft - warn only)
        unexpected = provided_inputs - expected_inputs
        if unexpected:
            logger.warning(
                "Unexpected input names: %s. Expected: %s",
                unexpected,
                expected_inputs,
            )

    def _prepare_inputs(
        self, inputs: dict[str, Any], session: ort.InferenceSession
    ) -> dict[str, np.ndarray]:
        """Convert inputs to numpy arrays and enforce correct dtypes.

        Args:
            inputs: Input tensors (torch.Tensor, numpy arrays, or convertible)
            session: ORT InferenceSession for metadata

        Returns:
            Dict of input_name -> numpy array with correct dtype
        """
        # Build dtype map from io_config
        io_cfg = self.io_config
        name_to_type = dict(zip(io_cfg["input_names"], io_cfg["input_types"], strict=True))

        ort_inputs = {}
        for name, value in inputs.items():
            # Convert to numpy
            if hasattr(value, "numpy"):  # torch.Tensor
                arr = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                arr = value
            else:
                arr = np.array(value)

            # Enforce correct dtype if known
            expected_type = name_to_type.get(name)
            if expected_type is not None and arr.dtype != expected_type:
                arr = arr.astype(expected_type)

            ort_inputs[name] = arr

        return ort_inputs

    def _detect_best_device(self) -> str:
        """Auto-detect best available device.

        Returns "auto" to let ORT select the best provider based on PREFER_NPU policy.
        This avoids using any explicit EP provider names.
        """
        # With PREFER_NPU policy, ORT will automatically select:
        # 1. NPU (QNN) if available
        # 2. GPU (CUDA/DML) if no NPU
        # 3. CPU as fallback
        logger.info("Auto-detecting device (using PREFER_NPU policy)")
        return "auto"

    def _get_compile_suggestion(self, device: str, error: Exception) -> str:
        """Get compile error suggestion based on device policy."""
        error_str = str(error).lower()

        if device in ("npu", "auto"):
            if "backend" in error_str:
                return "Ensure NPU backend DLLs are in PATH (e.g., Qualcomm AI Stack)"
            return "Verify NPU drivers and runtime are properly installed"

        if device == "gpu":
            return "Verify GPU drivers and ONNX Runtime GPU package are installed"

        return "Check error details above"

    def _get_install_suggestion(self, device: str) -> str:
        """Get install suggestion for device policy."""
        suggestions = {
            "npu": "Install appropriate NPU ONNX Runtime package",
            "gpu": "Install onnxruntime-gpu or onnxruntime-directml",
        }
        return suggestions.get(device.lower(), "")

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state

    @property
    def device(self) -> str:
        """Target device for this session."""
        return self._device

    @property
    def is_compiled(self) -> bool:
        """Check if session is compiled."""
        return self._session is not None

    @property
    def perf_stats(self) -> PerfStats | None:
        """Performance statistics (None if not in perf() context).

        Returns:
            PerfStats instance with timing data, or None if outside perf() context.
        """
        return self._perf_stats

    @contextmanager
    def perf(
        self,
        warmup: int = 0,
        monitor: EPMonitor | None = None,
    ) -> Generator[PerfContext, None, None]:
        """Run a scoped performance window yielding a :class:`PerfContext`.

        Args:
            warmup: Number of initial samples to exclude from statistics.
            monitor: Optional :class:`EPMonitor`. Contributes session/provider
                options at compile time (auto-resets the session if already
                compiled with different options — logs WARNING). Parses
                artifacts on exit.

        Yields:
            :class:`PerfContext` with ``stats: PerfStats`` and
            ``monitor: EPMonitor`` (:class:`NullEPMonitor` when caller passed
            ``monitor=None``).

        Raises:
            RuntimeError: If another ``perf()`` context is already active on
                this session (nested ``perf()`` is forbidden).

        Example:
            >>> with session.perf(warmup=10) as ctx:
            ...     for _ in range(110):
            ...         session.run(inputs)
            >>> print(f"P99: {ctx.stats.p99_ms:.2f} ms")
        """
        if self._perf_stats is not None:
            raise RuntimeError("session.perf() already active (nested perf is forbidden)")

        mon: EPMonitor = monitor if monitor is not None else NullEPMonitor()

        # Collect hook contributions — must be idempotent per EPMonitor contract
        extra_sess = mon.get_session_options()
        extra_prov = mon.get_provider_options()

        # Auto-reset if options to apply AND session is already compiled
        if (extra_sess or extra_prov) and self._session is not None:
            logger.warning(
                "session.perf(): auto-resetting compiled session to apply monitor "
                "session/provider options (monitor=%s)",
                type(mon).__name__,
            )
            self.reset()

        # Save + merge
        saved_sess_entries = dict(self._active_session_option_entries)
        saved_prov = dict(self._provider_options)
        self._active_session_option_entries = {**saved_sess_entries, **extra_sess}
        self._provider_options = {**saved_prov, **extra_prov}

        stats = PerfStats(warmup=warmup)
        self._perf_stats = stats
        mon.__enter__()

        try:
            yield PerfContext(stats=stats, monitor=mon)
        finally:
            self._perf_stats = None
            exc_info = sys.exc_info()
            try:
                if mon.requires_session_teardown:
                    self.reset()
                    # Windows: release file handles before monitor parses artifacts
                    gc.collect()
            finally:
                try:
                    mon.__exit__(*exc_info)
                finally:
                    self._active_session_option_entries = saved_sess_entries
                    self._provider_options = saved_prov

    @property
    def io_config(self) -> dict:
        """ONNX I/O metadata (lazy-loaded, cached).

        Available before session compilation. Loads ONNX model once
        to extract input/output metadata.

        Returns:
            dict with:
                - input_names: list of input tensor names
                - input_shapes: list of input shapes (None for dynamic dims)
                - input_types: list of numpy dtypes for inputs
                - input_value_ranges: dict of input_name -> [low, high] (optional)
                - output_names: list of output tensor names
                - output_shapes: list of output shapes
                - output_types: list of numpy dtypes for outputs
        """
        if self._io_config is None:
            from ..onnx import load_onnx

            model = load_onnx(self._onnx_path, load_weights=False, validate=False)
            self._io_config = get_io_config(model)
            # Enrich with value_range from build config if available
            self._io_config["input_value_ranges"] = self._load_input_value_ranges()
        return self._io_config

    def _load_input_value_ranges(self) -> dict[str, list[int]]:
        """Load input value ranges from the winml_build_config.json.

        Searches for the build config file in the same directory as the
        ONNX model. Returns a mapping of input_name -> [low, high].

        Returns:
            dict mapping input names to their value ranges, empty if
            no build config is found.
        """
        import json

        value_ranges: dict[str, list[int]] = {}
        model_dir = self._onnx_path.parent

        # Try exact name first, then glob for prefixed variants
        candidates = [model_dir / "winml_build_config.json"]
        candidates.extend(model_dir.glob("*_winml_build_config.json"))

        for cfg_path in candidates:
            if cfg_path.is_file():
                try:
                    with cfg_path.open() as f:
                        build_cfg = json.load(f)
                    for tensor in (build_cfg.get("export") or {}).get("input_tensors", []):
                        name = tensor.get("name")
                        vr = tensor.get("value_range")
                        if name and vr and len(vr) == 2:
                            value_ranges[name] = vr
                    if value_ranges:
                        logger.debug(
                            "Loaded value_ranges from %s: %s",
                            cfg_path,
                            value_ranges,
                        )
                        return value_ranges
                except (json.JSONDecodeError, OSError) as exc:
                    logger.debug("Could not read build config %s: %s", cfg_path, exc)

        return value_ranges

    def is_compatible(
        self,
        node: onnx.NodeProto,
        graph: onnx.GraphProto | None = None,
        *,
        device: str | None = None,
    ) -> bool:
        """Test if a single ONNX node is compatible with an EP.

        Wraps the node in a minimal graph, attempts to create an
        InferenceSession with the target device's policy configuration.
        Reuses the session's ``_build_session_options`` for consistency.

        Args:
            node: ONNX node to test.
            graph: Optional parent graph for shape/type context.
                When provided, extracts ValueInfoProto for accurate shapes.
                Without it, uses dummy [1,1] float32 shapes (less accurate).
            device: Target device for compatibility check (e.g., "npu", "gpu",
                "cpu"). Defaults to the session's own device.

        Returns:
            True if the EP can handle this node, False otherwise.

        Note:
            This is a standalone utility, not wired into the build pipeline.
            Results are more accurate when graph is provided.
        """
        from onnx import TensorProto, helper

        target_device = device or self._device

        if graph is None:
            logger.warning(
                "is_compatible() called without graph context for node '%s'. "
                "Using dummy shapes — results may be inaccurate.",
                node.name or node.op_type,
            )

        # 1. Resolve input/output ValueInfoProto
        inputs: list[onnx.ValueInfoProto] = []
        outputs: list[onnx.ValueInfoProto] = []

        if graph is not None:
            # Build lookup from parent graph
            all_value_info: dict[str, onnx.ValueInfoProto] = {
                vi.name: vi for vi in graph.value_info
            }
            for gi in graph.input:
                all_value_info[gi.name] = gi
            for go in graph.output:
                all_value_info[go.name] = go

            for name in node.input:
                if name and name in all_value_info:
                    inputs.append(all_value_info[name])
                elif name:
                    inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1]))
            for name in node.output:
                if name and name in all_value_info:
                    outputs.append(all_value_info[name])
                elif name:
                    outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1]))
        else:
            # No graph context — use dummy shapes
            inputs.extend(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1])
                for name in node.input
                if name
            )
            outputs.extend(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1])
                for name in node.output
                if name
            )

        if not inputs or not outputs:
            return False

        # 2. Build minimal model
        try:
            test_graph = helper.make_graph([node], "compat_test", inputs, outputs)
            test_model = helper.make_model(test_graph, opset_imports=[helper.make_opsetid("", 17)])
            test_model.ir_version = 8

            # 3. Try creating session with same device policy
            sess_options = self._build_session_options(target_device)
            sess_options.log_severity_level = 4  # Suppress ORT logs during probe
            ort.InferenceSession(
                test_model.SerializeToString(),
                sess_options=sess_options,
            )
            return True
        except Exception:
            return False
