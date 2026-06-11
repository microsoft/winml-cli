# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - Core ONNX Runtime session manager."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import onnxruntime as ort

from ..core.onnx_utils import get_io_config
from ..onnx import is_compiled_onnx
from ..utils.native_stderr import capture_native_stderr, suppress_native_stderr
from .ep_registry import WinMLEPRegistry
from .stats import PerfStats


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import onnx

    from ..compiler.configs import EPConfig
    from ..utils.constants import EPName, EPNameOrAlias


logger = logging.getLogger(__name__)


class SessionState(Enum):
    """WinMLSession states."""

    INITIALIZED = "INITIALIZED"
    COMPILED = "COMPILED"
    INFERRING = "INFERRING"
    ERROR = "ERROR"


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

    @classmethod
    def _init_winml_eps_once(cls) -> None:
        """Initialize WinML EP registry once at class level."""
        if cls._eps_initialized:
            return

        try:
            registry = WinMLEPRegistry.get_instance()
            if registry.winml_available:
                with suppress_native_stderr():
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
        ep: EPNameOrAlias | None = None,
        session_options: Callable[[], ort.SessionOptions] | None = None,
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
            ep: Explicit EP short name (e.g., "migraphx", "nv_tensorrt_rtx").
                When set, bypasses policy-based selection and uses
                add_provider_for_devices to force the specific EP.
            session_options: Factory returning an ``ort.SessionOptions``.
                Called once per ``_build_session_options`` invocation so each
                ORT session gets a fresh, un-poisoned options object
                (``add_provider_for_devices`` mutates the options and cannot
                be replayed). Defaults to ``ort.SessionOptions``.
        """
        WinMLSession._init_winml_eps_once()

        self._onnx_path = Path(onnx_path)
        if not self._onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # HF Pipeline may pass torch.device; coerce to string for downstream .lower() calls
        self._device = str(device) if not isinstance(device, str) else device
        self._ep = ep
        self._persist_jit = ep_config.enable_ep_context if ep_config else False
        self._embed_context = ep_config.embed_context if ep_config else False
        self._provider_options = ep_config.provider_options if ep_config else {}

        self._session_options_factory: Callable[[], ort.SessionOptions] = (
            session_options or ort.SessionOptions
        )

        # State management
        self._state = SessionState.INITIALIZED
        self._last_error: Exception | None = None

        # Single session (one session = one EP)
        self._session: ort.InferenceSession | None = None

        # ONNX model ORT actually loads (set during compile()). May differ from
        # _onnx_path when an EPContext model is compiled or a cached one reused.
        self._running_model_path: Path | None = None

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
            logger.debug("Already compiled for %s", self._device)
            return

        target_device = self._device

        logger.debug("Compiling for device: %s", target_device)

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

        if self._persist_jit and model_path == self._onnx_path:
            # Skip ModelCompiler if input model is already compiled (EPContext)
            if is_compiled_onnx(self._onnx_path):
                logger.info("Model already compiled (EPContext), skipping ModelCompiler")
            else:
                try:
                    sess_options, resolved_device, _ = self._build_session_options(target_device)
                    model_compiler = ort.ModelCompiler(
                        sess_options,
                        str(self._onnx_path),
                        embed_compiled_data_into_model=self._embed_context,
                    )
                    with capture_native_stderr(logging.INFO):
                        model_compiler.compile_to_file(str(ctx_path))

                    # Use compiled model if it was created
                    if ctx_path.exists():
                        model_path = ctx_path
                        logger.info("Compiled to EPContext: %s", ctx_path)

                except Exception as e:
                    # Some EPs don't support compilation - fall back to original
                    logger.warning("ModelCompiler failed, using original: %s", e)

        # Record the model ORT actually loads (original or EPContext).
        self._running_model_path = model_path

        try:
            # Create InferenceSession.
            # EP is either configured via add_provider_for_devices (WinML EP
            # registry, e.g. QNN) or left to ORT's device policy (fallback).
            # Never pass providers= — WinML-registered EPs don't support it.
            sess_options, resolved_device, _ = self._build_session_options(target_device)
            with capture_native_stderr(logging.INFO):
                session = ort.InferenceSession(str(model_path), sess_options=sess_options)

        except Exception as ep_err:
            self._state = SessionState.ERROR
            self._last_error = ep_err
            raise CompilationError(
                message=f"Failed to compile for {target_device}",
                context={
                    "device": target_device,
                    "onnx_path": str(self._onnx_path),
                    "error": str(ep_err),
                },
                suggestion=self._get_compile_suggestion(target_device, ep_err),
            ) from ep_err

        # Log which providers were selected by ORT (based on policy)
        actual_providers = session.get_providers()
        logger.info(
            "Session created with device %s, providers: %s",
            target_device,
            actual_providers,
        )

        # Store session
        self._session = session
        self._state = SessionState.COMPILED

        # Resolve device label from the primary provider ORT actually selected
        if self._device == "auto" and actual_providers:
            self._device = resolved_device
            logger.info("Auto-resolved device: %s", self._device)

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

    def _build_session_options(self, device: str) -> tuple[ort.SessionOptions, str, EPName]:
        """Build ORT SessionOptions from the session_options factory and device.

        Returns a **fresh** SessionOptions produced by ``self._session_options_factory``
        on every call so each ORT session gets an un-poisoned options object —
        ``add_provider_for_devices`` cannot be replayed on the same instance.
        """
        from ..sysinfo.device import resolve_device, resolve_eps
        from ..utils.constants import DEVICE_TO_DEVICE_TYPE, normalize_ep_name
        from ..winml import add_ep_for_device

        resolved_device, _ = resolve_device(device, ep=self._ep)
        resolved_ep = normalize_ep_name(self._ep) if self._ep else resolve_eps(resolved_device)[0]
        device_type = DEVICE_TO_DEVICE_TYPE.get(resolved_device.upper())

        opts = self._session_options_factory()
        if add_ep_for_device(opts, resolved_ep, device_type, self._provider_options):
            logger.info(
                "Built SessionOptions with EP: %s (%s) device=%s -> %s",
                resolved_ep,
                self._ep,
                device,
                resolved_device,
            )
            return opts, resolved_device, resolved_ep

        # Defensive: should not be reachable unless the EP enumeration races.
        raise DeviceNotAvailableError(
            message=f"No available provider found for device '{device}' and EP '{self._ep}'"
        )

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
            "npu": "Install onnxruntime-windowsml",
            "gpu": "Install onnxruntime-windowsml",
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
    def ep_name(self) -> EPName | None:
        """Primary EP ORT actually bound, or None before compile.

        Returns ``session.get_providers()[0]`` — the EP that owns node
        partitioning. ``CPUExecutionProvider`` may still appear later
        in the list as ORT's automatic fallback for unsupported ops.
        """
        if self._session is None:
            return None
        providers = self._session.get_providers()
        return cast("EPName", providers[0]) if providers else None

    @property
    def is_compiled(self) -> bool:
        """Check if session is compiled."""
        return self._session is not None

    @property
    def running_model_path(self) -> Path:
        """Path to the ONNX model ORT actually loads.

        May differ from the input ``onnx_path`` when an EPContext model is
        compiled or a cached one is reused. Falls back to the input path
        before ``compile()`` runs.
        """
        return self._running_model_path or self._onnx_path

    @property
    def perf_stats(self) -> PerfStats | None:
        """Performance statistics (None if not in perf() context).

        Returns:
            PerfStats instance with timing data, or None if outside perf() context.
        """
        return self._perf_stats

    @contextmanager
    def perf(self, warmup: int = 0) -> Generator[PerfStats, None, None]:
        """Context manager for scoped performance tracking.

        Args:
            warmup: Number of initial samples to exclude from statistics.

        Yields:
            PerfStats instance that collects timing data within the context.

        Example:
            >>> with session.perf(warmup=10) as stats:
            ...     for _ in range(110):
            ...         session.run(inputs)
            >>> print(f"P99: {stats.p99_ms:.2f} ms")  # Based on last 100 samples
        """
        self._perf_stats = PerfStats(warmup=warmup)
        try:
            yield self._perf_stats
        finally:
            self._perf_stats = None

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
                - precision: best-effort precision label (e.g. "fp16",
                  "int8", "w8a16"), or ``None`` when no signal could be
                  derived from the graph
        """
        if self._io_config is None:
            from ..onnx import load_onnx

            model = load_onnx(self._onnx_path, load_weights=False, validate=False)
            self._io_config = get_io_config(model)
            # Enrich with value_range from build config if available
            self._io_config["input_value_ranges"] = self._load_input_value_ranges()
            self._io_config["precision"] = self._get_precision(model)
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

    @staticmethod
    def _get_precision(model_proto: onnx.ModelProto) -> str | None:
        """Best-effort estimate of a model's numeric precision.

        Returns one of: ``"fp32"``, ``"fp16"``, ``"bf16"``, ``"int4"``,
        ``"int8"``, ``"int16"``, ``"w{w}a{a}"`` (mixed), or ``None`` when
        no signal can be derived.

        Detection is purely operator-schema-based (no model-architecture
        or naming assumptions). The ladder, first match wins:

        1. QDQ (``QuantizeLinear`` / ``DequantizeLinear``): dominant
           ``zero_point`` initializer bit width per side. A pair is
           weight-side when its source tensor is an initializer,
           activation-side otherwise.
        2. Block-wise quant (``MatMulNBits`` / ``GatherBlockQuantized``):
           schema ``bits`` attribute + dominant float bit width for
           activations → ``w{w}a{a}``.
        3. No quant markers → dominant float dtype among initializers.
        4. No signal → ``None``.
        """
        from onnx import TensorProto

        graph = model_proto.graph
        init_dtypes: dict[str, int] = {init.name: init.data_type for init in graph.initializer}
        init_names = set(init_dtypes)
        op_types = {n.op_type for n in graph.node}

        int_bits: dict[int, int] = {
            int(TensorProto.UINT4): 4,
            int(TensorProto.INT4): 4,
            int(TensorProto.UINT8): 8,
            int(TensorProto.INT8): 8,
            int(TensorProto.UINT16): 16,
            int(TensorProto.INT16): 16,
            int(TensorProto.UINT32): 32,
            int(TensorProto.INT32): 32,
        }

        def _label(w_bits: int, a_bits: int) -> str:
            return f"w{w_bits}a{a_bits}"

        # (1) QDQ — dominant zero_point bit width per side.
        if op_types & {"QuantizeLinear", "DequantizeLinear"}:
            weight_counts: dict[int, int] = {}
            act_counts: dict[int, int] = {}
            for node in graph.node:
                if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
                    continue
                if len(node.input) < 3:
                    continue
                zp_dtype = init_dtypes.get(node.input[2])
                if zp_dtype is None:
                    continue
                bits = int_bits.get(zp_dtype)
                if bits is None:
                    continue
                is_weight_side = node.input[0] in init_names
                # 32-bit zero_points on initializer-input DQs are bias
                # accumulators (standard for INT8 QDQ: INT8 weights, INT32
                # bias). They shouldn't drive the weight precision label.
                if is_weight_side and bits >= 32:
                    continue
                target = weight_counts if is_weight_side else act_counts
                target[bits] = target.get(bits, 0) + 1

            if weight_counts or act_counts:
                w = (
                    max(weight_counts, key=lambda k: weight_counts[k])
                    if weight_counts
                    else max(act_counts, key=lambda k: act_counts[k])
                )
                a = max(act_counts, key=lambda k: act_counts[k]) if act_counts else w
                return _label(w, a)

        # (2) Block-wise quantization carries a schema-defined `bits` attr.
        nbits: set[int] = set()
        for node in graph.node:
            if node.op_type in ("MatMulNBits", "GatherBlockQuantized"):
                for attr in node.attribute:
                    if attr.name == "bits":
                        nbits.add(attr.i)
        if nbits:
            w_bits = min(nbits)
            a_bits = WinMLSession._dominant_float_bits(graph) or 16
            return _label(w_bits, a_bits)

        # (3) Float-only model — dominant initializer dtype.
        dom = WinMLSession._dominant_float_bits(graph)
        if dom == 32:
            return "fp32"
        if dom == 16:
            has_bf16 = any(init.data_type == TensorProto.BFLOAT16 for init in graph.initializer)
            has_fp16 = any(init.data_type == TensorProto.FLOAT16 for init in graph.initializer)
            if has_bf16 and not has_fp16:
                return "bf16"
            return "fp16"

        # (4) No signal.
        return None

    @staticmethod
    def _dominant_float_bits(graph: onnx.GraphProto) -> int | None:
        """Return 32 or 16 — whichever float dtype dominates initializer count.

        ``None`` if no float initializers are present.
        """
        from onnx import TensorProto

        counts: dict[int, int] = {}
        for init in graph.initializer:
            if init.data_type in (
                TensorProto.FLOAT,
                TensorProto.FLOAT16,
                TensorProto.BFLOAT16,
            ):
                counts[init.data_type] = counts.get(init.data_type, 0) + 1
        if not counts:
            return None
        dominant = max(counts, key=lambda k: counts[k])
        return 32 if dominant == TensorProto.FLOAT else 16

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
            sess_options, _, _ = self._build_session_options(target_device)
            sess_options.log_severity_level = 4  # Suppress ORT logs during probe
            ort.InferenceSession(
                test_model.SerializeToString(),
                sess_options=sess_options,
            )
            return True
        except Exception:
            return False
