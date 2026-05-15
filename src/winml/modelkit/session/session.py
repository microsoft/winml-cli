# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - Core ONNX Runtime session manager."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import onnxruntime as ort

from ..core.onnx_utils import get_io_config
from ..onnx import is_compiled_onnx
from .ep_registry import WinMLEPRegistry
from .stats import PerfStats


if TYPE_CHECKING:
    from collections.abc import Generator

    import onnx

    from ..compiler.configs import EPConfig
    from ..utils.constants import EPName, EPNameOrAlias


logger = logging.getLogger(__name__)


# WORKAROUND: Suppress compatibility noise printed during EP DLL registration.
#
# Symptom: two lines appear on every run even when the EP is unused:
#   "The requested API version [24] is not available, only API versions
#    [1, 23] are supported in this build. Current ORT Version is: 1.23.5"
#
# Root cause: the WinApp SDK 2.0 EP DLLs were built against ORT API v24, but
# the currently bundled WinML runtime is still v1.8 (ORT 1.23.5, API v23).
# The DLL prints this mismatch warning to native stderr during registration.
# Functionality is NOT affected — ORT falls back cleanly to an available EP.
#
# Fix: upgrade the WinML runtime to 2.0. Remove this workaround once that
# upgrade lands and the API version mismatch is resolved.
#
# Technical note: the DLL writes via Win32 GetStdHandle(STD_ERROR_HANDLE)
# rather than the CRT fd table, so os.dup2 alone is not sufficient on
# Windows — SetStdHandle must also be updated.
@contextmanager
def _suppress_ep_registration_stderr():
    """Suppress native stderr during EP DLL registration (Win32 + CRT)."""
    null_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    # Capture the Win32 handle BEFORE os.dup2 changes STD_ERROR_HANDLE.
    # os.dup2(null_fd, 2) on Windows calls SetStdHandle internally, so reading
    # GetStdHandle after the redirect would return the devnull handle, not the
    # original — making the later restore a no-op.
    old_w32 = None
    if sys.platform == "win32":
        import ctypes
        import msvcrt

        k32 = ctypes.WinDLL("kernel32")
        _std_err = ctypes.c_uint32(0xFFFFFFF4)
        old_w32 = k32.GetStdHandle(_std_err)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    if sys.platform == "win32" and old_w32 is not None:
        k32.SetStdHandle(_std_err, msvcrt.get_osfhandle(2))
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        if old_w32 is not None:
            k32.SetStdHandle(_std_err, old_w32)


@contextmanager
def _suppress_native_output(log_path: str | Path | None = None, suppress_stderr: bool = False):
    """Redirect native stdout (and optionally stderr) to a log file (or devnull).

    QNN SDK compiler writes progress to stdout via native C++ code that
    Python logging/warnings cannot intercept. By default only redirects
    stdout — stderr is left untouched so Rich displays and Python logging work.
    Pass suppress_stderr=True to also redirect stderr to the same destination.
    """
    if log_path is not None:
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    else:
        fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    os.dup2(fd, 1)
    old_stderr = None
    if suppress_stderr:
        old_stderr = os.dup(2)
        os.dup2(fd, 2)
    os.close(fd)
    try:
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(old_stdout)
        if old_stderr is not None:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)


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
                with _suppress_ep_registration_stderr():
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
            ep: Explicit EP short name (e.g., "migraphx", "nv_tensorrt_rtx").
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
        self._ep = ep if ep else None
        self._persist_jit = ep_config.enable_ep_context if ep_config else False
        self._embed_context = ep_config.embed_context if ep_config else False
        self._provider_options = ep_config.provider_options if ep_config else {}

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
            # Create InferenceSession.
            # EP is either configured via add_provider_for_devices (WinML EP
            # registry, e.g. QNN) or left to ORT's device policy (fallback).
            # Never pass providers= — WinML-registered EPs don't support it.
            sess_options = self._build_session_options(target_device)
            with _suppress_native_output(compile_log):
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
        explicitly bind that EP — including ``"cpu"``, so the
        CPUExecutionProvider isn't silently displaced by another CPU-capable
        EP (e.g. OpenVINO) under PREFER_CPU policy.
        When ``self._ep`` is not set, the path forks on device: ``"cpu"``
        falls through to PREFER_CPU policy (skipping EP discovery so non-CPU
        EPs aren't probed), while other devices query ``get_ep_devices()``
        to discover an available EP. Policy-based selection is the
        last-resort fallback.

        Note: Returns a **fresh** SessionOptions when using explicit EP to
        avoid "already registered" errors from repeated calls.
        """
        # CPU never needs EP binding — skip device discovery entirely so that
        # non-CPU EPs (e.g. OpenVINO) are not probed via get_ep_devices(),
        # which would trigger their native shared-library load and emit
        # version-mismatch warnings even when the model runs on CPU.
        # Exception: when an explicit EP is set (e.g. --ep openvino --device cpu,
        # or --ep cpu --device cpu), fall through so the EP binding logic
        # below can honour it.
        if device.lower() == "cpu" and not self._ep:
            opts = self._session_options
            opts.set_provider_selection_policy(DEVICE_POLICY_MAP["cpu"])
            logger.info("Using PREFER_CPU policy for device cpu")
            return opts

        # Explicit EP targeting: create fresh opts to avoid double-registration.
        # When device is also specified (non-"auto"), narrow by both EP name
        # and device type so e.g. `--ep qnn --device cpu` finds QNN-on-CPU
        # instead of the first QNN ep_device (which may report as NPU).
        # `--ep cpu` is honoured here too so the CPUExecutionProvider gets
        # bound explicitly; otherwise PREFER_CPU policy lets ORT prefer
        # OV-on-CPU (or any other registered CPU-capable EP) over the basic
        # CPU EP, silently ignoring the user's --ep choice.
        if self._ep:
            from ..utils.constants import normalize_ep_name

            target_name = normalize_ep_name(self._ep)
            if target_name:
                matched = self._find_ep_device(ep_name=target_name, device=device)
                if matched:
                    from ..utils.constants import DEVICE_TYPE_TO_DEVICE

                    opts = ort.SessionOptions()
                    opts.add_provider_for_devices([matched], self._provider_options)
                    resolved = DEVICE_TYPE_TO_DEVICE.get(
                        matched.device.type, str(matched.device.type)
                    )
                    logger.info(
                        "Explicit EP: %s (%s) device=%s -> %s",
                        self._ep,
                        target_name,
                        device,
                        resolved,
                    )
                    return opts
                logger.warning(
                    "EP '%s' (%s) not found for device '%s'",
                    self._ep,
                    target_name,
                    device,
                )

        # No explicit EP — discover available EP for this device type
        if not self._ep and device.lower() != "cpu":
            matched = self._find_ep_device(device=device)
            if matched:
                opts = ort.SessionOptions()
                opts.add_provider_for_devices([matched], self._provider_options)
                logger.info("Discovered EP for %s: %s", device, matched.ep_name)
                return opts

        # Policy-based selection (last resort)
        opts = self._session_options
        policy = DEVICE_POLICY_MAP.get(
            device.lower(), ort.OrtExecutionProviderDevicePolicy.PREFER_NPU
        )
        opts.set_provider_selection_policy(policy)
        logger.info("Using provider selection policy %s for device %s", policy, device)

        return opts

    @staticmethod
    def _find_ep_device(device: str, ep_name: EPName | None = None) -> Any:
        """Find the first OrtEpDevice matching the given filters.

        Behavior:
            - ``ep_name`` set, ``device == "auto"`` → first ep_device
              matching ``ep_name`` (or None).
            - ``ep_name`` unset, ``device == "auto"`` → ``None`` (no
              effective filter — refuse to pick an arbitrary ep_device).
            - ``ep_name`` unset, ``device`` is a concrete type → first
              ep_device matching that device type (or None).
            - Both set → ep_device must satisfy both (or None).

        Note: Selection order is determined by the ORT EP registry, which is
        not part of any documented contract. On systems where multiple EPs
        match the same device type (e.g., QNN and DML both appear as GPU),
        a device-only query returns the first one in registry order. Pass
        ``ep_name`` to disambiguate.

        Args:
            device: Device policy ("cpu", "gpu", "npu", "auto"). ``"auto"``
                and unknown strings act as no-op device filters.
            ep_name: Full EP name (e.g., "DmlExecutionProvider"), or None
                to skip EP-name filtering.

        Returns:
            The matching OrtEpDevice, or None if not found.
        """
        from ..utils.constants import DEVICE_TO_DEVICE_TYPE

        device_type = DEVICE_TO_DEVICE_TYPE.get(device.upper())

        # No effective filter — refuse to pick an arbitrary ep_device.
        if not ep_name and device_type is None:
            return None

        for ep_dev in ort.get_ep_devices():
            if ep_name and ep_dev.ep_name != ep_name:
                continue
            if device_type is not None and ep_dev.device.type != device_type:
                continue
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
