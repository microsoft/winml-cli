# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - Core ONNX Runtime session manager."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort

from ..core.onnx_utils import get_io_config
from ..onnx import is_compiled_onnx
from .ep_device import (
    WinMLEPMonitorMismatch,
    expand_ep_name,
    lookup_device_spec,
)
from .monitor.ep_monitor import WinMLEPMonitor
from .stats import PerfStats


if TYPE_CHECKING:
    import onnx

    from ..compiler.configs import EPConfig
    from .ep_registry import WinMLEPDevice


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
    """Per-perf-window stats container yielded by ``WinMLSession.perf()``.

    Aggregates perf statistics and the optional attached EP monitor.
    Frozen: mutation is not a supported pattern — update the underlying
    objects instead.
    """

    stats: PerfStats
    monitor: WinMLEPMonitor  # NullEPMonitor when no monitor was passed


def _ep_defaults(ep_device: WinMLEPDevice) -> dict[str, str]:
    """EP-specific defaults from the EPDeviceSpec catalog.

    Most EPs return {} — they pick up settings via ep_config.provider_options
    and ep_monitor.get_provider_options(). Only EPs that have measured
    default_provider_options in EP_DEVICE_SPECS contribute non-empty results.

    Note: QNNExecutionProvider does NOT need ``backend_type`` here.
    When using ``add_provider_for_devices()``, the OrtEpDevice handle already
    encodes the backend target (NPU→HTP, GPU→GPU, CPU→CPU). Passing
    ``backend_type`` explicitly crashes ORT 1.23.5 with a native exit 127.

    Returns a fresh dict copy so callers can mutate without aliasing the
    catalog entry's immutable Mapping.
    """
    spec = lookup_device_spec(ep_device.device.ep_name, ep_device.device.device_type.lower())
    return dict(spec.default_provider_options) if spec else {}


def _build_provider_options(
    ep_device: WinMLEPDevice,
    ep_config: EPConfig | None,
    ep_monitor: WinMLEPMonitor | None,
) -> dict[str, str]:
    """Flat provider_options for add_provider_for_devices().

    Three layers, each overrides the previous:
      1. EP-specific defaults from ep_device (e.g. QNN backend_type).
      2. User overrides from ep_config.provider_options.
      3. WinMLEPMonitor-required options (e.g. QNN profiling_level).

    Monitor wins last because tracing correctness depends on its options
    actually reaching the EP. Callers who want to disable tracing should
    drop the monitor, not override its keys.
    """
    options: dict[str, str] = _ep_defaults(ep_device)
    if ep_config is not None and getattr(ep_config, "provider_options", None):
        options.update(ep_config.provider_options)
    if ep_monitor is not None:
        options.update(ep_monitor.get_provider_options())
    return options


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


def _build_session_options(
    ep_device: WinMLEPDevice,
    ep_config: EPConfig | None = None,
    ep_monitor: WinMLEPMonitor | None = None,
    base_session_options: ort.SessionOptions | None = None,
) -> ort.SessionOptions:
    """Build a fully-bound ort.SessionOptions for one WinMLEPDevice pair.

    Free function (not a method): pure inputs -> pure outputs. The caller
    (typically :meth:`WinMLEPRegistry.auto_device`) has already resolved
    the (source, device) pair, so no registry / handle filtering happens
    here.
    """
    so = base_session_options if base_session_options is not None else ort.SessionOptions()

    if ep_monitor is not None:
        for key, value in ep_monitor.get_session_options().items():
            so.add_session_config_entry(key, value)

    handle = ep_device.device._ort
    options = _build_provider_options(ep_device, ep_config, ep_monitor)
    so.add_provider_for_devices([handle], options)
    return so


class WinMLSession:
    """ONNX Runtime session bound to one resolved :class:`WinMLEPDevice`."""

    def __init__(
        self,
        onnx_path: str | Path,
        ep_device: WinMLEPDevice,
        *,
        ep_config: EPConfig | None = None,
        ep_monitor: WinMLEPMonitor | None = None,
        base_session_options: ort.SessionOptions | None = None,
    ) -> None:
        """Initialize WinMLSession.

        Args:
            onnx_path: Path to ONNX model.
            ep_device: Fully-resolved (source, device) pair. Construct via
                :meth:`WinMLEPRegistry.auto_device` after calling
                :func:`resolve_device`; tests may construct directly with a
                stub :class:`WinMLEP` + :class:`WinMLDevice`.
            ep_config: Optional EP configuration (provider_options, etc.).
            ep_monitor: Optional monitor. When passed, its session-config
                entries are threaded into the initial
                :func:`_build_session_options` call.
            base_session_options: ORT SessionOptions base. If None, creates default.
        """
        self._onnx_path = Path(onnx_path)
        self._ep_device = ep_device
        self._ep_config = ep_config
        self._ep_monitor = ep_monitor
        self._base_session_options = base_session_options

        # Snapshots preserved across perf() entry/exit (see perf()).
        self._provider_options: dict[str, str] = _build_provider_options(
            ep_device, ep_config, ep_monitor
        )
        self._active_session_option_entries: dict[str, str] = {}
        # Convenience: the canonical EP name from the chosen handle.
        self._ep: str = ep_device.device.ep_name

        # Derived convenience attributes consumed by compile(), device property, etc.
        self._device: str = ep_device.device.device_type.lower()
        self._persist_jit: bool = ep_config.enable_ep_context if ep_config else False
        self._embed_context: bool = ep_config.embed_context if ep_config else False

        # _session is None until InferenceSession construction completes; __del__
        # reads this attribute, so it must exist before any call that could raise.
        self._session: ort.InferenceSession | None = None

        # State management
        self._state = SessionState.INITIALIZED
        self._last_error: Exception | None = None

        # Cached I/O metadata (lazy-loaded)
        self._io_config: dict | None = None

        # Performance tracking (enabled via perf() context manager)
        self._perf_stats: PerfStats | None = None

        # Compile workflows defer session creation to compile(); runtime workflows
        # create the session eagerly here.
        if not self._persist_jit:
            so = _build_session_options(
                self._ep_device,
                self._ep_config,
                ep_monitor,
                self._base_session_options,
            )
            self._session = ort.InferenceSession(self._onnx_path, sess_options=so)
            _dev = self._ep_device.device
            logger.info(
                "ort.InferenceSession: ep=%s device=%s hardware=%r providers=%s",
                _dev.ep_name,
                _dev.device_type,
                _dev.hardware_name,
                self._session.get_providers(),
            )

    def compile(self) -> None:
        """Compile model for target device using ModelCompiler API.

        Only compiles once per session (idempotent).
        Device is immutable - set at __init__ time.

        For compile workflows (ep_config.enable_ep_context=True) this method
        runs ort.ModelCompiler.compile_to_file() to produce a .ctx.onnx, then
        creates the runtime InferenceSession against that compiled artifact.
        For runtime-only workflows (persist_jit=False) this is a no-op if the
        session was already created eagerly in __init__.
        """
        # If already compiled, ignore (idempotent)
        if self._session is not None:
            if self._is_verbose():
                logger.info("Already compiled for %s", self._device)
            return

        target_device = self._device

        if self._is_verbose():
            logger.info("Compiling for device: %s", target_device)

        # Derive the output ctx path from the original model path.
        ctx_path = self._onnx_path.parent / f"{self._onnx_path.stem}_{target_device}_ctx.onnx"
        model_path = self._onnx_path

        # Native QNN SDK compiler writes progress to stdout/stderr;
        # redirect to log file to keep the console clean.
        compile_log = self._onnx_path.parent / "compile.log"

        # Check for existing fresh EPContext (skip re-compile if cache is fresh).
        if ctx_path.exists() and ctx_path.stat().st_mtime >= self._onnx_path.stat().st_mtime:
            model_path = ctx_path
            logger.info("Using cached EPContext: %s", ctx_path)
        elif is_compiled_onnx(self._onnx_path):
            # Input model is already an EPContext — use it directly.
            logger.info("Model already compiled (EPContext), skipping ModelCompiler")
        else:
            # AOT compile to .ctx.onnx via ort.ModelCompiler.
            try:
                so = _build_session_options(
                    self._ep_device,
                    self._ep_config,
                    None,  # no monitor at compile time
                    self._base_session_options,
                )
                model_compiler = ort.ModelCompiler(
                    so,
                    str(self._onnx_path),
                    embed_compiled_data_into_model=self._embed_context,
                )
                with _suppress_native_output(compile_log):
                    model_compiler.compile_to_file(str(ctx_path))

                if ctx_path.exists():
                    model_path = ctx_path
                    logger.info("Compiled to EPContext: %s", ctx_path)

            except Exception as e:
                # Some EPs don't support compilation — fall back to original model.
                logger.warning("ModelCompiler failed, using original: %s", e)

        try:
            # Create the runtime InferenceSession against the (possibly compiled) model.
            runtime_so = _build_session_options(
                self._ep_device,
                self._ep_config,
                None,
                self._base_session_options,
            )
            with _suppress_native_output(compile_log):
                session = ort.InferenceSession(str(model_path), sess_options=runtime_so)

            actual_providers = session.get_providers()
            logger.info(
                "Session created for device %s, providers: %s",
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

        self._session = session
        self._state = SessionState.COMPILED

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

    @staticmethod
    def _build_op_type_map(onnx_path: Path | None) -> dict[str, str]:
        """Build a ``node.name -> node.op_type`` map from an ONNX file.

        Returns an empty dict on any failure (None path, missing file,
        corrupt ONNX, missing ``onnx`` package). Op-tracing monitors that
        receive an empty map fall through their fallback chain to
        EP-authoritative or heuristic sources.

        Used by :meth:`perf` to inject the map into op-tracing monitors
        via :meth:`WinMLEPMonitor.set_onnx_op_types`.
        """
        if onnx_path is None:
            return {}
        try:
            import onnx as _onnx

            model = _onnx.load(str(onnx_path), load_external_data=False)
            return {n.name: n.op_type for n in model.graph.node if n.name and n.op_type}
        except Exception as e:
            # Defensive: any exception during ONNX load (missing file,
            # corrupt protobuf, missing onnx package) returns empty.
            # Logged at DEBUG; non-op-tracing path doesn't care.
            logger.debug("Could not load ONNX op-type map from %s: %s", onnx_path, e)
            return {}

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
        monitor: WinMLEPMonitor | None = None,
    ):
        """Context manager for a scoped perf window.

        Yields a :class:`PerfContext` whose ``stats`` property accumulates
        timing from every :meth:`run` call made inside the ``with`` block.
        The optional *monitor* is entered/exited around the body.

        Session setup lifecycle
        -----------------------
        * If *monitor* contributes provider_options **and** a compiled session
          already exists, the compiled session is torn down first (auto-reset
          with a WARNING) so the new options take effect.
        * After the ``with`` block exits a bare (no-monitor) InferenceSession is
          rebuilt so subsequent :meth:`run` calls see the baseline configuration.

        Teardown ordering (C-2 invariant)
        ----------------------------------
        * For monitors with ``requires_session_teardown=True`` (e.g. QNNMonitor
          which flushes CSV only on session destroy), :meth:`reset` fires
          *before* ``monitor.__exit__`` so the flushed data is available inside
          ``__exit__``.
        * For other monitors the session is rebuilt *after* ``monitor.__exit__``.

        Args:
            warmup: Number of initial :meth:`run` calls to exclude from stats.
            monitor: Optional EP-specific monitor.  ``NullEPMonitor`` is used
                when *monitor* is ``None`` so callers need no null checks.

        Yields:
            :class:`PerfContext` with ``stats`` (a :class:`PerfStats`) and
            ``monitor`` (the effective :class:`WinMLEPMonitor`).

        Raises:
            RuntimeError: If a perf window is already active (re-entry guard).
            WinMLEPMonitorMismatch: If *monitor* targets a different EP than this session.
        """
        from .monitor.ep_monitor import NullEPMonitor

        if self._perf_stats is not None:
            raise RuntimeError(
                "WinMLSession.perf() is already active. Nested perf windows are not supported."
            )

        effective_monitor: WinMLEPMonitor = monitor if monitor is not None else NullEPMonitor()

        if (
            monitor is not None
            and monitor.ep_name is not None
            and expand_ep_name(monitor.ep_name) != self._ep
        ):
            raise WinMLEPMonitorMismatch(
                f"Monitor ep_name={monitor.ep_name!r} expands to "
                f"{expand_ep_name(monitor.ep_name)!r}, but session is bound "
                f"to {self._ep!r}. Monitor and session must agree."
            )

        # Build merged provider_options for this perf window.
        new_prov = _build_provider_options(self._ep_device, self._ep_config, monitor)

        # Auto-reset: if session is compiled AND monitor contributes options that
        # differ from the current provider_options, tear down the compiled session
        # so the new options take effect when a fresh InferenceSession is created.
        if self._session is not None and new_prov != self._provider_options:
            logger.warning(
                "auto-resetting compiled session to apply monitor session/provider options"
            )
            self.reset()

        # Snapshot state for restore-on-exit.
        saved_sess_entries = dict(self._active_session_option_entries)
        saved_prov = dict(self._provider_options)
        saved_ep = self._ep

        # Inject the ONNX op-type map into the monitor *before* __enter__ so
        # op-tracing monitors can prepare their state on the map.
        effective_monitor.set_onnx_op_types(self._build_op_type_map(self._onnx_path))

        # Activate PerfStats for this window.
        stats = PerfStats(warmup=warmup)
        self._perf_stats = stats

        # Rebuild InferenceSession only when monitor-contributed options differ
        # from the current session's options (i.e. a new session is needed).
        # Track whether we rebuilt so the teardown path knows whether to restore.
        _session_rebuilt = new_prov != self._provider_options or self._session is None
        if _session_rebuilt:
            self._provider_options = new_prov
            so = _build_session_options(
                self._ep_device,
                self._ep_config,
                monitor,
                self._base_session_options,
            )
            self._session = ort.InferenceSession(self._onnx_path, sess_options=so)

        ctx = PerfContext(stats=stats, monitor=effective_monitor)

        # Enter the monitor manually so we can control teardown order (C-2
        # invariant: requires_session_teardown monitors need self.reset() to
        # fire BEFORE monitor.__exit__).
        try:
            effective_monitor.__enter__()
        except Exception:
            # __enter__ failed — restore state and do NOT call __exit__.
            self._active_session_option_entries = saved_sess_entries
            self._provider_options = saved_prov
            self._ep = saved_ep
            self._perf_stats = None
            if _session_rebuilt:
                self._session = ort.InferenceSession(
                    self._onnx_path,
                    sess_options=_build_session_options(
                        self._ep_device,
                        self._ep_config,
                        None,
                        self._base_session_options,
                    ),
                )
            raise

        exc_info = (None, None, None)
        try:
            yield ctx
        except BaseException:
            import sys

            exc_info = sys.exc_info()
        finally:
            # C-2: for monitors that require session teardown, reset() BEFORE
            # monitor.__exit__ so the flushed data is available in __exit__.
            if getattr(effective_monitor, "requires_session_teardown", False):
                self.reset()

            # Call monitor.__exit__ — propagate exc_info so monitor sees the
            # exception (exception transparency contract).
            try:
                effective_monitor.__exit__(*exc_info)
            except Exception:
                pass  # monitor __exit__ errors do not override body exceptions

            # Restore snapshots.
            self._active_session_option_entries = saved_sess_entries
            self._provider_options = saved_prov
            self._ep = saved_ep
            self._perf_stats = None

            # Rebuild baseline session only when we created a new session at
            # the start of perf() (i.e. _session_rebuilt=True).  When the
            # monitor contributed no options we reused the existing session —
            # no teardown/rebuild needed (preserves the pre-perf InferenceSession
            # object identity, which tests assert on).
            if _session_rebuilt and self._session is not None:
                self._session = ort.InferenceSession(
                    self._onnx_path,
                    sess_options=_build_session_options(
                        self._ep_device,
                        self._ep_config,
                        None,
                        self._base_session_options,
                    ),
                )

            # Re-raise any exception from the body.
            if exc_info[1] is not None:
                raise exc_info[1].with_traceback(exc_info[2])

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
    ) -> bool:
        """Test if a single ONNX node is compatible with an EP.

        Wraps the node in a minimal graph, attempts to create an
        InferenceSession with the session's EPDeviceTarget binding.

        Args:
            node: ONNX node to test.
            graph: Optional parent graph for shape/type context.
                When provided, extracts ValueInfoProto for accurate shapes.
                Without it, uses dummy [1,1] float32 shapes (less accurate).

        Returns:
            True if the EP can handle this node, False otherwise.

        Note:
            This is a standalone utility, not wired into the build pipeline.
            Results are more accurate when graph is provided.
        """
        from onnx import TensorProto, helper

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

            # 3. Try creating session with same EPDeviceTarget binding
            sess_options = _build_session_options(
                self._ep_device,
                self._ep_config,
                None,
                self._base_session_options,
            )
            sess_options.log_severity_level = 4  # Suppress ORT logs during probe
            ort.InferenceSession(
                test_model.SerializeToString(),
                sess_options=sess_options,
            )
            return True
        except Exception:
            return False
