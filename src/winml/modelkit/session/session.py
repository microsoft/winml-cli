# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - Core ONNX Runtime session manager."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort
from filelock import FileLock, Timeout
from google.protobuf.message import DecodeError

from ..core.onnx_utils import get_io_config
from ..onnx import get_onnx_model_hash, is_compiled_onnx
from ..utils.native_stderr import (
    get_win32_fd_handle,
    get_win32_std_handle,
    native_fd_redirect_lock,
    refresh_click_windows_console_stream,
    restore_redirected_fd,
    set_win32_std_handle,
    set_win32_std_handle_to_current_fd,
)
from .ep_device import (
    WinMLEPMonitorMismatch,
    expand_ep_name,
    lookup_device_spec,
)
from .monitor.ep_monitor import WinMLEPMonitor
from .stats import PerfStats


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import TracebackType

    from onnx import GraphProto, ModelProto, NodeProto, ValueInfoProto

    from ..compiler.configs import EPConfig
    from .ep_registry import WinMLEPDevice


logger = logging.getLogger(__name__)

_EPCONTEXT_THREAD_LOCKS_GUARD = threading.Lock()
_EPCONTEXT_THREAD_LOCKS: dict[Path, threading.Lock] = {}
_EPCONTEXT_CACHE_MAX_GENERATIONS = 4


def _epcontext_thread_lock(lock_path: Path) -> threading.Lock:
    """Return the process-local lock paired with one EPContext lockfile."""
    resolved = lock_path.resolve(strict=False)
    with _EPCONTEXT_THREAD_LOCKS_GUARD:
        return _EPCONTEXT_THREAD_LOCKS.setdefault(resolved, threading.Lock())


@dataclass
class _EPContextCacheLease:
    """Held cache lock that can be transferred from compile selection to runtime open."""

    lock_path: Path
    thread_lock: threading.Lock
    file_lock: FileLock
    _released: bool = False

    @classmethod
    def acquire(cls, lock_path: Path, *, blocking: bool = True) -> _EPContextCacheLease | None:
        thread_lock = _epcontext_thread_lock(lock_path)
        if not thread_lock.acquire(blocking=blocking):
            return None
        file_lock = FileLock(lock_path)
        try:
            if blocking:
                file_lock.acquire()
            else:
                file_lock.acquire(timeout=0)
        except Timeout:
            thread_lock.release()
            return None
        except Exception:
            thread_lock.release()
            raise
        return cls(lock_path=lock_path, thread_lock=thread_lock, file_lock=file_lock)

    def release(self) -> None:
        if self._released:
            return
        try:
            self.file_lock.release()
        finally:
            self.thread_lock.release()
            self._released = True

    def __enter__(self) -> _EPContextCacheLease:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.release()


@dataclass
class _PreparedEPContextModel:
    """Model path selected for runtime loading plus any lock held until ORT opens it."""

    path: Path
    markerless: bool = False
    lease: _EPContextCacheLease | None = None

    @property
    def lock_path(self) -> Path | None:
        return self.lease.lock_path if self.lease is not None else None

    def release(self) -> None:
        if self.lease is not None:
            self.lease.release()

    def __enter__(self) -> _PreparedEPContextModel:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.release()


@contextmanager
def _suppress_native_output(log_path: str | Path | None = None) -> Iterator[None]:
    """Redirect native stdout to a log file (or devnull) for the block.

    QNN SDK's compiler writes progress via native C++ stdout that Python's
    logging can't intercept. Only stdout — stderr is left alone so Rich
    displays and Python logging still work.
    """
    with native_fd_redirect_lock():
        fd: int | None = None
        old_stdout: int | None = None
        old_win32_stdout = get_win32_std_handle(1)
        old_fd_stdout = get_win32_fd_handle(1)
        try:
            if log_path is not None:
                fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            else:
                fd = os.open(os.devnull, os.O_WRONLY)
            old_stdout = os.dup(1)
            os.dup2(fd, 1)
        except OSError:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    logger.debug("Could not close native stdout suppression fd", exc_info=True)
            if old_stdout is not None:
                try:
                    os.close(old_stdout)
                except OSError:
                    logger.debug("Could not close saved native stdout fd", exc_info=True)
            logger.debug(
                "Native stdout suppression setup failed; leaving stdout unchanged",
                exc_info=True,
            )
            yield
            return

        assert old_stdout is not None
        try:
            os.close(fd)
        except OSError:
            logger.debug("Could not close native stdout suppression fd", exc_info=True)
        set_win32_std_handle_to_current_fd(1)
        try:
            yield
        finally:
            restore_redirected_fd(1, old_stdout)
            if old_win32_stdout is not None and old_win32_stdout != old_fd_stdout:
                set_win32_std_handle(1, old_win32_stdout)
                refresh_click_windows_console_stream(1, old_win32_stdout)
            else:
                set_win32_std_handle_to_current_fd(1)
                refresh_click_windows_console_stream(1)
            try:
                os.close(old_stdout)
            except OSError:
                logger.debug("Could not close saved native stdout fd", exc_info=True)


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

    def __getattr__(self, name: str) -> Any:
        """Delegate legacy direct statistics access to :attr:`stats`."""
        return getattr(self.stats, name)


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


def _overlay_options(
    baseline: dict[str, str],
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a copy of ``baseline`` with optional override values applied."""
    merged = dict(baseline)
    if overrides:
        merged.update(overrides)
    return merged


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


class InferenceError(WinMLSessionError):
    """Inference failed."""


def _build_session_options(
    ep_device: WinMLEPDevice,
    ep_config: EPConfig | None = None,
    ep_monitor: WinMLEPMonitor | None = None,
    session_options_factory: Callable[[], ort.SessionOptions] | None = None,
    session_option_entries: dict[str, str] | None = None,
    provider_options: dict[str, str] | None = None,
) -> ort.SessionOptions:
    """Build a fully-bound ort.SessionOptions for one WinMLEPDevice pair.

    Free function (not a method): pure inputs -> pure outputs. The caller
    (typically :meth:`WinMLEPRegistry.auto_device`) has already resolved
    the (source, device) pair, so no registry / handle filtering happens
    here.
    """
    so = session_options_factory() if session_options_factory is not None else ort.SessionOptions()

    if session_option_entries is None and ep_monitor is not None:
        session_option_entries = dict(ep_monitor.get_session_options())

    if session_option_entries is not None:
        for key, value in session_option_entries.items():
            so.add_session_config_entry(key, value)

    if ep_monitor is not None:
        ep_monitor.configure_session_options(so)

    handle = ep_device.device._ort
    options = (
        dict(provider_options)
        if provider_options is not None
        else _build_provider_options(ep_device, ep_config, ep_monitor)
    )
    logger.info(
        "Building session options for ep=%s device=%s with provider_option_keys=%s",
        ep_device.ep.arg0,
        ep_device.device.device_type,
        sorted(options),
    )
    logger.debug("Session provider options: %s", options)
    so.add_provider_for_devices([handle], options)
    return so


class WinMLSession:
    """ONNX Runtime session bound to one resolved :class:`WinMLEPDevice`."""

    def __init__(
        self,
        onnx_path: str | Path,
        ep_device: WinMLEPDevice | str | None = None,
        *,
        device: str | None = None,
        ep: str | None = None,
        provider_options: dict[str, str] | None = None,
        ep_config: EPConfig | None = None,
        ep_monitor: WinMLEPMonitor | None = None,
        session_options: Callable[[], ort.SessionOptions] | None = None,
    ) -> None:
        """Initialize WinMLSession.

        Three invocation styles:

        1. **Fully-resolved (preferred for library callers):** pass
           ``ep_device=<WinMLEPDevice>`` constructed via
           :meth:`WinMLEPRegistry.auto_device` after :func:`resolve_device`.
        2. **Ergonomic (CLI + tests):** pass ``device="npu"|"gpu"|"cpu"|"auto"``
           and optionally ``ep="qnn"|...``; the session resolves an
           ``ep_device`` internally via the singleton registry.
        3. **Legacy:** omit the device for automatic selection, or supply the
           device policy as the second positional string.

        Args:
            onnx_path: Path to ONNX model.
            ep_device: Fully-resolved (source, device) pair, or a legacy positional
                device policy string.
            device: Device shortcut (npu/gpu/cpu/auto). Mutually resolved with ``ep``.
            ep: Optional EP short name — e.g. ``"qnn"`` — to pin.
            provider_options: EP-specific options dict, threaded into ep_config.
            ep_config: Optional EP configuration (provider_options, etc.).
            ep_monitor: Optional monitor. When passed, its session-config
                entries are threaded into the initial
                :func:`_build_session_options` call.
            session_options: Callable that returns configured ORT SessionOptions.
                A fresh object is requested for each ORT session construction.
        """
        # Legacy positional device strings share the ergonomic resolution path.
        # A resolved WinMLEPDevice remains the current positional API.
        if isinstance(ep_device, str):
            if device is not None:
                raise TypeError(
                    "WinMLSession received both a positional legacy device and device=."
                )
            device = ep_device
            ep_device = None

        # Ergonomic path: resolve ep_device from device/ep shortcuts.
        # Tests expect ``WinMLSession(onnx_path, device="cpu")`` to defer
        # InferenceSession creation to compile() (so ``ep_name`` returns
        # None before compile). Mark this path as lazy so the eager
        # runtime-workflow session-build at the bottom of __init__ is
        # skipped, preserving the compile-first contract for the CLI
        # ergonomic entry.
        _ergonomic_lazy = False
        if ep_device is None:
            if ep is not None and device is None:
                raise TypeError("WinMLSession requires device= when ep= is specified.")
            from .ep_device import EPDeviceTarget, resolve_device
            from .ep_registry import WinMLEPRegistry

            # NO silent CPU fallback here. If the requested (ep, device)
            # isn't available on this host, propagate the DeviceNotFound /
            # WinMLEPNotDiscovered / WinMLEPRegistrationFailed as-is —
            # silently rewriting a --device npu request to CPU would
            # produce wrong-device inference with no signal.
            target = resolve_device(
                EPDeviceTarget(ep=ep or "auto", device=(device or "auto").lower())
            )
            ep_device = WinMLEPRegistry.instance().auto_device(target)
            _ergonomic_lazy = True

        if provider_options is not None:
            # Fold provider_options into an EPConfig if the caller didn't
            # already supply one.
            if ep_config is None:
                from ..compiler.configs import EPConfig as _EPConfig

                ep_config = _EPConfig(
                    provider=None,
                    provider_options=dict(provider_options),
                )
            else:
                merged = dict(ep_config.provider_options or {})
                merged.update(provider_options)
                ep_config = replace(ep_config, provider_options=merged)

        self._onnx_path = Path(onnx_path)
        self._ep_device = ep_device
        self._ep_config = ep_config
        self._ep_monitor = ep_monitor
        self._session_options_factory = session_options

        initial_session_option_entries = (
            dict(ep_monitor.get_session_options()) if ep_monitor is not None else {}
        )

        # Snapshots preserved across perf()/reset()/compile() entry/exit (see perf()).
        self._provider_option_file_keys: frozenset[str] = frozenset(
            ep_config.provider_option_file_keys if ep_config is not None else ()
        )
        self._provider_options: dict[str, str] = self._canonicalize_option_files(
            _build_provider_options(ep_device, ep_config, ep_monitor),
            self._provider_option_file_keys,
        )
        self._active_session_option_entries: dict[str, str] = dict(initial_session_option_entries)
        self._markerless_epcontext_generations: dict[Path, Path | None] = {}
        # Convenience: the canonical EP name from the chosen handle.
        self._ep: str = ep_device.device.ep_name

        # Derived convenience attributes consumed by compile(), device property, etc.
        self._device: str = ep_device.device.device_type.lower()
        self._persist_jit: bool = ep_config.enable_ep_context if ep_config else False
        self._embed_context: bool = ep_config.embed_context if ep_config else False

        # _session is None until InferenceSession construction completes; __del__
        # reads this attribute, so it must exist before any call that could raise.
        self._session: ort.InferenceSession | None = None

        # ONNX model ORT actually loads (set during compile()). May differ from
        # _onnx_path when an EPContext model is compiled or a cached one reused.
        self._running_model_path: Path | None = None

        # State management
        self._state = SessionState.INITIALIZED
        self._last_error: Exception | None = None

        # Cached I/O metadata (lazy-loaded)
        self._io_config: dict | None = None

        # Performance tracking (enabled via perf() context manager)
        self._perf_stats: PerfStats | None = None

        # Compile workflows defer session creation to compile(); runtime workflows
        # create the session eagerly here.
        if not self._persist_jit and not _ergonomic_lazy:
            so = _build_session_options(
                self._ep_device,
                self._ep_config,
                None,
                self._session_options_factory,
                session_option_entries=self._active_session_option_entries,
                provider_options=self._provider_options,
            )
            with _suppress_native_output():
                self._session = ort.InferenceSession(self._onnx_path, sess_options=so)
            self._running_model_path = self._onnx_path
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
            logger.debug("Already compiled for %s", self._device)
            return

        target_device = self._device

        logger.info("Compiling for device: %s", target_device)

        if not self._persist_jit:
            try:
                with _suppress_native_output():
                    session = ort.InferenceSession(
                        str(self._onnx_path),
                        sess_options=_build_session_options(
                            self._ep_device,
                            self._ep_config,
                            None,
                            self._session_options_factory,
                            session_option_entries=self._active_session_option_entries,
                            provider_options=self._provider_options,
                        ),
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
            self._running_model_path = self._onnx_path
            self._state = SessionState.COMPILED
            return

        prepared_model = _PreparedEPContextModel(self._onnx_path)

        # Native QNN SDK compiler writes progress to stdout/stderr;
        # redirect to log file to keep the console clean.
        compile_log = self._onnx_path.parent / "compile.log"

        if is_compiled_onnx(self._onnx_path):
            # Input model is already an EPContext — use it directly.
            logger.info("Model already compiled (EPContext), skipping ModelCompiler")
        else:
            prepared_model = self._compile_epcontext_with_stable_source(compile_log)

        model_path = prepared_model.path

        try:
            # Create the runtime InferenceSession against the (possibly compiled) model.
            runtime_so = _build_session_options(
                self._ep_device,
                self._ep_config,
                None,
                self._session_options_factory,
                session_option_entries=self._active_session_option_entries,
                provider_options=self._provider_options,
            )
            with prepared_model, _suppress_native_output(compile_log):
                session = ort.InferenceSession(str(model_path), sess_options=runtime_so)

            actual_providers = session.get_providers()
            logger.info(
                "Session created for device %s, providers: %s",
                target_device,
                actual_providers,
            )

        except Exception as e:
            prepared_model.release()
            if prepared_model.markerless and model_path != self._onnx_path:
                self._discard_epcontext_generation(model_path)
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
        self._running_model_path = model_path
        self._state = SessionState.COMPILED
        if prepared_model.markerless and model_path != self._onnx_path:
            self._markerless_epcontext_generations[model_path] = prepared_model.lock_path

    def _compile_epcontext_with_stable_source(self, compile_log: Path) -> _PreparedEPContextModel:
        """Prepare an EPContext whose marker matches a stable source snapshot."""
        for _attempt in range(3):
            try:
                expected_identity = self._epcontext_cache_identity()
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Could not establish EPContext source identity; cache reuse disabled: %s",
                    exc,
                )
                cache_path = self._epcontext_cache_path(None)
                lock_path = cache_path.with_name(f"{cache_path.name}.lock")
                lease = _EPContextCacheLease.acquire(lock_path)
                assert lease is not None
                try:
                    prepared = self._prepare_epcontext_model(
                        cache_path,
                        compile_log,
                        None,
                    )
                    if prepared is not None and prepared.path != self._onnx_path:
                        prepared.lease = lease
                        lease = None
                    return prepared or _PreparedEPContextModel(self._onnx_path)
                finally:
                    if lease is not None:
                        lease.release()

            cache_path = self._epcontext_cache_path(expected_identity)
            lock_path = cache_path.with_name(f"{cache_path.name}.lock")
            lease = _EPContextCacheLease.acquire(lock_path)
            assert lease is not None
            try:
                try:
                    locked_identity = self._epcontext_cache_identity()
                except (OSError, ValueError):
                    continue
                if locked_identity != expected_identity:
                    continue
                prepared = self._prepare_epcontext_model(
                    cache_path,
                    compile_log,
                    locked_identity,
                )
                if prepared is None:
                    continue
                try:
                    final_identity = self._epcontext_cache_identity()
                except (OSError, ValueError):
                    if prepared.markerless and prepared.path != self._onnx_path:
                        self._discard_epcontext_generation(prepared.path)
                    continue
                if final_identity == locked_identity:
                    if prepared.path != self._onnx_path:
                        prepared.lease = lease
                        lease = None
                    return prepared
                if prepared.markerless and prepared.path != self._onnx_path:
                    self._discard_epcontext_generation(prepared.path)
            finally:
                if lease is not None:
                    lease.release()

        logger.warning("ONNX source changed during EPContext compilation; using original model")
        return _PreparedEPContextModel(self._onnx_path)

    def _prepare_epcontext_model(
        self,
        cache_path: Path,
        compile_log: Path,
        cache_identity: dict[str, object] | None,
    ) -> _PreparedEPContextModel | None:
        """Reuse or compile one EPContext while the caller holds its file lock."""
        if cache_identity is not None:
            cached_generation = self._epcontext_cached_generation(cache_path, cache_identity)
            if cached_generation is not None:
                logger.info("Using cached EPContext: %s", cached_generation)
                return _PreparedEPContextModel(cached_generation)

        generation_path = cache_path.with_name(
            f"{cache_path.stem}_{uuid.uuid4().hex[:16]}{cache_path.suffix}"
        )
        try:
            so = _build_session_options(
                self._ep_device,
                self._ep_config,
                None,
                self._session_options_factory,
                session_option_entries=self._active_session_option_entries,
                provider_options=self._provider_options,
            )
            model_compiler = ort.ModelCompiler(
                so,
                str(self._onnx_path),
                embed_compiled_data_into_model=self._embed_context,
            )
            with _suppress_native_output(compile_log):
                model_compiler.compile_to_file(str(generation_path))
        except Exception as exc:
            self._discard_epcontext_generation(generation_path)
            logger.warning("ModelCompiler failed, using original: %s", exc)
            return _PreparedEPContextModel(self._onnx_path)

        if not generation_path.exists():
            self._discard_epcontext_generation(generation_path)
            return _PreparedEPContextModel(self._onnx_path)
        markerless = cache_identity is None
        if cache_identity is not None:
            try:
                current_identity = self._epcontext_cache_identity()
            except (OSError, ValueError):
                self._discard_epcontext_generation(generation_path)
                return None
            if current_identity != cache_identity:
                self._discard_epcontext_generation(generation_path)
                return None
            try:
                self._write_epcontext_cache_marker(
                    cache_path,
                    generation_path,
                    cache_identity,
                )
            except (OSError, TypeError, ValueError) as exc:
                logger.warning(
                    "Compiled EPContext but could not write cache marker %s: %s",
                    self._epcontext_cache_marker_path(cache_path),
                    exc,
                )
                markerless = True
            else:
                self._prune_epcontext_cache(cache_path)
        logger.info("Compiled to EPContext: %s", generation_path)
        return _PreparedEPContextModel(generation_path, markerless=markerless)

    @classmethod
    def _discard_epcontext_generation(cls, generation_path: Path) -> None:
        """Best-effort removal of an unpublished generation and its sidecars."""
        paths: dict[Path, None] = {}
        try:
            for sidecar in cls._epcontext_external_sidecars(generation_path):
                paths.setdefault(sidecar, None)
        except (OSError, ValueError, DecodeError):
            logger.debug(
                "Could not parse EPContext sidecars for %s; using prefix cleanup fallback",
                generation_path,
                exc_info=True,
            )
        paths.setdefault(generation_path, None)
        try:
            for candidate in generation_path.parent.iterdir():
                if candidate.name.startswith(generation_path.stem) and candidate.is_file():
                    paths.setdefault(candidate, None)
        except OSError:
            logger.debug(
                "Could not enumerate EPContext generation sidecars for %s",
                generation_path,
            )
        for path in paths:
            cls._unlink_generation_file(path)

    def _prune_epcontext_cache(self, current_cache_path: Path) -> None:
        """Bound successful EPContext cache entries for this source model and device."""
        keep_count = max(1, _EPCONTEXT_CACHE_MAX_GENERATIONS)
        marker_suffix = ".meta.json"
        cache_name_suffix = "_ctx.onnx"
        marker_name_suffix = f"{cache_name_suffix}{marker_suffix}"
        marker_prefix = f"{self._onnx_path.stem}_{self._device}_"
        current_marker = self._epcontext_cache_marker_path(current_cache_path).resolve(strict=False)
        entries: list[tuple[bool, int, str, Path, Path, Path | None, bytes]] = []
        for marker_path in current_cache_path.parent.iterdir():
            try:
                if not marker_path.is_file():
                    continue
                if not marker_path.name.startswith(marker_prefix) or not marker_path.name.endswith(
                    marker_name_suffix
                ):
                    continue
                cache_name = marker_path.name.removesuffix(marker_suffix)
                if not cache_name.startswith(marker_prefix) or not cache_name.endswith(
                    cache_name_suffix
                ):
                    continue
                cache_path = marker_path.with_name(cache_name)
                marker_contents = marker_path.read_bytes()
                generation_path = self._epcontext_marker_generation(
                    marker_path,
                    marker_contents,
                )
                lock_path = cache_path.with_name(f"{cache_path.name}.lock")
                marker_stat = marker_path.stat()
            except OSError:
                continue
            is_current = marker_path.resolve(strict=False) == current_marker
            entries.append(
                (
                    is_current,
                    marker_stat.st_mtime_ns,
                    marker_path.name,
                    marker_path,
                    lock_path,
                    generation_path,
                    marker_contents,
                )
            )
        entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
        for (
            is_current,
            *_unused,
            marker_path,
            lock_path,
            generation_path,
            marker_contents,
        ) in entries[keep_count:]:
            if is_current:
                continue
            lease = _EPContextCacheLease.acquire(lock_path, blocking=False)
            if lease is None:
                continue
            try:
                try:
                    locked_contents = marker_path.read_bytes()
                except OSError:
                    continue
                locked_generation = self._epcontext_marker_generation(
                    marker_path,
                    locked_contents,
                )
                if locked_contents != marker_contents:
                    if generation_path is not None and generation_path != locked_generation:
                        self._discard_epcontext_generation(generation_path)
                    continue
                if locked_generation is not None:
                    self._discard_epcontext_generation(locked_generation)
                self._unlink_generation_file(marker_path)
            finally:
                lease.release()
            self._unlink_generation_file(lock_path)

    @staticmethod
    def _epcontext_marker_generation(marker_path: Path, contents: bytes) -> Path | None:
        """Return a marker's validated sibling generation path."""
        try:
            recorded = json.loads(contents)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(recorded, dict):
            return None
        generation_name = recorded.get("generation")
        if not isinstance(generation_name, str) or Path(generation_name).name != generation_name:
            return None
        return marker_path.parent / generation_name

    def _cleanup_markerless_epcontext_generations(self) -> None:
        """Best-effort cleanup for non-reusable EPContext generations owned by this session."""
        remaining: dict[Path, Path | None] = {}
        for generation_path, lock_path in self._markerless_epcontext_generations.items():
            self._discard_epcontext_generation(generation_path)
            if self._epcontext_generation_artifacts_exist(generation_path):
                remaining[generation_path] = lock_path
                continue
            if lock_path is not None:
                self._unlink_generation_file(lock_path)
        self._markerless_epcontext_generations = remaining

    @staticmethod
    def _epcontext_generation_artifacts_exist(generation_path: Path) -> bool:
        try:
            return any(
                candidate.name.startswith(generation_path.stem) and candidate.is_file()
                for candidate in generation_path.parent.iterdir()
            )
        except OSError:
            return generation_path.exists()

    @staticmethod
    def _unlink_generation_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove stale EPContext generation file %s", path)

    @staticmethod
    def _epcontext_cache_marker_path(ctx_path: Path) -> Path:
        """Return the sidecar that records a direct-session cache identity."""
        return ctx_path.with_name(f"{ctx_path.name}.meta.json")

    def _epcontext_cache_path(self, identity: dict[str, object] | None) -> Path:
        """Return an immutable identity path, or a unique non-cacheable path."""
        if identity is None:
            identity_token = uuid.uuid4().hex[:16]
        else:
            encoded_identity = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            identity_token = hashlib.sha256(encoded_identity.encode("utf-8")).hexdigest()[:16]
        return self._onnx_path.with_name(
            f"{self._onnx_path.stem}_{self._device}_{identity_token}_ctx.onnx"
        )

    def _epcontext_cache_identity(self) -> dict[str, object]:
        """Return all inputs that affect a direct-session EPContext artifact."""
        if self._session_options_factory is not None:
            raise ValueError(
                "custom SessionOptions factory cannot be represented in cache identity"
            )
        hardware = self._ep_device.device.ort_handle.device

        def _optional_text(value: object) -> str | None:
            return value if isinstance(value, str) and value else None

        dll_fingerprint = (
            None if self._ep_device.is_builtin else self._file_fingerprint(self._ep_device.dll_path)
        )

        return {
            "schema_version": 1,
            "source_model_hash": get_onnx_model_hash(self._onnx_path, strict=True),
            "ep": self._ep_device.device.ep_name,
            "ep_source": self._ep_device.source_tag,
            "ep_version": self._ep_device.version,
            "ep_dll": dll_fingerprint,
            "device": self._device,
            "hardware": {
                "vendor_id": hardware.vendor_id,
                "device_id": hardware.device_id,
                "name": _optional_text(self._ep_device.device.hardware_name),
                "driver_version": _optional_text(self._ep_device.device.driver_version),
                "compiler_version": _optional_text(self._ep_device.device.compiler_version),
            },
            "provider_options": dict(sorted(self._provider_options.items())),
            "provider_option_files": self._option_file_fingerprints(
                self._provider_options,
                self._provider_option_file_keys,
            ),
            "session_options": dict(sorted(self._active_session_option_entries.items())),
            "session_option_files": {},
            "embed_context": self._embed_context,
            "ort_version": ort.__version__,
        }

    def _option_file_fingerprints(
        self,
        options: dict[str, str],
        file_keys: frozenset[str],
    ) -> dict[str, dict[str, object]]:
        """Fingerprint provider options explicitly declared as file-backed."""
        fingerprints = {}
        for key in sorted(file_keys):
            value = options.get(key)
            if value is None:
                continue
            option_path = self._resolve_option_file(value)
            if option_path is None:
                raise ValueError(f"file-backed provider option {key!r} does not resolve to a file")
            fingerprints[key] = self._file_fingerprint(option_path)
        return fingerprints

    def _canonicalize_option_files(
        self,
        options: dict[str, str],
        file_keys: frozenset[str],
    ) -> dict[str, str]:
        """Resolve only provider options explicitly declared as file-backed."""
        canonicalized = dict(options)
        for key in file_keys:
            value = options.get(key)
            if value is None:
                raise ValueError(f"file-backed provider option {key!r} is not configured")
            option_path = self._resolve_option_file(value)
            if option_path is None:
                raise ValueError(f"file-backed provider option {key!r} does not resolve to a file")
            canonicalized[key] = str(option_path)
        return canonicalized

    def _resolve_option_file(self, value: object) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            raw_path = Path(value).expanduser()
        except (RuntimeError, ValueError):
            return None
        candidates: tuple[Path, ...] = (
            (raw_path,) if raw_path.is_absolute() else (Path.cwd() / raw_path,)
        )
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError, ValueError):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file():
                return resolved
        return None

    @staticmethod
    def _file_fingerprint(path: Path) -> dict[str, object]:
        """Return a strict metadata and content fingerprint for one file."""
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }

    @classmethod
    def _epcontext_cached_generation(
        cls,
        cache_path: Path,
        expected_identity: dict[str, object],
    ) -> Path | None:
        """Return the immutable generation selected by a valid identity marker."""
        marker_path = cls._epcontext_cache_marker_path(cache_path)
        try:
            recorded = json.loads(marker_path.read_text(encoding="utf-8"))
            if not isinstance(recorded, dict) or recorded.get("identity") != expected_identity:
                return None
            generation_name = recorded.get("generation")
            if (
                not isinstance(generation_name, str)
                or Path(generation_name).name != generation_name
            ):
                return None
            generation_path = cache_path.parent / generation_name
            current_artifacts = cls._epcontext_artifact_fingerprint(generation_path)
        except Exception:
            return None
        if recorded.get("artifacts") != current_artifacts:
            return None
        return generation_path

    @staticmethod
    def _epcontext_external_sidecars(ctx_path: Path) -> tuple[Path, ...]:
        """Return validated external binaries referenced by an EPContext graph."""
        from onnx import AttributeProto

        from ..onnx import load_onnx

        model = load_onnx(ctx_path, load_weights=False, validate=False)
        source_root = ctx_path.parent.resolve()
        sidecars: dict[Path, None] = {}
        for node in model.graph.node:
            if node.op_type != "EPContext":
                continue
            attrs = {attr.name: attr for attr in node.attribute}
            embed_mode = attrs.get("embed_mode")
            if embed_mode is None or (embed_mode.type == AttributeProto.INT and embed_mode.i != 0):
                continue
            if embed_mode.type != AttributeProto.INT:
                raise ValueError("EPContext embed_mode must be an integer")
            main_context = attrs.get("main_context")
            cache_attr = attrs.get("ep_cache_context")
            is_secondary = (
                main_context is not None
                and main_context.type == AttributeProto.INT
                and main_context.i == 0
            )
            if cache_attr is None and is_secondary:
                continue
            if cache_attr is None or cache_attr.type != AttributeProto.STRING:
                raise ValueError("External EPContext node must have a string ep_cache_context")
            try:
                cache_ref = cache_attr.s.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("EPContext cache reference must be valid UTF-8") from exc
            relative_ref = Path(cache_ref)
            if not cache_ref or relative_ref.is_absolute() or relative_ref.drive:
                raise ValueError(f"unsafe EPContext cache reference: {cache_ref!r}")
            try:
                sidecar = (source_root / relative_ref).resolve()
                sidecar.relative_to(source_root)
            except (OSError, ValueError) as exc:
                raise ValueError(f"unsafe EPContext cache reference: {cache_ref!r}") from exc
            if not sidecar.is_file() or sidecar.stat().st_size == 0:
                raise FileNotFoundError(f"EPContext sidecar is unavailable: {sidecar}")
            sidecars.setdefault(sidecar, None)
        return tuple(sidecars)

    @classmethod
    def _epcontext_artifact_fingerprint(cls, ctx_path: Path) -> list[dict[str, object]]:
        """Return metadata fingerprints for the graph and external binaries."""
        source_root = ctx_path.parent.resolve()
        paths = (ctx_path.resolve(), *cls._epcontext_external_sidecars(ctx_path))
        fingerprints = []
        for path in paths:
            fingerprint = cls._file_fingerprint(path)
            fingerprints.append(
                {
                    "path": path.relative_to(source_root).as_posix(),
                    "size": fingerprint["size"],
                    "mtime_ns": fingerprint["mtime_ns"],
                    "sha256": fingerprint["sha256"],
                }
            )
        return fingerprints

    @classmethod
    def _write_epcontext_cache_marker(
        cls,
        cache_path: Path,
        generation_path: Path,
        identity: dict[str, object],
    ) -> None:
        """Atomically publish the identity for a successfully compiled context."""
        marker_path = cls._epcontext_cache_marker_path(cache_path)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{marker_path.name}.", suffix=".tmp", dir=marker_path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as marker_file:
                json.dump(
                    {
                        "identity": identity,
                        "generation": generation_path.name,
                        "artifacts": cls._epcontext_artifact_fingerprint(generation_path),
                    },
                    marker_file,
                    indent=2,
                    sort_keys=True,
                )
            temporary_path.replace(marker_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

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

        # compile() populates self._session or raises; bind a non-None local so
        # the narrowing survives into the lambda / comprehension below (mypy drops
        # self-attribute narrowing inside nested scopes).
        session = self._session
        if session is None:
            raise InferenceError(
                message="Session not available after compile",
                context={"onnx_path": str(self._onnx_path), "device": self._device},
            )

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
            ort_inputs = self._prepare_inputs(inputs, session)

            # Run inference (with optional perf tracking)
            output_names = [o.name for o in session.get_outputs()]
            if self._perf_stats:
                outputs = self._perf_stats.record(lambda: session.run(output_names, ort_inputs))
            else:
                outputs = session.run(output_names, ort_inputs)

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
        self._reset_runtime_state(cleanup_markerless=True)

    def _reset_runtime_state(self, *, cleanup_markerless: bool) -> None:
        """Release the native session and optionally its private EPContext artifacts."""
        self._session = None
        if cleanup_markerless:
            self._cleanup_markerless_epcontext_generations()
        self._running_model_path = None
        self._state = SessionState.INITIALIZED
        self._last_error = None
        logger.info("Session reset")

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        try:
            self._session = None
            self._cleanup_markerless_epcontext_generations()
        except Exception:
            pass  # Suppress errors during interpreter shutdown

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
            if name not in name_to_type:
                continue

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

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state

    @property
    def device(self) -> str:
        """Target device for this session."""
        return self._device

    @property
    def ep_name(self) -> str | None:
        """Primary EP ORT actually bound, or None before compile.

        Returns ``session.get_providers()[0]`` — the EP that owns node
        partitioning. ``CPUExecutionProvider`` may still appear later
        in the list as ORT's automatic fallback for unsupported ops.
        """
        if self._session is None:
            return None
        providers = self._session.get_providers()
        return providers[0] if providers else None

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
    def perf(
        self,
        warmup: int = 0,
        monitor: WinMLEPMonitor | None = None,
    ) -> Iterator[PerfContext]:
        """Context manager for a scoped perf window.

        Yields a :class:`PerfContext` whose ``stats`` property accumulates
        timing from every :meth:`run` call made inside the ``with`` block.
        The optional *monitor* is entered/exited around the body.

        Session setup lifecycle
        -----------------------
        * If *monitor* contributes provider/session options that differ from
          the active session, the compiled session is torn down first
          (auto-reset with an INFO diagnostic) so the new options take effect.
        * After the ``with`` block exits, any rebuilt session is restored from
          the saved baseline provider/session-option snapshots.

        Teardown ordering (C-2 invariant)
        ----------------------------------
        * For monitors with ``requires_session_teardown=True`` (e.g. QNNMonitor
          which flushes CSV only on session destroy), :meth:`reset` fires
          *before* ``monitor.__exit__`` so the flushed data is available inside
          ``__exit__``.
        * After ``monitor.__exit__``, the baseline is rebuilt from its saved
          path and options without retaining a native session object.

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

        # Snapshot state for restore-on-exit.
        saved_sess_entries = dict(self._active_session_option_entries)
        saved_prov = dict(self._provider_options)
        saved_ep = self._ep
        had_baseline_session = self._session is not None
        saved_state = self._state
        saved_last_error = self._last_error
        saved_running_model_path = self._running_model_path
        active_model_path = (
            saved_running_model_path
            if had_baseline_session and saved_running_model_path is not None
            else self._onnx_path
        )

        # Inject source ONNX context before rebuilding; the active runtime
        # model path is injected after the monitored session is ready.
        effective_monitor.set_onnx_model_path(self._onnx_path)
        effective_monitor.set_onnx_op_types(self._build_op_type_map(self._onnx_path))

        desired_sess_entries = _overlay_options(
            saved_sess_entries,
            dict(monitor.get_session_options()) if monitor is not None else None,
        )
        new_prov = self._canonicalize_option_files(
            _overlay_options(
                saved_prov,
                dict(monitor.get_provider_options()) if monitor is not None else None,
            ),
            self._provider_option_file_keys,
        )

        # Rebuild InferenceSession only when monitor-contributed provider/session
        # options differ from the current session's options (i.e. a new session
        # is needed). Track whether we rebuilt so the teardown path knows
        # whether to restore.
        _session_rebuilt = (
            new_prov != self._provider_options
            or desired_sess_entries != self._active_session_option_entries
            or effective_monitor.configures_session_options
            or self._session is None
        )
        if had_baseline_session and _session_rebuilt:
            logger.info("auto-resetting compiled session to apply monitor session/provider options")
            self._reset_runtime_state(cleanup_markerless=False)

        stats = PerfStats(warmup=warmup)
        restore_baseline = _session_rebuilt or getattr(
            effective_monitor, "requires_session_teardown", False
        )

        def _restore_baseline() -> Exception | None:
            """Reconstruct the pre-perf session from snapshots, never retained objects."""
            self._session = None
            self._active_session_option_entries = saved_sess_entries
            self._provider_options = saved_prov
            self._ep = saved_ep
            self._perf_stats = None

            if not had_baseline_session:
                self._state = saved_state
                self._last_error = saved_last_error
                self._running_model_path = saved_running_model_path
                return None

            try:
                with _suppress_native_output():
                    self._session = ort.InferenceSession(
                        active_model_path,
                        sess_options=_build_session_options(
                            self._ep_device,
                            self._ep_config,
                            None,
                            self._session_options_factory,
                            session_option_entries=saved_sess_entries,
                            provider_options=saved_prov,
                        ),
                    )
            except Exception as error:
                self._session = None
                self._state = SessionState.ERROR
                self._last_error = error
                self._running_model_path = None
                logger.exception("Restoring baseline InferenceSession failed")
                return error

            self._state = saved_state
            self._last_error = saved_last_error
            self._running_model_path = saved_running_model_path
            return None

        try:
            if _session_rebuilt:
                so = _build_session_options(
                    self._ep_device,
                    self._ep_config,
                    effective_monitor,
                    self._session_options_factory,
                    session_option_entries=desired_sess_entries,
                    provider_options=new_prov,
                )
                with _suppress_native_output():
                    self._session = ort.InferenceSession(active_model_path, sess_options=so)
                self._provider_options = new_prov
                self._active_session_option_entries = desired_sess_entries
                self._running_model_path = active_model_path
            effective_monitor.set_running_model_path(active_model_path)
        except Exception:
            _restore_baseline()
            raise

        self._perf_stats = stats

        ctx = PerfContext(stats=stats, monitor=effective_monitor)

        # Enter the monitor manually so we can control teardown order (C-2
        # invariant: requires_session_teardown monitors need self.reset() to
        # fire BEFORE monitor.__exit__).
        try:
            effective_monitor.__enter__()
        except Exception:
            # __enter__ failed — rebuild the baseline and do NOT call __exit__.
            _restore_baseline()
            raise

        exc_info: tuple[type[BaseException] | None, BaseException | None, TracebackType | None] = (
            None,
            None,
            None,
        )
        try:
            yield ctx
        except BaseException:
            import sys

            exc_info = sys.exc_info()
        finally:
            # Give sampling monitors the completed perf-window counts before
            # any session teardown flushes and parses their artifacts.
            monitor_error: Exception | None = None
            try:
                effective_monitor.set_perf_window(
                    warmup=min(stats.warmup, stats.total_count),
                    measured_iterations=stats.count,
                )
            except Exception as error:
                logger.exception("Monitor set_perf_window failed")
                if exc_info[1] is None:
                    monitor_error = error

            # C-2: for monitors that require session teardown, release the
            # native session BEFORE monitor.__exit__ so flushed data is available.
            if getattr(effective_monitor, "requires_session_teardown", False):
                self._reset_runtime_state(cleanup_markerless=False)

            # Call monitor.__exit__ — propagate exc_info so monitor sees the
            # exception (exception transparency contract).
            try:
                effective_monitor.__exit__(*exc_info)
            except Exception as error:
                logger.exception("Monitor __exit__ failed")
                if exc_info[1] is None and monitor_error is None:
                    monitor_error = error

            restore_error: Exception | None = None

            if restore_baseline:
                restore_error = _restore_baseline()
            else:
                self._perf_stats = None

            # Re-raise any exception from the body.
            if exc_info[1] is not None:
                raise exc_info[1].with_traceback(exc_info[2])
            if monitor_error is not None:
                raise monitor_error
            if restore_error is not None:
                raise restore_error

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
    def _get_precision(model_proto: ModelProto) -> str | None:
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
    def _dominant_float_bits(graph: GraphProto) -> int | None:
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
        node: NodeProto,
        graph: GraphProto | None = None,
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
        inputs: list[ValueInfoProto] = []
        outputs: list[ValueInfoProto] = []

        if graph is not None:
            # Build lookup from parent graph
            all_value_info: dict[str, ValueInfoProto] = {vi.name: vi for vi in graph.value_info}
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
                self._session_options_factory,
            )
            sess_options.log_severity_level = 4  # Suppress ORT logs during probe
            with _suppress_native_output():
                ort.InferenceSession(
                    test_model.SerializeToString(),
                    sess_options=sess_options,
                )
            return True
        except Exception:
            return False
