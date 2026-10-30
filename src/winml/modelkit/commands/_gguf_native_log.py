# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Capture llama.cpp native log lines during GGUF load and the first warmup request.

The Windows ML Runtime installs its own llama.cpp logger once, while the GGUF
tokenizer opens; that logger drops everything below ERROR. llama.cpp reports
the effective context, batch, KV-cache and offload configuration only in its
INFO lines, so an evidence run swaps in a capturing callback through
``llama_log_set`` on the already loaded ``WinMLllama.dll``, keeps every line
from model load through warmup 0, and then restores the Runtime's callback so
later requests run with the Runtime's own (native) logging policy.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import re
import time
from typing import TYPE_CHECKING, Any, Protocol

from ._gguf_identity import loaded_module_handle


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


# ggml_log_level value of a continuation fragment of the previous line.
GGML_LOG_LEVEL_CONT = 5
# void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data)
LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
NATIVE_VALUE_KEYS: tuple[str, ...] = (
    "n_ctx",
    "n_batch",
    "n_ubatch",
    "flash_attn",
    "type_k",
    "type_v",
    "n_layer_offloaded",
    "n_layer_total",
)

_LLAMA_LIBRARY = "WinMLllama.dll"
_NOT_LOADED = "WinMLllama.dll is not loaded in this process after the tokenizer opened."
_NOT_ATTEMPTED = "The run ended before the native log capture was installed."
_OUTSIDE_CAPTURE = (
    "Backend registration and device lines are written to process stderr before the "
    "Runtime installs its logger."
)
_CONTEXT_VALUE = re.compile(r"^llama_context: (n_ctx|n_batch|n_ubatch|flash_attn)\s*=\s*(\S+)$")
_OFFLOADED = re.compile(r"^load_tensors: offloaded (\d+)/(\d+) layers to GPU$")
_KV_TYPES = re.compile(r"^llama_kv_cache: .*\bK \((\w+)\):.*\bV \((\w+)\):")
_INTEGER_CONTEXT_KEYS = frozenset({"n_ctx", "n_batch", "n_ubatch"})


class LlamaLogApi(Protocol):
    """``llama_log_get``/``llama_log_set``, with callbacks and user data as addresses."""

    def get(self) -> tuple[int | None, int | None]:
        """Return the installed ``(callback, user_data)``."""

    def set(self, callback: int | None, user_data: int | None) -> None:
        """Install ``callback`` with ``user_data``."""


class _CtypesLlamaLogApi:
    """The log API exported by an already loaded ``WinMLllama.dll``."""

    def __init__(self, path: str, handle: int) -> None:
        library = ctypes.CDLL(path, handle=handle)  # wraps the loaded module; no LoadLibrary
        self._get = library.llama_log_get
        self._get.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        self._get.restype = None
        self._set = library.llama_log_set
        self._set.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._set.restype = None

    def get(self) -> tuple[int | None, int | None]:
        callback, user_data = ctypes.c_void_p(), ctypes.c_void_p()
        self._get(ctypes.pointer(callback), ctypes.pointer(user_data))
        return callback.value, user_data.value

    def set(self, callback: int | None, user_data: int | None) -> None:
        self._set(callback, user_data)


def _loaded_llama_log_api() -> tuple[str, LlamaLogApi] | None:
    """Bind the log API of ``WinMLllama.dll`` only if the Runtime already loaded it.

    Raises:
        AttributeError: The loaded library does not export the log API.
    """
    module = loaded_module_handle(_LLAMA_LIBRARY)
    if module is None:
        return None
    handle, path = module
    return path, _CtypesLlamaLogApi(path, handle)


def join_continuations(raw: Iterable[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
    """Join ``GGML_LOG_LEVEL_CONT`` fragments onto the line they continue.

    A fragment with no preceding line starts its own line. Trailing line breaks
    are removed after joining.
    """
    joined: list[list[Any]] = []
    for level, text, qpc_ns in raw:
        if level == GGML_LOG_LEVEL_CONT and joined:
            joined[-1][1] += text
        else:
            joined.append([level, text, qpc_ns])
    return [(level, text.rstrip("\r\n"), qpc_ns) for level, text, qpc_ns in joined]


def parse_native_values(lines: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Read the effective llama.cpp configuration from captured log lines.

    For reference only: the benchmark re-parses the raw lines with its shared
    parser. Each key keeps its first match, as ``{"value", "line"}`` where
    ``line`` indexes ``lines``.
    """
    values: dict[str, dict[str, Any]] = {}

    def keep(key: str, value: Any, index: int) -> None:
        values.setdefault(key, {"value": value, "line": index})

    for index, line in enumerate(lines):
        text = str(line.get("text", ""))
        if match := _CONTEXT_VALUE.match(text):
            key, raw = match.group(1), match.group(2)
            keep(key, int(raw) if key in _INTEGER_CONTEXT_KEYS else raw, index)
        elif match := _OFFLOADED.match(text):
            keep("n_layer_offloaded", int(match.group(1)), index)
            keep("n_layer_total", int(match.group(2)), index)
        elif match := _KV_TYPES.match(text):
            keep("type_k", match.group(1), index)
            keep("type_v", match.group(2), index)
    return values


class NativeLogCapture:
    """Swap in a capturing llama.cpp log callback and restore the Runtime's afterwards.

    Args:
        load_api: Returns ``(library path, log API)`` for the loaded library, or
            ``None`` when it is not loaded (injectable for tests).
        clock_ns: Integer-nanosecond clock stamping each captured line.
    """

    def __init__(
        self,
        *,
        load_api: Callable[[], tuple[str, LlamaLogApi] | None] = _loaded_llama_log_api,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._load_api = load_api
        self._clock_ns = clock_ns
        self._api: LlamaLogApi | None = None
        self._saved: tuple[int | None, int | None] = (None, None)
        self._raw: list[tuple[int, str, int]] = []
        # The callback object must stay referenced while native code can call it.
        self._callback = LOG_CALLBACK(self._on_log)
        self._address = ctypes.cast(self._callback, ctypes.c_void_p).value
        self._restore_attempted = False
        self._restored = False
        self._intact: bool | None = None
        self._interrupted = False
        self.installed = False
        # Why the capture is not installed; None once it is.
        self.reason: str | None = _NOT_ATTEMPTED
        self.library: str | None = None

    def _on_log(self, level: int, text: bytes | None, _user_data: int | None) -> None:
        # Runs on native threads; it must never raise back into llama.cpp.
        try:
            with contextlib.suppress(Exception):
                if text is not None:
                    line = (int(level), text.decode("utf-8", "replace"), self._clock_ns())
                    self._raw.append(line)
        except KeyboardInterrupt:
            # Ctrl+C that lands while llama.cpp logs on the main thread cannot
            # propagate through native frames; keep it for raise_if_interrupted().
            self._interrupted = True

    def raise_if_interrupted(self) -> None:
        """Raise the Ctrl+C a log callback absorbed while native code was running.

        Call this after each native call made while the capture is installed.
        Each absorbed interrupt is raised once.

        Raises:
            KeyboardInterrupt: Ctrl+C arrived while the callback was running.
        """
        if self._interrupted:
            self._interrupted = False
            raise KeyboardInterrupt

    def install(self) -> None:
        """Install the capturing callback; failures are recorded in ``reason``, never raised."""
        if self.installed:
            return
        # ``reason`` keeps explaining the missing capture until installation succeeds,
        # so an install interrupted by Ctrl+C still leaves a reason in the evidence.
        try:
            loaded = self._load_api()
        except AttributeError as exc:
            self.reason = f"WinMLllama.dll does not export the llama log API: {exc}"
            return
        except Exception as exc:  # binding failures are evidence, not run failures
            self.reason = f"Could not bind the llama log API: {type(exc).__name__}: {exc}"
            return
        if loaded is None:
            self.reason = _NOT_LOADED
            return
        self.library, api = loaded
        try:
            saved = api.get()
        except Exception as exc:  # a failed read must not fail the run
            self.reason = f"Could not install the capturing callback: {type(exc).__name__}: {exc}"
            return
        try:
            api.set(self._address, None)
            current, _ = api.get()
        except Exception as exc:  # a failed swap must not fail the run
            self.reason = f"Could not install the capturing callback: {type(exc).__name__}: {exc}"
        else:
            if current == self._address:
                self._api, self._saved, self.installed = api, saved, True
                self.reason = None
            else:
                self.reason = "llama_log_set did not install the capturing callback."
        finally:
            # Native code must never keep calling a capture that is not tracked as
            # installed: put the Runtime callback back on every other exit, including
            # Ctrl+C cutting the swap short.
            if not self.installed:
                with contextlib.suppress(Exception):
                    api.set(*saved)

    def restore(self) -> None:
        """Reinstate the Runtime callback, unless something else replaced ours first.

        Safe to call more than once; only the first call acts.
        """
        if not self.installed or self._api is None or self._restore_attempted:
            return
        self._restore_attempted = True
        with contextlib.suppress(Exception):
            current, _ = self._api.get()
            self._intact = current == self._address
            if self._intact:
                self._api.set(*self._saved)
            after, _ = self._api.get()
            self._restored = after == self._saved[0]

    def lines(self, warmup0_submit_ns: int | None) -> list[dict[str, Any]]:
        """Captured lines; those stamped before the warmup-0 submission are ``load``."""
        return [
            {
                "level": level,
                "text": text,
                "qpc_ns": qpc_ns,
                "phase": "load"
                if warmup0_submit_ns is None or qpc_ns < warmup0_submit_ns
                else "warmup-0",
            }
            for level, text, qpc_ns in join_continuations(self._raw)
        ]

    def to_dict(self, warmup0_submit_ns: int | None) -> dict[str, Any]:
        """Return the evidence ``native_log`` object."""
        lines = self.lines(warmup0_submit_ns)
        return {
            "capture": {
                "api": "llama_log_set",
                "library": self.library,
                "installed": self.installed,
                "reason": None if self.installed else self.reason,
            },
            "window": "load_through_warmup_0",
            "lines": lines,
            "lines_sha256": hashlib.sha256(
                json.dumps(lines, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest(),
            "after_capture": {
                "policy": "runtime_callback_restored"
                if self.installed
                else "runtime_default_unobserved",
                "restored": self._restored,
                "verified_by_llama_log_get": self.installed and self._restore_attempted,
                "intact_until_restore": self._intact,
            },
            "outside_capture": _OUTSIDE_CAPTURE,
        }
