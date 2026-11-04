# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Capture llama.cpp's native log during GGUF load and the first warmup request.

The Windows ML Runtime installs its own llama.cpp logger once, while the GGUF
tokenizer opens; that logger drops everything below ERROR. llama.cpp reports
the effective context, batch, KV-cache and offload configuration only in its
INFO lines, so an evidence run swaps in a capturing callback through
``llama_log_set`` on the already loaded ``WinMLllama.dll``, keeps every
fragment the callback receives from model load through warmup 0, and then
restores the Runtime's callback so later requests run with the Runtime's own
(native) logging policy.

Each fragment is kept as llama.cpp emitted it, with its own QPC stamp:
continuation fragments (level 5, ``GGML_LOG_LEVEL_CONT``) are not joined to
the line they continue.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.wintypes as wintypes
import functools
import time
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from collections.abc import Callable


# void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data)
LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)

_LLAMA_LIBRARY = "WinMLllama.dll"
_NOT_LOADED = "WinMLllama.dll is not loaded in this process after the tokenizer opened."
_NOT_ATTEMPTED = "The run ended before the native log capture was installed."
_MAX_PATH_CHARS = 32768


class LlamaLogApi(Protocol):
    """``llama_log_get``/``llama_log_set``, with callbacks and user data as addresses."""

    def get(self) -> tuple[int | None, int | None]:
        """Return the installed ``(callback, user_data)``."""

    def set(self, callback: int | None, user_data: int | None) -> None:
        """Install ``callback`` with ``user_data``."""


@functools.cache
def _kernel32() -> Any:
    """A private kernel32 instance, so its prototypes never clash with other callers."""
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    return kernel32


def _module_file_name(handle: Any) -> str | None:
    """Full path of a loaded module, or ``None`` when it is no longer loaded."""
    buffer = ctypes.create_unicode_buffer(_MAX_PATH_CHARS)
    length = _kernel32().GetModuleFileNameW(handle, buffer, len(buffer))
    return buffer.value if length else None


def loaded_module_handle(name: str) -> tuple[int, str] | None:
    """Handle and full path of a module already loaded in this process, or ``None``."""
    handle = _kernel32().GetModuleHandleW(name)
    if not handle:
        return None
    path = _module_file_name(handle)
    return None if path is None else (int(handle), path)


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


def _loaded_llama_log_api() -> LlamaLogApi | None:
    """Bind the log API of ``WinMLllama.dll`` only if the Runtime already loaded it.

    Raises:
        AttributeError: The loaded library does not export the log API.
    """
    module = loaded_module_handle(_LLAMA_LIBRARY)
    if module is None:
        return None
    handle, path = module
    return _CtypesLlamaLogApi(path, handle)


class NativeLogCapture:
    """Swap in a capturing llama.cpp log callback and restore the Runtime's afterwards.

    Args:
        load_api: Returns the log API of the loaded library, or ``None`` when it
            is not loaded (injectable for tests).
        clock_ns: Integer-nanosecond clock stamping each captured fragment.
    """

    def __init__(
        self,
        *,
        load_api: Callable[[], LlamaLogApi | None] = _loaded_llama_log_api,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._load_api = load_api
        self._clock_ns = clock_ns
        self._api: LlamaLogApi | None = None
        self._saved: tuple[int | None, int | None] = (None, None)
        self._fragments: list[tuple[int, str, int]] = []
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

    def _on_log(self, level: int, text: bytes | None, _user_data: int | None) -> None:
        # Runs on native threads; it must never raise back into llama.cpp.
        try:
            with contextlib.suppress(Exception):
                if text is not None:
                    fragment = (int(level), text.decode("utf-8", "replace"), self._clock_ns())
                    self._fragments.append(fragment)
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
        try:
            api = self._load_api()
        except AttributeError as exc:
            self.reason = f"WinMLllama.dll does not export the llama log API: {exc}"
            return
        except Exception as exc:  # binding failures are evidence, not run failures
            self.reason = f"Could not bind the llama log API: {type(exc).__name__}: {exc}"
            return
        if api is None:
            self.reason = _NOT_LOADED
            return
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

    def to_dict(self) -> dict[str, Any]:
        """Return the evidence ``native_log`` object.

        ``intact`` is whether the capturing callback was still installed when
        :meth:`restore` ran: ``None`` before that, and also when the
        ``llama_log_get`` read inside :meth:`restore` raised (restore suppresses
        that error). ``restored`` is whether the Runtime callback is back.
        ``fragments`` lists every callback call in order, each with its
        ``level``, its ``text`` exactly as emitted (line breaks included) and
        its own ``qpc_ns`` stamp.
        """
        return {
            "installed": self.installed,
            "reason": self.reason,
            "intact": self._intact,
            "restored": self._restored,
            "fragments": [
                {"level": level, "text": text, "qpc_ns": qpc_ns}
                for level, text, qpc_ns in self._fragments
            ],
        }
