# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Redirect native stderr written by ORT / QNN on Windows.

ORT's native code writes diagnostics (e.g. "Init provider bridge failed.")
directly to fd 2 / Win32 STD_ERROR_HANDLE, bypassing Python logging.
Two context managers are provided:

* ``suppress_native_stderr``  - discard to devnull  (startup noise)
* ``capture_native_stderr``   - capture via pipe and re-log  (compilation output)

Both are no-ops on non-Windows.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .._env import env_flag_enabled


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)
_NATIVE_WARNING_LINE_RE = re.compile(rb"\[[Ww]:")

# ---------------------------------------------------------------------------
# Win32 kernel32 (configured once)
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    import ctypes.wintypes
    import msvcrt

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.GetStdHandle.argtypes = [ctypes.wintypes.DWORD]
    _k32.GetStdHandle.restype = ctypes.wintypes.HANDLE
    _k32.SetStdHandle.argtypes = [ctypes.wintypes.DWORD, ctypes.wintypes.HANDLE]
    _k32.SetStdHandle.restype = ctypes.wintypes.BOOL
    _STD_ERROR_HANDLE = ctypes.wintypes.DWORD(0xFFFFFFF4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def suppress_native_stderr(*, enabled: bool = True) -> Iterator[None]:
    """Redirect native stderr to devnull.  No-op on non-Windows."""
    if not enabled:
        yield
        return
    if sys.platform != "win32":
        yield
        return

    old_fd = os.dup(2)
    old_w32 = _k32.GetStdHandle(_STD_ERROR_HANDLE)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    _k32.SetStdHandle(_STD_ERROR_HANDLE, msvcrt.get_osfhandle(2))
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        _k32.SetStdHandle(_STD_ERROR_HANDLE, old_w32)


@contextmanager
def capture_native_stderr(level: int = logging.INFO) -> Iterator[None]:
    """Capture native stderr via pipe and re-emit through Python logging.

    No-op on non-Windows.
    """
    if sys.platform != "win32":
        yield
        return

    read_fd, write_fd = os.pipe()
    # Drain the pipe on a background thread *while* the wrapped block runs.
    # A chatty native EP (e.g. a VitisAI model compilation) can emit far more
    # than the OS pipe buffer holds (~64 KB on Windows) before we regain
    # control. If the pipe were only drained after the yield, the native
    # write() would block on a full buffer and stall the process indefinitely.
    # Reading concurrently keeps the buffer from ever filling up.
    chunks: list[bytes] = []

    def _drain() -> None:
        try:
            while chunk := os.read(read_fd, 4096):
                chunks.append(chunk)
        except OSError:
            pass  # read end closed or broken; stop draining
        finally:
            os.close(read_fd)

    reader = threading.Thread(target=_drain, name="capture-native-stderr", daemon=True)

    old_fd = os.dup(2)
    old_w32 = _k32.GetStdHandle(_STD_ERROR_HANDLE)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    _k32.SetStdHandle(_STD_ERROR_HANDLE, msvcrt.get_osfhandle(2))
    reader.start()
    try:
        yield
    finally:
        # Restoring fd 2 drops the last reference to the pipe's write end, which
        # signals EOF to the reader thread so it can finish and close the read end.
        os.dup2(old_fd, 2)
        os.close(old_fd)
        _k32.SetStdHandle(_STD_ERROR_HANDLE, old_w32)
        reader.join()
        # Re-emit each captured line through Python logging.
        _ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        for raw in b"".join(chunks).decode("utf-8", errors="replace").splitlines():
            line = _ansi_re.sub("", raw).strip()
            if line:
                logger.log(level, "[ORT] %s", line)


@contextmanager
def suppress_native_warnings(*, enabled: bool = True) -> Iterator[None]:
    """Hide native warning lines while preserving native errors and diagnostics.

    Native ORT/QNN diagnostics use severity tokens such as ``[W:...]`` and
    ``[E:...]``. Normal CLI output hides warning-level native chatter; ``-v`` /
    ``-vv`` or ``WINMLCLI_SHOW_ALL_WARNINGS=1`` leaves stderr untouched.
    """
    if not enabled or _show_native_warnings_requested():
        yield
        return

    old_fd = os.dup(2)
    old_w32 = _get_win32_stderr_handle()
    try:
        with tempfile.TemporaryFile() as captured:
            os.dup2(captured.fileno(), 2)
            _set_win32_stderr_to_current_fd()
            try:
                yield
            finally:
                os.dup2(old_fd, 2)
                _restore_win32_stderr_handle(old_w32)
                captured.seek(0)
                _write_all(old_fd, _filter_native_warning_stderr(captured.read()))
    finally:
        os.close(old_fd)


def _show_native_warnings_requested() -> bool:
    return env_flag_enabled("WINMLCLI_SHOW_ALL_WARNINGS") or logging.getLogger().isEnabledFor(
        logging.INFO
    )


def _filter_native_warning_stderr(data: bytes) -> bytes:
    return b"".join(
        line for line in data.splitlines(keepends=True) if not _is_native_warning_line(line)
    )


def _is_native_warning_line(line: bytes) -> bool:
    return bool(_NATIVE_WARNING_LINE_RE.search(line))


def _get_win32_stderr_handle() -> object | None:
    if sys.platform != "win32":
        return None
    return _k32.GetStdHandle(_STD_ERROR_HANDLE)


def _set_win32_stderr_to_current_fd() -> None:
    if sys.platform == "win32":
        _k32.SetStdHandle(_STD_ERROR_HANDLE, msvcrt.get_osfhandle(2))


def _restore_win32_stderr_handle(handle: object | None) -> None:
    if sys.platform == "win32" and handle is not None:
        _k32.SetStdHandle(_STD_ERROR_HANDLE, handle)


def _write_all(fd: int, data: bytes) -> None:
    while data:
        written = os.write(fd, data)
        data = data[written:]
