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
from contextlib import contextmanager


logger = logging.getLogger(__name__)

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
def suppress_native_stderr():
    """Redirect native stderr to devnull.  No-op on non-Windows."""
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
def capture_native_stderr(level: int = logging.INFO):
    """Capture native stderr via pipe and re-emit through Python logging.

    No-op on non-Windows.
    """
    if sys.platform != "win32":
        yield
        return

    read_fd, write_fd = os.pipe()
    old_fd = os.dup(2)
    old_w32 = _k32.GetStdHandle(_STD_ERROR_HANDLE)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    _k32.SetStdHandle(_STD_ERROR_HANDLE, msvcrt.get_osfhandle(2))
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        _k32.SetStdHandle(_STD_ERROR_HANDLE, old_w32)
        # Drain pipe and re-emit each line.
        _ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        chunks: list[bytes] = []
        try:
            while chunk := os.read(read_fd, 4096):
                chunks.append(chunk)
        finally:
            os.close(read_fd)
        for raw in b"".join(chunks).decode("utf-8", errors="replace").splitlines():
            line = _ansi_re.sub("", raw).strip()
            if line:
                logger.log(level, "[ORT] %s", line)
