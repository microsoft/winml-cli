# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Capture and replay native stderr written by ORT / QNN during initialization."""

from __future__ import annotations

import logging
import os
import re
import sys
from contextlib import contextmanager


logger = logging.getLogger(__name__)

# Matches ANSI SGR escape sequences (e.g. the colour codes ORT emits).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# ORT messages captured before any logging handler is configured are buffered
# here so callers can replay them once the logging infrastructure is ready.
_ort_startup_logs: list[str] = []


@contextmanager
def suppress_ep_registration_stderr(
    level: int = logging.DEBUG,
    *,
    use_startup_buffer: bool = True,
):
    """Capture native stderr and re-emit each line through Python logging.

    ORT and the QNN SDK write diagnostics (e.g. "Init provider bridge failed.",
    "DSP_INFO UNSUPPORTED_KEY:", "Starting stage:") directly to native stderr
    (fd 2 / Win32 STD_ERROR_HANDLE), bypassing Python's logging system.
    This context manager captures those writes via a pipe and re-emits them
    through ``logger`` at the requested level.

    On non-Windows platforms the bug does not manifest, so this is a no-op
    that yields immediately.

    Args:
        level: Logging level for re-emitted lines (default: ``logging.DEBUG``).
            Use ``logging.INFO`` for compilation output that should appear with
            ``-v`` but not by default.
        use_startup_buffer: When True (default), lines are also appended to
            the internal startup buffer for deferred replay via
            ``replay_ort_startup_logs()``.  Set to False for post-startup
            callers where logging is already configured.

    Restore order on Windows: ``os.dup2`` is called first (UCRT's dup2
    for fds 0-2 internally calls ``SetStdHandle``), then the original
    Win32 ``STD_ERROR_HANDLE`` is explicitly restored via ``SetStdHandle``
    so the true original HANDLE is preserved, not the duplicated one.
    """
    if sys.platform != "win32":
        yield
        return

    import ctypes
    import msvcrt
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetStdHandle.argtypes = [wintypes.DWORD]
    k32.GetStdHandle.restype = wintypes.HANDLE
    k32.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
    k32.SetStdHandle.restype = wintypes.BOOL

    std_error_handle = wintypes.DWORD(0xFFFFFFF4)

    read_fd, write_fd = os.pipe()
    old_fd = os.dup(2)
    # Capture the Win32 handle BEFORE dup2 — UCRT's dup2 for fds 0-2
    # internally calls SetStdHandle, so reading it afterwards would
    # return the pipe's handle instead of the original.
    old_w32 = k32.GetStdHandle(std_error_handle)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    k32.SetStdHandle(std_error_handle, msvcrt.get_osfhandle(2))
    try:
        yield
    finally:
        # 1. Restore CRT fd 2; this closes the pipe write end (the last
        #    reference), so the subsequent read reaches EOF without blocking.
        os.dup2(old_fd, 2)
        os.close(old_fd)
        # 2. Restore the original Win32 STD_ERROR_HANDLE.  This must happen
        #    AFTER dup2 because UCRT's dup2 for fds 0-2 internally calls
        #    SetStdHandle, overwriting whatever was set before.
        k32.SetStdHandle(std_error_handle, old_w32)
        # 3. Read all captured output and re-emit each line through logging.
        chunks: list[bytes] = []
        try:
            while chunk := os.read(read_fd, 4096):
                chunks.append(chunk)
        finally:
            os.close(read_fd)
        captured = b"".join(chunks).decode("utf-8", errors="replace")
        for line in captured.splitlines():
            line = _ANSI_RE.sub("", line).strip()
            if line:
                if use_startup_buffer:
                    _ort_startup_logs.append(line)
                logger.log(level, "[ORT] %s", line)


def replay_ort_startup_logs(target_logger: logging.Logger | None = None) -> None:
    """Replay captured ORT startup lines and clear the buffer.

    Call this after :func:`configure_logging` has set up handlers so the
    buffered messages are emitted at the correct log level.  The buffer is
    drained on first call; subsequent calls are no-ops.

    Args:
        target_logger: Logger to emit to.  Defaults to this module's logger.
    """
    if not _ort_startup_logs:
        return

    log = target_logger or logger
    while _ort_startup_logs:
        log.debug("[ORT] %s", _ort_startup_logs.pop(0))
