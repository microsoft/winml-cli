# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Redirect/filter native stderr written by ORT / QNN.

ORT's native code writes diagnostics (e.g. "Init provider bridge failed.")
directly to fd 2 / Win32 STD_ERROR_HANDLE, bypassing Python logging.
Three context managers are provided:

* ``suppress_native_stderr``  - discard to devnull  (startup noise)
* ``capture_native_stderr``   - capture via pipe and re-log  (compilation output)
* ``suppress_native_warnings`` - hide warning lines, replay everything else

The first two are no-ops on non-Windows. ``suppress_native_warnings`` works via
fd 2 on all platforms and also keeps the Win32 stderr handle in sync on Windows.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .._env import env_flag_enabled


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)
_NATIVE_SEVERITY_TOKEN_RE = re.compile(rb"\[([A-Za-z]):")
_NATIVE_READER_JOIN_TIMEOUT_SECONDS = 1.0
_NATIVE_FD_REDIRECT_LOCK = threading.RLock()

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
    _STD_OUTPUT_HANDLE = ctypes.wintypes.DWORD(0xFFFFFFF5)
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

    with native_fd_redirect_lock():
        old_fd: int | None = None
        try:
            old_fd = os.dup(2)
            with Path(os.devnull).open("wb") as devnull:
                os.dup2(devnull.fileno(), 2)
        except OSError:
            _close_fd(old_fd)
            logger.debug(
                "Native stderr suppression setup failed; leaving stderr unchanged",
                exc_info=True,
            )
            yield
            return

        _set_win32_std_handle_to_current_fd(2)
        try:
            yield
        finally:
            _restore_redirected_fd(2, old_fd)
            _set_win32_std_handle_to_current_fd(2)
            _refresh_click_windows_console_stream(2)
            _close_fd(old_fd)


@contextmanager
def capture_native_stderr(level: int = logging.INFO) -> Iterator[None]:
    """Capture native stderr via pipe and re-emit through Python logging.

    No-op on non-Windows.
    """
    if sys.platform != "win32":
        yield
        return

    read_fd: int | None = None
    write_fd: int | None = None
    old_fd: int | None = None
    try:
        read_fd, write_fd = os.pipe()
    except OSError:
        logger.debug(
            "Native stderr capture setup failed; leaving stderr unchanged",
            exc_info=True,
        )
        yield
        return

    # Drain the pipe on a background thread *while* the wrapped block runs.
    # A chatty native EP (e.g. a VitisAI model compilation) can emit far more
    # than the OS pipe buffer holds (~64 KB on Windows) before we regain
    # control. If the pipe were only drained after the yield, the native
    # write() would block on a full buffer and stall the process indefinitely.
    # Reading concurrently keeps the buffer from ever filling up.
    chunks: list[bytes] = []

    def _drain() -> None:
        assert read_fd is not None
        try:
            while chunk := os.read(read_fd, 4096):
                chunks.append(chunk)
        except OSError:
            pass  # read end closed or broken; stop draining
        finally:
            os.close(read_fd)

    reader = threading.Thread(target=_drain, name="capture-native-stderr", daemon=True)

    with native_fd_redirect_lock():
        try:
            old_fd = os.dup(2)
            os.dup2(write_fd, 2)
            _close_fd(write_fd)
            write_fd = None
            _set_win32_std_handle_to_current_fd(2)
            reader.start()
        except (OSError, RuntimeError):
            _restore_redirected_fd(2, old_fd)
            _close_fd(old_fd)
            _close_fd(read_fd)
            _close_fd(write_fd)
            _set_win32_std_handle_to_current_fd(2)
            _refresh_click_windows_console_stream(2)
            logger.debug(
                "Native stderr capture redirect failed; leaving stderr unchanged",
                exc_info=True,
            )
            yield
            return

        assert old_fd is not None
        try:
            yield
        finally:
            # Restoring fd 2 drops the last reference to the pipe's write end, which
            # signals EOF to the reader thread so it can finish and close the read end.
            _restore_redirected_fd(2, old_fd)
            _set_win32_std_handle_to_current_fd(2)
            _refresh_click_windows_console_stream(2)
            reader.join(timeout=_NATIVE_READER_JOIN_TIMEOUT_SECONDS)
            if reader.is_alive():
                logger.debug("Native stderr capture reader did not finish after stderr restore")
            _close_fd(old_fd)
            # Re-emit each captured line through Python logging.
            _ansi_re = re.compile(r"\x1b\[[0-9;]*m")
            for raw in b"".join(chunks).decode("utf-8", errors="replace").splitlines():
                line = _ansi_re.sub("", raw).strip()
                if line:
                    logger.log(level, "[ORT] %s", line)


@contextmanager
def suppress_native_warnings(
    *,
    enabled: bool = True,
    preserve_unclassified: bool = True,
) -> Iterator[None]:
    """Hide native warning lines while preserving native errors and diagnostics.

    Native ORT/QNN diagnostics use severity tokens such as ``[W:...]`` and
    ``[E:...]``. Normal CLI output hides warning-level native chatter; ``-v`` /
    ``-vv`` or ``WINMLCLI_SHOW_ALL_WARNINGS=1`` leaves stderr untouched.
    """
    if not enabled or _show_native_warnings_requested():
        yield
        return

    read_fd: int | None = None
    write_fd: int | None = None
    old_fd: int | None = None
    try:
        read_fd, write_fd = os.pipe()
    except OSError:
        _close_fd(read_fd)
        _close_fd(write_fd)
        logger.debug(
            "Native warning suppression setup failed; leaving stderr unchanged",
            exc_info=True,
        )
        yield
        return

    redirect_failed = False
    with native_fd_redirect_lock():
        try:
            old_fd = os.dup(2)
        except OSError:
            _close_fd(read_fd)
            _close_fd(write_fd)
            logger.debug(
                "Native warning suppression setup failed; leaving stderr unchanged",
                exc_info=True,
            )
            redirect_failed = True
        if not redirect_failed:
            assert old_fd is not None
            reader = threading.Thread(
                target=_drain_filtered_native_stderr,
                args=(read_fd, old_fd, preserve_unclassified),
                name="suppress-native-warnings",
                daemon=True,
            )
            try:
                os.dup2(write_fd, 2)
            except OSError:
                _close_fd(read_fd)
                _close_fd(write_fd)
                _close_fd(old_fd)
                logger.debug(
                    "Native warning suppression redirect failed; leaving stderr unchanged",
                    exc_info=True,
                )
                redirect_failed = True
            if not redirect_failed:
                _close_fd(write_fd)
                _set_win32_std_handle_to_current_fd(2)
                reader.start()
                try:
                    yield
                finally:
                    _restore_redirected_native_stderr_fd(old_fd)
                    _set_win32_std_handle_to_current_fd(2)
                    _refresh_click_windows_console_stream(2)
                    reader.join(timeout=_NATIVE_READER_JOIN_TIMEOUT_SECONDS)
                    if reader.is_alive():
                        logger.debug(
                            "Native warning suppression reader did not finish after stderr restore"
                        )
                    _close_fd(old_fd)
    if redirect_failed:
        yield


@contextmanager
def native_fd_redirect_lock() -> Iterator[None]:
    """Serialize process-wide fd/Win32 stdout/stderr redirection helpers."""
    with _NATIVE_FD_REDIRECT_LOCK:
        yield


def _show_native_warnings_requested() -> bool:
    return env_flag_enabled("WINMLCLI_SHOW_ALL_WARNINGS") or logging.getLogger().isEnabledFor(
        logging.INFO
    )


def _restore_redirected_native_stderr_fd(old_fd: int) -> None:
    _restore_redirected_fd(2, old_fd)


def _restore_redirected_fd(fd: int, old_fd: int | None) -> None:
    if old_fd is None:
        return
    try:
        os.dup2(old_fd, fd)
        return
    except OSError:
        logger.debug("Could not restore redirected native fd %d", fd, exc_info=True)

    try:
        os.close(fd)
    except OSError:
        logger.debug("Could not close redirected native fd %d", fd, exc_info=True)

    try:
        os.dup2(old_fd, fd)
    except OSError:
        logger.debug(
            "Could not restore native fd %d after closing redirected fd",
            fd,
            exc_info=True,
        )


def _drain_filtered_native_stderr(
    read_fd: int,
    target_fd: int,
    preserve_unclassified: bool,
) -> None:
    pending = b""
    try:
        while chunk := os.read(read_fd, 4096):
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                _write_non_warning_line(target_fd, line + b"\n", preserve_unclassified)
        if pending:
            _write_non_warning_line(target_fd, pending, preserve_unclassified)
    except OSError:
        # fd redirection is best-effort cleanup; restore path handles usability.
        pass
    finally:
        os.close(read_fd)


def _write_non_warning_line(fd: int, line: bytes, preserve_unclassified: bool) -> None:
    if _should_preserve_native_line(line, preserve_unclassified):
        _write_all(fd, line)


def _should_preserve_native_line(line: bytes, preserve_unclassified: bool) -> bool:
    match = _NATIVE_SEVERITY_TOKEN_RE.search(line)
    if match is None:
        return preserve_unclassified
    return match.group(1).lower() != b"w"


def _is_native_warning_line(line: bytes) -> bool:
    match = _NATIVE_SEVERITY_TOKEN_RE.search(line)
    if match is None:
        return False
    return match.group(1).lower() == b"w"


def _get_win32_stderr_handle() -> object | None:
    if sys.platform != "win32":
        return None
    return cast("object", _k32.GetStdHandle(_STD_ERROR_HANDLE))


def _set_win32_stderr_to_current_fd() -> None:
    _set_win32_std_handle_to_current_fd(2)


def _set_win32_std_handle_to_current_fd(fd: int) -> None:
    if sys.platform != "win32":
        return
    if fd == 1:
        std_handle = _STD_OUTPUT_HANDLE
    elif fd == 2:
        std_handle = _STD_ERROR_HANDLE
    else:
        return
    try:
        _k32.SetStdHandle(std_handle, msvcrt.get_osfhandle(fd))
    except OSError:
        logger.debug("Could not sync Win32 std handle for fd %d", fd, exc_info=True)


def _restore_win32_stderr_handle(handle: object | None) -> None:
    if sys.platform == "win32":
        _k32.SetStdHandle(_STD_ERROR_HANDLE, handle)


def _refresh_click_windows_console_stream(fd: int) -> None:
    """Refresh Click's cached Windows console writer after fd redirection.

    Click caches ``STDOUT_HANDLE`` / ``STDERR_HANDLE`` at import time and may
    cache ConsoleStream instances keyed by ``sys.stdout`` / ``sys.stderr``.
    ``os.dup2`` closes the original OS handle, so those cached writers must be
    repointed at the restored fd handle to avoid ``Windows error: 6`` later.
    """
    if sys.platform != "win32":
        return
    try:
        import click._compat as click_compat
        import click._winconsole as click_winconsole
    except ImportError:
        return

    try:
        handle = msvcrt.get_osfhandle(fd)
    except OSError:
        return

    if fd == 1:
        click_winconsole.STDOUT_HANDLE = handle
        getters = (
            click_compat.get_text_stdout,
            getattr(click_compat, "_default_text_stdout", None),
        )
    elif fd == 2:
        click_winconsole.STDERR_HANDLE = handle
        getters = (
            click_compat.get_text_stderr,
            getattr(click_compat, "_default_text_stderr", None),
        )
    else:
        return

    seen: set[int] = set()
    for getter in getters:
        if getter is None:
            continue
        try:
            stream = getter()
        except Exception:
            continue
        _replace_click_console_handle(stream, handle, seen)


def _replace_click_console_handle(obj: object, handle: object, seen: set[int]) -> None:
    ident = id(obj)
    if ident in seen:
        return
    seen.add(ident)

    if hasattr(obj, "handle"):
        try:
            obj.handle = handle
        except Exception:
            # Best effort: not every object in Click's stream wrapper graph is mutable.
            pass

    for attr in (
        "_text_stream",
        "buffer",
        "raw",
        "wrapped",
        "stream",
        "_StreamWrapper__wrapped",
    ):
        try:
            child = getattr(obj, attr)
        except Exception:
            continue
        if child is not None and child is not obj:
            _replace_click_console_handle(child, handle, seen)


def _write_all(fd: int, data: bytes) -> None:
    while data:
        written = os.write(fd, data)
        data = data[written:]


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        logger.debug("Could not close fd %d", fd, exc_info=True)
