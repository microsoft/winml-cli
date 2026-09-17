# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Own disposable subprocess trees without enumerating processes or open files."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
from ctypes import wintypes
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType
    from typing import IO


__all__ = ["ManagedProcess"]

_WINDOWS = sys.platform == "win32"
_CREATE_SUSPENDED = 0x00000004
_CREATE_NO_WINDOW = 0x08000000
_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_WAIT_SECONDS = 2.0


def _cleanup(*actions: Callable[[], None], error: BaseException | None = None) -> None:
    primary = error
    for action in actions:
        try:
            action()
        except BaseException as cleanup_error:
            if primary is None:
                primary = cleanup_error
            else:
                primary.add_note(f"Process cleanup also failed: {cleanup_error!r}")
    if primary is not None and error is None:
        raise primary


class _IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IOCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsJob:
    """Keep the sole, non-inheritable handle to a kill-on-close job."""

    def __init__(self) -> None:
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        self._kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self._kernel32.TerminateJobObject.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        self._ntdll.NtResumeProcess.restype = wintypes.LONG

        self._handle: int | None = self._kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info = _ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
            if not self._kernel32.SetInformationJobObject(
                self._handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException as error:
            _cleanup(self.close, error=error)
            raise

    def assign_and_resume(self, process: subprocess.Popen) -> None:
        handle = wintypes.HANDLE(process._handle)
        if not self._kernel32.AssignProcessToJobObject(self._handle, handle):
            raise ctypes.WinError(ctypes.get_last_error())
        status = self._ntdll.NtResumeProcess(handle)
        if status < 0:
            raise OSError(f"NtResumeProcess failed with NTSTATUS 0x{status & 0xFFFFFFFF:08X}")

    def terminate(self) -> None:
        if self._handle is not None and not self._kernel32.TerminateJobObject(self._handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self._handle is not None:
            if not self._kernel32.CloseHandle(self._handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self._handle = None


class ManagedProcess:
    """Own a subprocess tree with single-threaded, release-once cleanup.

    Windows contains the suspended root in a kill-on-close job, also covering owner death.
    POSIX descendants must stay in the group; owner death alone does not kill it.
    Cleanup preserves primary exceptions, bounds waits, and never drains caller-owned streams.
    """

    def __init__(
        self, args: list[str], *,
        stdin: IO[bytes] | int | None = None,
        stdout: IO[bytes] | int | None = None,
        stderr: IO[bytes] | int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._process: subprocess.Popen | None = None
        self._closed = False
        self._job = _WindowsJob() if _WINDOWS else None
        try:
            self._process = subprocess.Popen(  # noqa: S603
                args, stdin=stdin, stdout=stdout, stderr=stderr, env=env,
                creationflags=_CREATE_SUSPENDED | _CREATE_NO_WINDOW if _WINDOWS else 0,
                start_new_session=not _WINDOWS,
            )
            if self._job is not None:
                self._job.assign_and_resume(self._process)
        except BaseException as error:
            # Assignment failure can leave the suspended root outside the job.
            if self._process is not None:
                _cleanup(self._process.kill, error=error)
            _cleanup(self.close, error=error)
            raise

    @property
    def process(self) -> subprocess.Popen:
        """Return the underlying process for polling, waiting and its exit code."""
        if self._process is None:
            raise RuntimeError("The subprocess has not been created")
        return self._process

    def terminate(self) -> None:
        """Forcibly kill the owned tree, without waiting or inspecting descendants."""
        if self._closed:
            return
        if self._job is not None:
            self._job.terminate()
        elif self._process is not None:
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _reap_root(self) -> None:
        if self._process is None:
            return
        try:
            self._process.wait(timeout=_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=_WAIT_SECONDS)

    def close(self) -> None:
        """Attempt tree cleanup and root reaping once, even when either fails."""
        if self._closed:
            return
        actions = [self.terminate]
        if self._job is not None:
            actions.append(self._job.close)
        try:
            _cleanup(*actions, self._reap_root)
        finally:
            self._closed = True

    def __enter__(self) -> ManagedProcess:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc_value: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        _cleanup(self.close, error=exc_value)
