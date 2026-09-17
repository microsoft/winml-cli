# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Exercise real disposable process trees and injected OS failure paths."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import socket
import subprocess
import sys
import textwrap
from contextlib import ExitStack, nullcontext, suppress
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "e2e_eval" / "utils" / "process_tree.py"
)
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="Requires Windows Job Objects")


@pytest.fixture(scope="module")
def process_tree():
    spec = importlib.util.spec_from_file_location("_e2e_process_tree", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def live_tree(tmp_path):
    """Socket EOF proves descendant death; test teardown also releases workers."""
    with ExitStack() as stack:
        listener = stack.enter_context(socket.socket())
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)
        host, port = listener.getsockname()
        worker = textwrap.dedent("""\
            import socket
            import subprocess
            import sys

            if sys.argv[3]:
                subprocess.Popen([sys.executable, "-c", sys.argv[3], *sys.argv[1:3], ""])
            with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=15) as conn:
                conn.sendall(b"R")
                conn.recv(1)
            """)
        root = textwrap.dedent(f"""\
            import os
            import subprocess
            import sys
            import time

            worker = {worker!r}
            subprocess.Popen([sys.executable, "-c", worker, {host!r}, {str(port)!r}, worker])
            if sys.argv[1] == "exit":
                os._exit(len(sys.argv[1]))
            time.sleep(15)
            """)
        compile(worker, "<tree-worker>", "exec")
        compile(root, "<tree-root>", "exec")
        connections = []

        def ready():
            for _ in range(2):
                conn, _address = listener.accept()
                stack.enter_context(conn)
                conn.settimeout(5)
                assert conn.recv(1) == b"R", "Descendant exited before announcing readiness"
                connections.append(conn)

        def assert_stopped():
            assert len(connections) == 2
            for conn in connections:
                # Windows may reset a socket when terminating its owner.
                with suppress(ConnectionResetError):
                    assert conn.recv(1) == b""

        yield SimpleNamespace(
            args=[sys.executable, "-c", root],
            kwargs={
                "stdin": stack.enter_context((tmp_path / "stdin").open("w+b")),
                "stdout": stack.enter_context((tmp_path / "stdout").open("w+b")),
                "stderr": stack.enter_context((tmp_path / "stderr").open("w+b")),
            },
            ready=ready,
            assert_stopped=assert_stopped,
        )


@pytest.mark.parametrize("input_size", [0, 31])
def test_normal_output_and_exit_code_match_popen(process_tree, tmp_path, input_size):
    code = (
        "import os, sys; data = sys.stdin.buffer.read(); "
        "sys.stdout.buffer.write(os.environ['E2E_PROCESS_TREE_VALUE'].encode() + data[::-1]); "
        "sys.stderr.buffer.write(data.hex().encode()); "
        "sys.exit(len(data) % 17)"
    )
    compile(code, "<stream-worker>", "exec")
    args = [sys.executable, "-c", code]
    env = MappingProxyType({**os.environ, "E2E_PROCESS_TREE_VALUE": tmp_path.name})
    with ExitStack() as stack:
        source = stack.enter_context((tmp_path / "input").open("w+b"))
        source.write(bytes(range(input_size)))
        results = []
        for name in ("baseline", "managed"):
            source.seek(0)
            stdout = stack.enter_context((tmp_path / f"{name}.out").open("w+b"))
            stderr = stack.enter_context((tmp_path / f"{name}.err").open("w+b"))
            kwargs = {"stdin": source, "stdout": stdout, "stderr": stderr, "env": env}
            if name == "baseline":
                with subprocess.Popen(args, **kwargs) as process:  # noqa: S603
                    exit_code = process.wait(timeout=5)
            else:
                with process_tree.ManagedProcess(args, **kwargs) as owner:
                    assert isinstance(owner.process, subprocess.Popen)
                    exit_code = owner.process.wait(timeout=5)
                assert owner._closed
                assert owner.process.poll() == exit_code
                assert owner.process.wait(timeout=0) == exit_code
                if sys.platform == "win32":
                    assert not owner.process._handle.closed
            assert not source.closed and not stdout.closed and not stderr.closed
            stdout.seek(0)
            stderr.seek(0)
            results.append((exit_code, stdout.read(), stderr.read()))
        assert results[1] == results[0]


@pytest.mark.parametrize("scenario", ["terminate", "timeout", "root-exit", "exception"])
def test_cleanup_reaches_children_and_grandchildren(process_tree, live_tree, scenario):
    mode = "exit" if scenario == "root-exit" else "wait"
    owner = process_tree.ManagedProcess([*live_tree.args, mode], **live_tree.kwargs)
    try:
        live_tree.ready()
        if scenario == "root-exit":
            exit_code = owner.process.wait(timeout=5)
            assert exit_code == len(mode)
            owner.close()
            assert owner.process.returncode == exit_code
        elif scenario == "terminate":
            owner.terminate()
            owner.terminate()
            live_tree.assert_stopped()
            owner.close()
        elif scenario == "timeout":
            with pytest.raises(subprocess.TimeoutExpired), owner:
                owner.process.wait(timeout=0.01)
        else:
            with pytest.raises(RuntimeError, match="caller failed"), owner:
                raise RuntimeError("caller failed")
        live_tree.assert_stopped()
        assert owner.process.returncode is not None
        assert owner._closed
    finally:
        owner.close()


def test_spawn_error_is_not_hidden(process_tree, tmp_path):
    with pytest.raises(FileNotFoundError):
        process_tree.ManagedProcess([str(tmp_path / "missing-executable")])


@pytest.mark.parametrize("final_timeout", [False, True])
def test_root_waits_are_bounded_and_close_is_release_once(process_tree, monkeypatch, final_timeout):
    process = Mock(pid=os.getpid(), returncode=None)
    error = subprocess.TimeoutExpired("disposable", 2)
    process.wait.side_effect = [error, error if final_timeout else 0]
    monkeypatch.setattr(process_tree.subprocess, "Popen", Mock(return_value=process))
    # No real process or group may be targeted by this injected wait failure.
    monkeypatch.setattr(process_tree, "_WINDOWS", True)
    job = Mock()
    monkeypatch.setattr(process_tree, "_WindowsJob", Mock(return_value=job))
    owner = process_tree.ManagedProcess(["disposable"])
    process_tree.subprocess.Popen.assert_called_once_with(
        ["disposable"], stdin=None, stdout=None, stderr=None, env=None,
        creationflags=process_tree._CREATE_SUSPENDED | process_tree._CREATE_NO_WINDOW,
        start_new_session=False,
    )
    job.assign_and_resume.assert_called_once_with(process)
    with pytest.raises(subprocess.TimeoutExpired) if final_timeout else nullcontext():
        owner.close()
    assert [call.kwargs["timeout"] for call in process.wait.call_args_list] == [2.0, 2.0]
    process.kill.assert_called_once_with()
    assert owner._closed
    owner.close()
    owner.terminate()
    assert process.wait.call_count == 2
    job.terminate.assert_called_once_with()
    job.close.assert_called_once_with()


@pytest.mark.parametrize("root_exited", [False, True])
@pytest.mark.parametrize("group_error", [None, ProcessLookupError, PermissionError])
def test_posix_group_cleanup_reaps_once_regardless_of_root_liveness(
    process_tree, monkeypatch, root_exited, group_error,
):
    monkeypatch.setattr(process_tree, "_WINDOWS", False)
    process = Mock(pid=os.getpid(), returncode=0 if root_exited else None)
    popen = Mock(return_value=process)
    killpg = Mock(side_effect=group_error)
    monkeypatch.setattr(process_tree.subprocess, "Popen", popen)
    monkeypatch.setattr(process_tree.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(process_tree.signal, "SIGKILL", 9, raising=False)
    owner = process_tree.ManagedProcess(["disposable"])
    popen.assert_called_once_with(
        ["disposable"], stdin=None, stdout=None, stderr=None, env=None,
        creationflags=0, start_new_session=True,
    )
    with pytest.raises(PermissionError) if group_error is PermissionError else nullcontext():
        owner.close()
    assert owner._closed
    owner.close()
    owner.terminate()
    killpg.assert_called_once_with(process.pid, process_tree.signal.SIGKILL)
    process.poll.assert_not_called()
    process.wait.assert_called_once_with(timeout=2.0)


@pytest.fixture
def windows_api(monkeypatch, process_tree):
    """Wrap real DLLs so injected failures still exercise native resource cleanup."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **_kwargs: {
        "kernel32": kernel32, "ntdll": ntdll,
    }[name])
    created = []
    real_popen = subprocess.Popen

    def spawn(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(process_tree.subprocess, "Popen", Mock(side_effect=spawn))
    real_close = kernel32.CloseHandle
    real_close.argtypes = [ctypes.wintypes.HANDLE]
    real_close.restype = ctypes.wintypes.BOOL
    monkeypatch.setattr(kernel32, "CloseHandle", Mock(side_effect=real_close))
    yield SimpleNamespace(kernel32=kernel32, ntdll=ntdll, created=created)
    # Independent fallback ensures a failing test never leaves suspended roots.
    for process in created:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)


@WINDOWS_ONLY
@pytest.mark.parametrize("stage", ["assign", "resume"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_windows_initialization_failure_cleans_resources(
    process_tree, windows_api, monkeypatch, tmp_path, stage, interrupt,
):
    api = windows_api
    marker = tmp_path / "must-not-run"
    code = "from pathlib import Path; import sys; Path(sys.argv[1]).touch()"
    compile(code, "<suspended-root>", "exec")
    args = [sys.executable, "-c", code, str(marker)]
    error_type = KeyboardInterrupt if interrupt else OSError
    dll, function, result = {
        "assign": (api.kernel32, "AssignProcessToJobObject", 0),
        "resume": (api.ntdll, "NtResumeProcess", ctypes.c_long(0xC0000001).value),
    }[stage]
    failure = Mock(return_value=result, side_effect=KeyboardInterrupt if interrupt else None)
    monkeypatch.setattr(dll, function, failure)
    with pytest.raises(error_type):
        process_tree.ManagedProcess(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert not marker.exists()
    api.kernel32.CloseHandle.assert_called_once()
    process, = api.created
    assert process.returncode is not None
    assert not process._handle.closed
    assert process.wait(timeout=0) == process.returncode


@WINDOWS_ONLY
@pytest.mark.parametrize("close_failure", [False, True])
def test_windows_signatures_full_width_handles_and_single_close(
    process_tree, monkeypatch, close_failure,
):
    from ctypes import wintypes

    job_handle = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8 - 2)) + 123
    process_handle = job_handle + 1
    kernel32 = SimpleNamespace(**{name: Mock(return_value=1) for name in (
        "SetInformationJobObject", "AssignProcessToJobObject", "TerminateJobObject", "CloseHandle",
    )})
    kernel32.CreateJobObjectW = Mock(return_value=job_handle)
    ntdll = SimpleNamespace(NtResumeProcess=Mock(return_value=0))
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **_kwargs: {
        "kernel32": kernel32, "ntdll": ntdll,
    }[name])
    events = Mock()
    events.attach_mock(kernel32.AssignProcessToJobObject, "assign")
    events.attach_mock(ntdll.NtResumeProcess, "resume")
    job = process_tree._WindowsJob()
    job.assign_and_resume(SimpleNamespace(_handle=process_handle))
    job.terminate()
    kernel32.CloseHandle.return_value = not close_failure
    with pytest.raises(OSError) if close_failure else nullcontext():
        job.close()
    if not close_failure:
        job.close()
    signatures = [
        (kernel32.CreateJobObjectW, [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        (kernel32.SetInformationJobObject,
         [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        (kernel32.AssignProcessToJobObject, [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        (kernel32.TerminateJobObject, [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        (kernel32.CloseHandle, [wintypes.HANDLE], wintypes.BOOL),
        (ntdll.NtResumeProcess, [wintypes.HANDLE], wintypes.LONG),
    ]
    for function, argtypes, restype in signatures:
        assert function.argtypes == argtypes
        assert function.restype is restype
    kernel32.CreateJobObjectW.assert_called_once_with(None, None)
    assert [call[0] for call in events.mock_calls] == ["assign", "resume"]
    assert kernel32.AssignProcessToJobObject.call_args.args[0] == job_handle
    assert kernel32.AssignProcessToJobObject.call_args.args[1].value == process_handle
    assert ntdll.NtResumeProcess.call_args.args[0].value == process_handle
    kernel32.CloseHandle.assert_called_once_with(job_handle)
    info_call = kernel32.SetInformationJobObject.call_args.args
    info_pointer = ctypes.cast(info_call[2], ctypes.POINTER(process_tree._ExtendedLimitInformation))
    info = info_pointer.contents
    assert info.BasicLimitInformation.LimitFlags == process_tree._KILL_ON_JOB_CLOSE
    assert info_call[3] == ctypes.sizeof(info)
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(info) == 144


@WINDOWS_ONLY
@pytest.mark.parametrize("error_type", [None, RuntimeError, KeyboardInterrupt])
def test_windows_terminate_failure_still_closes_job_and_reaps(
    process_tree, windows_api, monkeypatch, live_tree, error_type,
):
    owner = process_tree.ManagedProcess([*live_tree.args, "wait"], **live_tree.kwargs)
    try:
        live_tree.ready()
        monkeypatch.setattr(windows_api.kernel32, "TerminateJobObject", Mock(return_value=0))
        if error_type is None:
            with pytest.raises(OSError):
                owner.close()
        else:
            primary = error_type("caller failed")
            with pytest.raises(error_type) as caught, owner:
                raise primary
            assert caught.value is primary
            assert len(primary.__notes__) == 1
        live_tree.assert_stopped()
        assert owner.process.returncode is not None
        assert not owner.process._handle.closed
        assert owner._closed
        owner.close()
        owner.terminate()
        windows_api.kernel32.TerminateJobObject.assert_called_once()
        windows_api.kernel32.CloseHandle.assert_called_once()
    finally:
        owner.close()


@WINDOWS_ONLY
@pytest.mark.parametrize("root_mode", ["wait", "exit"])
def test_windows_parent_abnormal_exit_kills_job(live_tree, root_mode):
    # No context cleanup: the OS must release the killed parent's job handle.
    code = textwrap.dedent("""\
        import importlib.util
        import sys
        import time

        spec = importlib.util.spec_from_file_location("process_tree", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        owner = module.ManagedProcess(sys.argv[2:])
        time.sleep(15)
        """)
    compile(code, "<abnormal-parent>", "exec")
    args = [sys.executable, "-c", code, str(MODULE_PATH), *live_tree.args, root_mode]
    with subprocess.Popen(args, **live_tree.kwargs) as parent:  # noqa: S603
        try:
            live_tree.ready()
            parent.kill()
            parent.wait(timeout=2)
            live_tree.assert_stopped()
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=2)
