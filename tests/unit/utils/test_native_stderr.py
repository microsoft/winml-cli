# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml.modelkit.utils.native_stderr."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

import pytest

from winml.modelkit.utils import native_stderr as native_stderr_module
from winml.modelkit.utils.native_stderr import (
    capture_native_stderr,
    suppress_native_stderr,
)


@pytest.fixture(autouse=True)
def _restore_root_logger_level():
    root = logging.getLogger()
    level = root.level
    yield
    root.setLevel(level)


class TestSuppressNativeStderr:
    """Tests for suppress_native_stderr (devnull-based)."""

    def test_suppresses_native_stderr(self, capfd):
        with suppress_native_stderr():
            os.write(2, b"should be discarded\n")
        assert "should be discarded" not in capfd.readouterr().err

    def test_stderr_works_after_context(self, capfd):
        with suppress_native_stderr():
            pass
        os.write(2, b"after\n")
        assert "after" in capfd.readouterr().err

    def test_serializes_concurrent_warning_and_stderr_redirects(self, monkeypatch):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)
        first_entered = threading.Event()
        first_can_exit = threading.Event()
        second_entered_while_first_active = False

        def first_redirect() -> None:
            with native_stderr_module.suppress_native_warnings(enabled=True):
                first_entered.set()
                assert first_can_exit.wait(timeout=5)

        def second_redirect() -> None:
            nonlocal second_entered_while_first_active
            assert first_entered.wait(timeout=5)
            with suppress_native_stderr():
                second_entered_while_first_active = not first_can_exit.is_set()

        first = threading.Thread(target=first_redirect)
        second = threading.Thread(target=second_redirect)
        first.start()
        assert first_entered.wait(timeout=5)
        second.start()
        time.sleep(0.1)
        assert not second_entered_while_first_active
        first_can_exit.set()
        first.join(timeout=5)
        second.join(timeout=5)

        assert not first.is_alive()
        assert not second.is_alive()
        assert not second_entered_while_first_active

    def test_disabled_leaves_native_stderr_visible(self, capfd):
        with suppress_native_stderr(enabled=False):
            os.write(2, b"visible when disabled\n")
        assert "visible when disabled" in capfd.readouterr().err

    def test_setup_failure_does_not_abort_wrapped_code(self, monkeypatch):
        monkeypatch.setattr(native_stderr_module.sys, "platform", "win32")

        def fail_dup(fd: int) -> int:
            raise OSError("dup failed")

        monkeypatch.setattr(native_stderr_module.os, "dup", fail_dup)
        ran = False

        with suppress_native_stderr():
            ran = True

        assert ran

    def test_redirect_failure_does_not_abort_wrapped_code(self, monkeypatch):
        ran = False

        with monkeypatch.context() as m:
            m.setattr(native_stderr_module.sys, "platform", "win32")

            def fail_dup2(src: int, dst: int) -> None:
                raise OSError("dup2 failed")

            m.setattr(native_stderr_module.os, "dup2", fail_dup2)

            with suppress_native_stderr():
                ran = True

        assert ran

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 only")
    def test_win32_std_error_handle_usable_after_restore(self):
        with suppress_native_stderr():
            pass
        _write_win32_stderr(b"after restore\n")

    @pytest.mark.skipif(sys.platform == "win32", reason="Non-Windows only")
    def test_noop_on_non_windows(self, capfd):
        with suppress_native_stderr():
            os.write(2, b"passthrough\n")
        assert "passthrough" in capfd.readouterr().err


class TestCaptureNativeStderr:
    """Tests for capture_native_stderr (pipe-based, re-logs)."""

    def test_captures_and_logs(self, caplog):
        with (
            caplog.at_level(logging.INFO, logger="winml.modelkit.utils.native_stderr"),
            capture_native_stderr(logging.INFO),
        ):
            os.write(2, b"hello\nworld\n")
        assert any("hello" in r.message for r in caplog.records)
        assert any("world" in r.message for r in caplog.records)

    def test_strips_ansi(self, caplog):
        with (
            caplog.at_level(logging.INFO, logger="winml.modelkit.utils.native_stderr"),
            capture_native_stderr(logging.INFO),
        ):
            os.write(2, b"\x1b[31mred message\x1b[0m\n")
        assert any("red message" in r.message for r in caplog.records)

    def test_stderr_works_after_context(self, capfd):
        with capture_native_stderr():
            pass
        os.write(2, b"after\n")
        assert "after" in capfd.readouterr().err

    def test_pipe_setup_failure_does_not_abort_wrapped_code(self, monkeypatch):
        monkeypatch.setattr(native_stderr_module.sys, "platform", "win32")

        def fail_pipe() -> tuple[int, int]:
            raise OSError("pipe failed")

        monkeypatch.setattr(native_stderr_module.os, "pipe", fail_pipe)
        ran = False

        with capture_native_stderr():
            ran = True

        assert ran

    def test_redirect_failure_does_not_abort_wrapped_code(self, monkeypatch):
        ran = False

        with monkeypatch.context() as m:
            m.setattr(native_stderr_module.sys, "platform", "win32")

            def fail_dup2(src: int, dst: int) -> None:
                raise OSError("dup2 failed")

            m.setattr(native_stderr_module.os, "dup2", fail_dup2)

            with capture_native_stderr():
                ran = True

        assert ran

    def test_skips_blank_lines(self, caplog):
        with (
            caplog.at_level(logging.INFO, logger="winml.modelkit.utils.native_stderr"),
            capture_native_stderr(logging.INFO),
        ):
            os.write(2, b"  \n\nkeep\n  \n")
        messages = [r.message for r in caplog.records]
        assert any("keep" in m for m in messages)
        assert not any(m == "  " for m in messages)

    @pytest.mark.timeout(30)
    def test_no_deadlock_on_large_output(self, caplog):
        """Regression: a native writer emitting more than the OS pipe buffer
        (~64 KB on Windows) inside the context must not stall.

        The pipe used to be drained only after the wrapped block returned, so a
        native write() that filled the buffer blocked forever -- observed as an
        indefinite stall while an EP compiled a model. The reader thread now
        drains concurrently, so this loop completes immediately. The timeout
        marker turns a reintroduced deadlock into a failure instead of a hang.
        """
        line = b"[ORT] partitioning subgraph node ...\n"
        written = 0
        with (
            caplog.at_level(logging.INFO, logger="winml.modelkit.utils.native_stderr"),
            capture_native_stderr(logging.INFO),
        ):
            for _ in range(20000):  # ~720 KB, dwarfs the ~64 KB pipe buffer
                os.write(2, line)
                written += len(line)
        assert written > 64 * 1024
        # On Windows the redirected output is re-logged; elsewhere the context is
        # a no-op. Either way the point is that the loop above did not deadlock.
        if sys.platform == "win32":
            assert any("partitioning subgraph" in r.message for r in caplog.records)


class TestSuppressNativeWarnings:
    """Tests for warning-only native stderr suppression."""

    def test_warning_filtering_is_opt_in(self, monkeypatch, capfd):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)

        with native_stderr_module.suppress_native_warnings():
            os.write(2, b"2026 [W:custom-native:, file.cc:1 WarningFunc] library warning\n")

        assert "library warning" in capfd.readouterr().err

    def test_filters_native_warning_lines_and_preserves_errors(self, monkeypatch, capfd):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)

        with native_stderr_module.suppress_native_warnings(enabled=True):
            os.write(2, b"2026 [W:custom-native:, file.cc:1 WarningFunc] noisy warning\n")
            os.write(2, b"2026 [E:onnxruntime:, qnn_backend.cc:2 ErrorFunc] useful error\n")
            os.write(2, b"plain diagnostic\n")

        stderr = capfd.readouterr().err
        assert "noisy warning" not in stderr
        assert "useful error" in stderr
        assert "plain diagnostic" in stderr

    def test_uses_file_backed_capture_instead_of_pipe(self, monkeypatch):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)
        pipe_called = False

        def fail_pipe() -> tuple[int, int]:
            nonlocal pipe_called
            pipe_called = True
            raise AssertionError("warning suppression must not create a pipe")

        monkeypatch.setattr(native_stderr_module.os, "pipe", fail_pipe)

        with native_stderr_module.suppress_native_warnings(enabled=True):
            os.write(2, b"2026 [W:custom-native:, file.cc:1 WarningFunc] hidden\n")

        assert pipe_called is False

    def test_filters_native_prefix_info_without_dropping_python_stderr(self, monkeypatch, capfd):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)

        with native_stderr_module.suppress_native_warnings(enabled=True):
            os.write(2, b"DSP_INFO UNSUPPORTED_KEY: 49\n")
            os.write(2, b"plain Python diagnostic\n")

        stderr = capfd.readouterr().err
        assert "DSP_INFO" not in stderr
        assert "plain Python diagnostic" in stderr

    def test_preserves_error_lines_that_reference_warning_tokens(self, monkeypatch, capfd):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)

        with native_stderr_module.suppress_native_warnings(enabled=True):
            os.write(
                2,
                b"2026 [E:onnxruntime:, qnn.cc:2 ErrorFunc] failed after previous [W:note]\n",
            )

        assert "failed after previous [W:note]" in capfd.readouterr().err

    def test_can_filter_unclassified_native_diagnostics(self, monkeypatch, capfd):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)

        with native_stderr_module.suppress_native_warnings(
            enabled=True, preserve_unclassified=False
        ):
            os.write(2, b"DSP_INFO UNSUPPORTED_KEY: 49\n")
            os.write(2, b"2026 [E:custom-native:, file.cc:2 ErrorFunc] useful error\n")

        stderr = capfd.readouterr().err
        assert "DSP_INFO" not in stderr
        assert "useful error" in stderr

    def test_verbose_logging_leaves_native_warnings_visible(self, monkeypatch, capfd):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.INFO)

        with native_stderr_module.suppress_native_warnings(enabled=True):
            os.write(2, b"2026 [W:custom-native:, file.cc:1 WarningFunc] visible warning\n")

        assert "visible warning" in capfd.readouterr().err

    def test_show_all_warnings_env_leaves_native_warnings_visible(self, monkeypatch, capfd):
        monkeypatch.setenv("WINMLCLI_SHOW_ALL_WARNINGS", "1")
        logging.getLogger().setLevel(logging.WARNING)

        with native_stderr_module.suppress_native_warnings(enabled=True):
            os.write(2, b"2026 [W:custom-native:, file.cc:1 WarningFunc] env warning\n")

        assert "env warning" in capfd.readouterr().err

    def test_capture_setup_failure_does_not_abort_wrapped_code(self, monkeypatch):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)

        def fail_capture(*args: object, **kwargs: object):
            raise OSError(1, "Incorrect function")

        monkeypatch.setattr(native_stderr_module.tempfile, "TemporaryFile", fail_capture)

        ran = False
        with native_stderr_module.suppress_native_warnings(enabled=True):
            ran = True

        assert ran

    def test_restore_failure_closes_redirected_fd(self, monkeypatch):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)
        closed: list[int] = []
        dup2_calls = 0

        def fake_dup2(src: int, dst: int) -> None:
            nonlocal dup2_calls
            dup2_calls += 1
            if dup2_calls == 2:
                raise OSError(1, "restore failed")

        monkeypatch.setattr(native_stderr_module.os, "dup", lambda fd: 12)
        monkeypatch.setattr(native_stderr_module.os, "dup2", fake_dup2)
        monkeypatch.setattr(native_stderr_module.os, "close", lambda fd: closed.append(fd))
        monkeypatch.setattr(
            native_stderr_module,
            "_set_win32_std_handle_to_current_fd",
            lambda fd: None,
        )
        monkeypatch.setattr(
            native_stderr_module,
            "_refresh_click_windows_console_stream",
            lambda fd, handle=None: None,
        )

        with native_stderr_module.suppress_native_warnings(enabled=True):
            pass

        assert 2 in closed
        assert 12 in closed

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 only")
    def test_win32_std_handle_sync_failure_does_not_abort_wrapped_code(self, monkeypatch):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)

        def fail_get_osfhandle(fd: int) -> int:
            raise OSError(1, "Incorrect function")

        monkeypatch.setattr(
            native_stderr_module.msvcrt,
            "get_osfhandle",
            fail_get_osfhandle,
        )

        ran = False
        with native_stderr_module.suppress_native_warnings(enabled=True):
            ran = True

        assert ran

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 native stderr only")
    def test_filters_win32_std_error_handle_warning(self, monkeypatch, capfd):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)

        with native_stderr_module.suppress_native_warnings(enabled=True):
            _write_win32_stderr(b"2026 [W:custom-native:, file.cc:1 WarningFunc] win32 warning\n")
            _write_win32_stderr(b"2026 [E:custom-native:, file.cc:2 ErrorFunc] win32 error\n")

        stderr = capfd.readouterr().err
        assert "win32 warning" not in stderr
        assert "win32 error" in stderr

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 native stderr only")
    def test_win32_std_error_handle_usable_after_exception(self, monkeypatch):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)

        def fail_with_runtime_error() -> None:
            raise RuntimeError("boom")

        with (
            pytest.raises(RuntimeError, match="boom"),
            native_stderr_module.suppress_native_warnings(enabled=True),
        ):
            fail_with_runtime_error()

        _write_win32_stderr(b"after exception\n")

    def test_set_win32_std_handle_accepts_null_handle(self, monkeypatch):
        calls = []
        std_error_handle = object()

        class FakeKernel32:
            def __init__(self) -> None:
                self.SetStdHandle = self._set_std_handle

            def _set_std_handle(self, handle_kind: object, handle: object | None) -> bool:
                calls.append((handle_kind, handle))
                return True

        monkeypatch.setattr(native_stderr_module.sys, "platform", "win32")
        monkeypatch.setattr(native_stderr_module, "_k32", FakeKernel32(), raising=False)
        monkeypatch.setattr(
            native_stderr_module,
            "_STD_ERROR_HANDLE",
            std_error_handle,
            raising=False,
        )

        native_stderr_module._set_win32_std_handle(2, None)

        assert calls == [(std_error_handle, None)]

    @pytest.mark.skipif(sys.platform != "win32", reason="Click Win32 console only")
    def test_refreshes_click_console_handle_cache(self, monkeypatch):
        import click._compat as click_compat
        import click._winconsole as click_winconsole

        class FakeRaw:
            handle = 1

        class FakeBuffer:
            raw = FakeRaw()

        class FakeText:
            buffer = FakeBuffer()

        class FakeConsoleStream:
            _text_stream = FakeText()

        stream = FakeConsoleStream()

        monkeypatch.setattr(click_compat, "get_text_stderr", lambda: stream)
        monkeypatch.setattr(click_winconsole, "STDERR_HANDLE", 1)
        monkeypatch.setattr(native_stderr_module.msvcrt, "get_osfhandle", lambda fd: 12345)

        native_stderr_module._refresh_click_windows_console_stream(2)

        assert click_winconsole.STDERR_HANDLE == 12345
        assert stream._text_stream.buffer.raw.handle == 12345

    @pytest.mark.skipif(sys.platform != "win32", reason="Click Win32 console only")
    def test_refreshes_click_default_console_handle_cache(self, monkeypatch):
        import click._compat as click_compat
        import click._winconsole as click_winconsole

        class FakeRaw:
            def __init__(self) -> None:
                self.handle = 1

        class FakeBuffer:
            def __init__(self) -> None:
                self.raw = FakeRaw()

        class FakeText:
            def __init__(self) -> None:
                self.buffer = FakeBuffer()

        class FakeConsoleStream:
            def __init__(self) -> None:
                self._text_stream = FakeText()

        uncached_stream = FakeConsoleStream()
        default_cached_stream = FakeConsoleStream()

        monkeypatch.setattr(click_compat, "get_text_stderr", lambda: uncached_stream)
        monkeypatch.setattr(click_compat, "_default_text_stderr", lambda: default_cached_stream)
        monkeypatch.setattr(click_winconsole, "STDERR_HANDLE", 1)
        monkeypatch.setattr(native_stderr_module.msvcrt, "get_osfhandle", lambda fd: 12345)

        native_stderr_module._refresh_click_windows_console_stream(2)

        assert click_winconsole.STDERR_HANDLE == 12345
        assert uncached_stream._text_stream.buffer.raw.handle == 12345
        assert default_cached_stream._text_stream.buffer.raw.handle == 12345

    @pytest.mark.timeout(30)
    def test_no_deadlock_on_large_warning_output(self, monkeypatch, capfd):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)
        warning_line = b"2026 [W:custom-native:, file.cc:1 WarningFunc] noisy warning\n"
        error_line = b"2026 [E:custom-native:, file.cc:2 ErrorFunc] useful error\n"
        written = 0

        with native_stderr_module.suppress_native_warnings(enabled=True):
            for _ in range(20000):
                os.write(2, warning_line)
                written += len(warning_line)
            os.write(2, error_line)

        assert written > 64 * 1024
        stderr = capfd.readouterr().err
        assert "noisy warning" not in stderr
        assert "useful error" in stderr

    def test_nested_warning_suppression_is_reentrant(self, monkeypatch):
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)
        started = time.monotonic()

        with (
            native_stderr_module.suppress_native_warnings(enabled=True),
            native_stderr_module.suppress_native_warnings(enabled=True),
        ):
            pass

        assert time.monotonic() - started < 1.0


def _write_win32_stderr(data: bytes) -> None:
    import ctypes.wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetStdHandle.argtypes = [ctypes.wintypes.DWORD]
    k32.GetStdHandle.restype = ctypes.wintypes.HANDLE
    k32.WriteFile.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.wintypes.LPCVOID,
        ctypes.wintypes.DWORD,
        ctypes.POINTER(ctypes.wintypes.DWORD),
        ctypes.wintypes.LPVOID,
    ]
    k32.WriteFile.restype = ctypes.wintypes.BOOL

    std_error_handle = ctypes.wintypes.DWORD(0xFFFFFFF4)
    written = ctypes.wintypes.DWORD(0)
    buffer = ctypes.create_string_buffer(data)
    ok = k32.WriteFile(
        k32.GetStdHandle(std_error_handle),
        buffer,
        len(data),
        ctypes.byref(written),
        None,
    )
    assert ok
    assert written.value == len(data)
