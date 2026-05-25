# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml.modelkit.utils.native_stderr."""

from __future__ import annotations

import logging
import os
import sys

import pytest

from winml.modelkit.utils.native_stderr import (
    _ort_startup_logs,
    replay_ort_startup_logs,
    suppress_ep_registration_stderr,
)


@pytest.fixture(autouse=True)
def _clear_startup_logs():
    """Ensure a clean buffer for every test."""
    _ort_startup_logs.clear()
    yield
    _ort_startup_logs.clear()


class TestSuppressEpRegistrationStderr:
    """Tests for suppress_ep_registration_stderr context manager."""

    def test_captures_native_stderr(self):
        with suppress_ep_registration_stderr():
            os.write(2, b"hello\nworld\n")
        assert "hello" in _ort_startup_logs
        assert "world" in _ort_startup_logs

    def test_strips_ansi(self):
        with suppress_ep_registration_stderr():
            os.write(2, b"\x1b[31mred message\x1b[0m\n")
        assert "red message" in _ort_startup_logs

    def test_stderr_works_after_context(self, capfd):
        with suppress_ep_registration_stderr():
            pass
        os.write(2, b"after\n")
        assert "after" in capfd.readouterr().err

    def test_skips_blank_lines(self):
        with suppress_ep_registration_stderr():
            os.write(2, b"  \n\nkeep\n  \n")
        assert _ort_startup_logs == ["keep"]

    def test_startup_buffer_disabled(self):
        with suppress_ep_registration_stderr(use_startup_buffer=False):
            os.write(2, b"should not buffer\n")
        assert _ort_startup_logs == []

    def test_custom_log_level(self, caplog):
        with (
            caplog.at_level(logging.INFO, logger="winml.modelkit.utils.native_stderr"),
            suppress_ep_registration_stderr(logging.INFO),
        ):
            os.write(2, b"info line\n")
        assert any("info line" in r.message for r in caplog.records)

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 only")
    def test_win32_std_error_handle_restored(self):
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetStdHandle.argtypes = [wintypes.DWORD]
        k32.GetStdHandle.restype = wintypes.HANDLE
        std_error_handle = wintypes.DWORD(0xFFFFFFF4)

        before = k32.GetStdHandle(std_error_handle)
        with suppress_ep_registration_stderr():
            pass
        after = k32.GetStdHandle(std_error_handle)
        assert before == after, "STD_ERROR_HANDLE not restored"

    @pytest.mark.skipif(sys.platform == "win32", reason="Non-Windows only")
    def test_noop_on_non_windows(self):
        """On non-Windows, the context manager yields without fd manipulation."""
        with suppress_ep_registration_stderr():
            os.write(2, b"passthrough\n")
        assert _ort_startup_logs == []


class TestReplayOrtStartupLogs:
    """Tests for replay_ort_startup_logs."""

    def test_replays_and_clears_buffer(self, caplog):
        _ort_startup_logs.extend(["msg1", "msg2"])
        target = logging.getLogger("winml.modelkit.utils.native_stderr")
        with caplog.at_level(logging.DEBUG, logger="winml.modelkit.utils.native_stderr"):
            replay_ort_startup_logs(target)
        assert _ort_startup_logs == []
        assert any("msg1" in r.message for r in caplog.records)
        assert any("msg2" in r.message for r in caplog.records)

    def test_noop_when_empty(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="winml.modelkit.utils.native_stderr"):
            replay_ort_startup_logs()
        assert len(caplog.records) == 0

    def test_custom_target_logger(self, caplog):
        _ort_startup_logs.append("custom")
        custom = logging.getLogger("test.custom")
        with caplog.at_level(logging.DEBUG, logger="test.custom"):
            replay_ort_startup_logs(custom)
        assert any("custom" in r.message for r in caplog.records)
        assert _ort_startup_logs == []
