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
    capture_native_stderr,
    suppress_native_stderr,
)


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

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 only")
    def test_win32_std_error_handle_restored(self):
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetStdHandle.argtypes = [wintypes.DWORD]
        k32.GetStdHandle.restype = wintypes.HANDLE
        std_error_handle = wintypes.DWORD(0xFFFFFFF4)

        before = k32.GetStdHandle(std_error_handle)
        with suppress_native_stderr():
            pass
        after = k32.GetStdHandle(std_error_handle)
        assert before == after, "STD_ERROR_HANDLE not restored"

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

    def test_skips_blank_lines(self, caplog):
        with (
            caplog.at_level(logging.INFO, logger="winml.modelkit.utils.native_stderr"),
            capture_native_stderr(logging.INFO),
        ):
            os.write(2, b"  \n\nkeep\n  \n")
        messages = [r.message for r in caplog.records]
        assert any("keep" in m for m in messages)
        assert not any(m == "  " for m in messages)
