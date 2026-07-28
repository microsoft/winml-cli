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

    def test_disabled_leaves_native_stderr_visible(self, capfd):
        with suppress_native_stderr(enabled=False):
            os.write(2, b"visible when disabled\n")
        assert "visible when disabled" in capfd.readouterr().err

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 only")
    def test_win32_std_error_handle_restored(self):
        import ctypes.wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetStdHandle.argtypes = [ctypes.wintypes.DWORD]
        k32.GetStdHandle.restype = ctypes.wintypes.HANDLE
        std_error_handle = ctypes.wintypes.DWORD(0xFFFFFFF4)

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
