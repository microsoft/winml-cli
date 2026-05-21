# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for sysinfo command module.

Tests the _get_platform_info function with Windows version detection.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestGetWindowsNativeMachine:
    """Test _get_windows_native_machine and the IMAGE_FILE_MACHINE mapping.

    These tests lock in the IMAGE_FILE_MACHINE_* → display-name mapping so
    accidentally renaming "ARM64" → "Arm64" (or similar) ships with a test
    failure rather than silently changing user-visible output. The
    higher-level _get_platform_info tests mock the helper's return value
    and would not catch a mapping rename.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (0x8664, "AMD64"),
            (0xAA64, "ARM64"),
            (0x14C, "x86"),
            # ARMNT (0xC4) is not a Windows 11 host arch — intentionally unmapped
            (0xC4, None),
            # IMAGE_FILE_MACHINE_UNKNOWN — IsWow64Process2 returns this for the
            # process slot when the process is native on the host
            (0x0, None),
        ],
    )
    def test_native_machine_mapping(
        self, raw: int, expected: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from winml.modelkit.commands import sys as sys_mod

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sys_mod, "_query_native_machine_via_win32", lambda: raw)

        assert sys_mod._get_windows_native_machine() == expected

    def test_returns_none_when_win32_query_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from winml.modelkit.commands import sys as sys_mod

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sys_mod, "_query_native_machine_via_win32", lambda: None)

        assert sys_mod._get_windows_native_machine() is None

    def test_returns_none_on_non_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from winml.modelkit.commands import sys as sys_mod

        monkeypatch.setattr(sys, "platform", "linux")

        assert sys_mod._get_windows_native_machine() is None


class TestGetPlatformInfo:
    """Test _get_platform_info function."""

    @patch("winml.modelkit.commands.sys._get_windows_native_machine", return_value=None)
    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_11_detection(
        self, mock_platform: MagicMock, mock_os_class: MagicMock, _mock_native: MagicMock
    ) -> None:
        """Test Windows 11 is correctly detected."""
        from winml.modelkit.commands.sys import _get_platform_info

        # Setup mocks
        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "10"  # Platform reports wrong version
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        mock_os_instance = MagicMock()
        mock_os_instance.is_windows_11.return_value = True
        mock_os_class.get.return_value = mock_os_instance

        result = _get_platform_info()

        assert result["system"] == "Windows"
        assert result["release"] == "11"  # Should be corrected to 11
        assert result["machine"] == "AMD64"
        mock_os_class.get.assert_called_once()

    @patch("winml.modelkit.commands.sys._get_windows_native_machine", return_value=None)
    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_10_detection(
        self, mock_platform: MagicMock, mock_os_class: MagicMock, _mock_native: MagicMock
    ) -> None:
        """Test Windows 10 is correctly detected."""
        from winml.modelkit.commands.sys import _get_platform_info

        # Setup mocks
        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "10"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        mock_os_instance = MagicMock()
        mock_os_instance.is_windows_11.return_value = False
        mock_os_class.get.return_value = mock_os_instance

        result = _get_platform_info()

        assert result["system"] == "Windows"
        assert result["release"] == "10"
        assert result["machine"] == "AMD64"

    @patch("winml.modelkit.commands.sys._get_windows_native_machine", return_value=None)
    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_7_preserved(
        self, mock_platform: MagicMock, mock_os_class: MagicMock, _mock_native: MagicMock
    ) -> None:
        """Test Windows 7 version is preserved (not changed to 10)."""
        from winml.modelkit.commands.sys import _get_platform_info

        # Setup mocks
        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "7"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        mock_os_instance = MagicMock()
        mock_os_instance.is_windows_11.return_value = False
        mock_os_class.get.return_value = mock_os_instance

        result = _get_platform_info()

        assert result["system"] == "Windows"
        assert result["release"] == "7"  # Should keep original value
        assert result["machine"] == "AMD64"

    @patch("winml.modelkit.commands.sys._get_windows_native_machine", return_value=None)
    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_81_preserved(
        self, mock_platform: MagicMock, mock_os_class: MagicMock, _mock_native: MagicMock
    ) -> None:
        """Test Windows 8.1 version is preserved (not changed to 10)."""
        from winml.modelkit.commands.sys import _get_platform_info

        # Setup mocks
        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "8.1"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        mock_os_instance = MagicMock()
        mock_os_instance.is_windows_11.return_value = False
        mock_os_class.get.return_value = mock_os_instance

        result = _get_platform_info()

        assert result["system"] == "Windows"
        assert result["release"] == "8.1"  # Should keep original value
        assert result["machine"] == "AMD64"

    @patch("winml.modelkit.commands.sys._get_windows_native_machine", return_value=None)
    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_detection_fallback_on_exception(
        self, mock_platform: MagicMock, mock_os_class: MagicMock, _mock_native: MagicMock
    ) -> None:
        """Test fallback to platform.release() when OS detection fails."""
        from winml.modelkit.commands.sys import _get_platform_info

        # Setup mocks
        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "10"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        # OS.get() raises exception
        mock_os_class.get.side_effect = RuntimeError("WMI error")

        result = _get_platform_info()

        # Should use fallback value from platform.release()
        assert result["system"] == "Windows"
        assert result["release"] == "10"
        assert result["machine"] == "AMD64"

    @patch("winml.modelkit.commands.sys._get_windows_native_machine")
    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_arm64_host_with_x64_python(
        self,
        mock_platform: MagicMock,
        mock_os_class: MagicMock,
        mock_native: MagicMock,
    ) -> None:
        """x64 Python on ARM64 host: IsWow64Process2 reveals the real host arch.

        platform.machine() returns the process arch ("AMD64") under emulation;
        winml sys should display the host arch ("ARM64") instead.
        """
        from winml.modelkit.commands.sys import _get_platform_info

        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "10"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Snapdragon"

        mock_os_instance = MagicMock()
        mock_os_instance.is_windows_11.return_value = True
        mock_os_class.get.return_value = mock_os_instance

        mock_native.return_value = "ARM64"

        result = _get_platform_info()

        assert result["machine"] == "ARM64"

    @patch("winml.modelkit.commands.sys._get_windows_native_machine")
    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_native_lookup_failure_falls_back(
        self,
        mock_platform: MagicMock,
        mock_os_class: MagicMock,
        mock_native: MagicMock,
    ) -> None:
        """When IsWow64Process2 yields None, fall back to platform.machine()."""
        from winml.modelkit.commands.sys import _get_platform_info

        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "10"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        mock_os_instance = MagicMock()
        mock_os_instance.is_windows_11.return_value = True
        mock_os_class.get.return_value = mock_os_instance

        mock_native.return_value = None

        result = _get_platform_info()

        assert result["machine"] == "AMD64"

    @patch("winml.modelkit.commands.sys.platform")
    def test_non_windows_platform(self, mock_platform: MagicMock) -> None:
        """Test non-Windows platforms pass through unchanged."""
        from winml.modelkit.commands.sys import _get_platform_info

        # Setup mocks for Linux
        mock_platform.system.return_value = "Linux"
        mock_platform.release.return_value = "5.15.0"
        mock_platform.machine.return_value = "x86_64"
        mock_platform.processor.return_value = "x86_64"

        result = _get_platform_info()

        assert result["system"] == "Linux"
        assert result["release"] == "5.15.0"
        assert result["machine"] == "x86_64"

    @patch("winml.modelkit.commands.sys.platform")
    def test_macos_platform(self, mock_platform: MagicMock) -> None:
        """Test macOS platforms pass through unchanged."""
        from winml.modelkit.commands.sys import _get_platform_info

        # Setup mocks for macOS
        mock_platform.system.return_value = "Darwin"
        mock_platform.release.return_value = "21.6.0"
        mock_platform.machine.return_value = "arm64"
        mock_platform.processor.return_value = "arm"

        result = _get_platform_info()

        assert result["system"] == "Darwin"
        assert result["release"] == "21.6.0"
        assert result["machine"] == "arm64"

    @patch("winml.modelkit.commands.sys.platform")
    def test_processor_unknown_fallback(self, mock_platform: MagicMock) -> None:
        """Test processor defaults to 'Unknown' when empty."""
        from winml.modelkit.commands.sys import _get_platform_info

        # Setup mocks
        mock_platform.system.return_value = "Linux"
        mock_platform.release.return_value = "5.15.0"
        mock_platform.machine.return_value = "x86_64"
        mock_platform.processor.return_value = ""  # Empty string

        result = _get_platform_info()

        assert result["processor"] == "Unknown"


class TestGetTorchInfo:
    """Test _get_torch_info function."""

    def test_non_verbose_uses_metadata_not_torch_import(self) -> None:
        """Non-verbose torch info must derive version from importlib.metadata,
        not from ``import torch``.

        Importing torch costs ~1.5 s warm and used to dominate ``winml sys``
        latency (issue #558). Setting ``sys.modules["torch"] = None`` inside
        a patch.dict block makes ``import torch`` raise ImportError without
        disturbing torch's actual loaded state — so if the function uses
        the metadata path, availability + version are still reported.
        """
        from winml.modelkit.commands.sys import _get_torch_info

        with patch.dict(sys.modules, {"torch": None}):
            info = _get_torch_info(verbose=False)

        assert info["available"] is True
        assert info["version"]

    def test_non_verbose_omits_cuda_keys(self) -> None:
        """Non-verbose torch info must not query CUDA (which requires torch import)."""
        from winml.modelkit.commands.sys import _get_torch_info

        info = _get_torch_info(verbose=False)
        assert "cuda_available" not in info
        assert "cuda_version" not in info
        assert "gpu_devices" not in info

    @patch("winml.modelkit.commands.sys.version")
    def test_returns_unavailable_when_torch_missing(self, mock_version: MagicMock) -> None:
        """When torch is not installed, info reports unavailable."""
        from importlib.metadata import PackageNotFoundError

        from winml.modelkit.commands.sys import _get_torch_info

        mock_version.side_effect = PackageNotFoundError("torch")
        info = _get_torch_info(verbose=False)
        assert info["available"] is False


class TestGatherDeviceInfo:
    """Test the parallel hardware detection path in _gather_device_info."""

    def test_runs_hw_queries_in_parallel(self) -> None:
        """Wall time of _gather_device_info should approach the slowest
        single query, not the sum — proving the three subprocesses run
        concurrently rather than sequentially.
        """
        import threading
        import time

        from winml.modelkit.commands import sys as sys_cmd

        call_starts: list[float] = []
        call_lock = threading.Lock()

        def slow_empty() -> list:
            with call_lock:
                call_starts.append(time.perf_counter())
            time.sleep(0.15)
            return []

        with (
            patch.object(sys_cmd, "_gather_device_info", wraps=sys_cmd._gather_device_info),
            patch("winml.modelkit.sysinfo.hardware.CPU.get_all", side_effect=slow_empty),
            patch("winml.modelkit.sysinfo.hardware.GPU.get_all", side_effect=slow_empty),
            patch("winml.modelkit.sysinfo.hardware.NPU.get_all", side_effect=slow_empty),
        ):
            sys_cmd._gather_device_info()

        assert len(call_starts) == 3

        # If the queries are submitted sequentially, each call starts about
        # 0.15s after the previous one. Parallel submission should make the
        # start times overlap closely even on slower CI runners.
        start_spread = max(call_starts) - min(call_starts)
        assert start_spread < 0.05, (
            f"Hardware queries appear to start sequentially ({start_spread:.2f}s)"
        )
