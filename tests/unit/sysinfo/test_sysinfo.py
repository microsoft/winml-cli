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


class TestGetPlatformInfo:
    """Test _get_platform_info function."""

    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_11_detection(self, mock_platform: MagicMock, mock_os_class: MagicMock) -> None:
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

    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_10_detection(self, mock_platform: MagicMock, mock_os_class: MagicMock) -> None:
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

    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_7_preserved(self, mock_platform: MagicMock, mock_os_class: MagicMock) -> None:
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

    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_81_preserved(self, mock_platform: MagicMock, mock_os_class: MagicMock) -> None:
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

    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_detection_fallback_on_exception(
        self, mock_platform: MagicMock, mock_os_class: MagicMock
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
        import time

        from winml.modelkit.commands import sys as sys_cmd

        def slow_empty() -> list:
            time.sleep(0.15)
            return []

        with (
            patch.object(sys_cmd, "_gather_device_info", wraps=sys_cmd._gather_device_info),
            patch("winml.modelkit.sysinfo.hardware.CPU.get_all", side_effect=slow_empty),
            patch("winml.modelkit.sysinfo.hardware.GPU.get_all", side_effect=slow_empty),
            patch("winml.modelkit.sysinfo.hardware.NPU.get_all", side_effect=slow_empty),
        ):
            t0 = time.perf_counter()
            sys_cmd._gather_device_info()
            elapsed = time.perf_counter() - t0

        # Sequential would be 3 x 0.15 = 0.45s; parallel should be ~0.15s.
        # Allow generous headroom for ThreadPoolExecutor scheduling.
        assert elapsed < 0.35, f"Hardware queries appear to run sequentially ({elapsed:.2f}s)"
