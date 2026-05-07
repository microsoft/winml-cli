# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for sysinfo command module.

Tests the _get_platform_info function with Windows version detection.
"""

from __future__ import annotations

import os
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

    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_processor_unknown_fallback(
        self, mock_platform: MagicMock, mock_os_class: MagicMock
    ) -> None:
        """Test processor defaults to 'Unknown' when empty."""
        from winml.modelkit.commands.sys import _get_platform_info

        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "10"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = ""  # Empty string

        mock_os_instance = MagicMock()
        mock_os_instance.is_windows_11.return_value = False
        mock_os_class.get.return_value = mock_os_instance

        with patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "AMD64"}, clear=True):
            result = _get_platform_info()

        assert result["processor"] == "Unknown"


class TestWindowsHostArchitecture:
    """Test that Windows machine architecture reports the host, not the Python process arch.

    Regression coverage for https://github.com/microsoft/WinML-ModelKit/issues/58:
    `platform.machine()` returns the running Python process architecture, which is
    incorrect on ARM64 Windows when an x64 Python runs under WOW64 emulation.
    """

    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_native_arm64_python_reports_arm64(
        self, mock_platform: MagicMock, mock_os_class: MagicMock
    ) -> None:
        """Native ARM64 Python on ARM64 Windows must report ARM64."""
        from winml.modelkit.commands.sys import _get_platform_info

        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "10"
        mock_platform.machine.return_value = "ARM64"
        mock_platform.processor.return_value = "ARMv8 (64-bit)"

        mock_os_instance = MagicMock()
        mock_os_instance.is_windows_11.return_value = True
        mock_os_class.get.return_value = mock_os_instance

        with patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "ARM64"}, clear=True):
            result = _get_platform_info()

        assert result["machine"] == "ARM64"

    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_x64_emulated_python_on_arm64_reports_arm64(
        self, mock_platform: MagicMock, mock_os_class: MagicMock
    ) -> None:
        """x64-emulated Python on ARM64 Windows must report ARM64, not AMD64.

        Under WOW64 emulation, PROCESSOR_ARCHITECTURE reflects the emulated
        process arch (AMD64) and PROCESSOR_ARCHITEW6432 holds the true host
        arch (ARM64). The host arch is what users want to see.
        """
        from winml.modelkit.commands.sys import _get_platform_info

        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "11"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        mock_os_instance = MagicMock()
        mock_os_instance.is_windows_11.return_value = True
        mock_os_class.get.return_value = mock_os_instance

        with patch.dict(
            os.environ,
            {"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_ARCHITEW6432": "ARM64"},
            clear=True,
        ):
            result = _get_platform_info()

        assert result["machine"] == "ARM64"

    @patch("winml.modelkit.commands.sys.OS")
    @patch("winml.modelkit.commands.sys.platform")
    def test_native_amd64_python_reports_amd64(
        self, mock_platform: MagicMock, mock_os_class: MagicMock
    ) -> None:
        """Native AMD64 Python on AMD64 Windows must still report AMD64."""
        from winml.modelkit.commands.sys import _get_platform_info

        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "11"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        mock_os_instance = MagicMock()
        mock_os_instance.is_windows_11.return_value = True
        mock_os_class.get.return_value = mock_os_instance

        with patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "AMD64"}, clear=True):
            result = _get_platform_info()

        assert result["machine"] == "AMD64"
