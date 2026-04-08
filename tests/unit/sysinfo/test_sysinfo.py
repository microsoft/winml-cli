# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for sysinfo command module.

Tests the _get_platform_info and _is_windows_11 functions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestIsWindows11:
    """Test _is_windows_11 helper function."""

    @patch("winml.modelkit.commands.sys.platform")
    def test_build_26200_is_windows_11(self, mock_platform: MagicMock) -> None:
        from winml.modelkit.commands.sys import _is_windows_11

        mock_platform.version.return_value = "10.0.26200"
        assert _is_windows_11() is True

    @patch("winml.modelkit.commands.sys.platform")
    def test_build_22000_is_windows_11(self, mock_platform: MagicMock) -> None:
        from winml.modelkit.commands.sys import _is_windows_11

        mock_platform.version.return_value = "10.0.22000"
        assert _is_windows_11() is True

    @patch("winml.modelkit.commands.sys.platform")
    def test_build_21999_is_not_windows_11(self, mock_platform: MagicMock) -> None:
        from winml.modelkit.commands.sys import _is_windows_11

        mock_platform.version.return_value = "10.0.21999"
        assert _is_windows_11() is False

    @patch("winml.modelkit.commands.sys.platform")
    def test_build_19045_is_not_windows_11(self, mock_platform: MagicMock) -> None:
        from winml.modelkit.commands.sys import _is_windows_11

        mock_platform.version.return_value = "10.0.19045"
        assert _is_windows_11() is False

    @patch("winml.modelkit.commands.sys.platform")
    def test_malformed_version_returns_false(self, mock_platform: MagicMock) -> None:
        from winml.modelkit.commands.sys import _is_windows_11

        mock_platform.version.return_value = "10.0"
        assert _is_windows_11() is False

    @patch("winml.modelkit.commands.sys.platform")
    def test_non_numeric_build_returns_false(self, mock_platform: MagicMock) -> None:
        from winml.modelkit.commands.sys import _is_windows_11

        mock_platform.version.return_value = "10.0.abc"
        assert _is_windows_11() is False


class TestGetPlatformInfo:
    """Test _get_platform_info function."""

    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_11_detection(self, mock_platform: MagicMock) -> None:
        """Test Windows 11 is correctly detected via build number."""
        from winml.modelkit.commands.sys import _get_platform_info

        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "10"  # Platform reports wrong version
        mock_platform.version.return_value = "10.0.26200"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        result = _get_platform_info()

        assert result["system"] == "Windows"
        assert result["release"] == "11"  # Should be corrected to 11
        assert result["machine"] == "AMD64"

    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_10_detection(self, mock_platform: MagicMock) -> None:
        """Test Windows 10 is correctly detected."""
        from winml.modelkit.commands.sys import _get_platform_info

        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "10"
        mock_platform.version.return_value = "10.0.19045"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        result = _get_platform_info()

        assert result["system"] == "Windows"
        assert result["release"] == "10"
        assert result["machine"] == "AMD64"

    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_7_preserved(self, mock_platform: MagicMock) -> None:
        """Test Windows 7 version is preserved."""
        from winml.modelkit.commands.sys import _get_platform_info

        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "7"
        mock_platform.version.return_value = "6.1.7601"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        result = _get_platform_info()

        assert result["system"] == "Windows"
        assert result["release"] == "7"
        assert result["machine"] == "AMD64"

    @patch("winml.modelkit.commands.sys.platform")
    def test_windows_81_preserved(self, mock_platform: MagicMock) -> None:
        """Test Windows 8.1 version is preserved."""
        from winml.modelkit.commands.sys import _get_platform_info

        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "8.1"
        mock_platform.version.return_value = "6.3.9600"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Intel64 Family 6"

        result = _get_platform_info()

        assert result["system"] == "Windows"
        assert result["release"] == "8.1"
        assert result["machine"] == "AMD64"

    @patch("winml.modelkit.commands.sys.platform")
    def test_non_windows_platform(self, mock_platform: MagicMock) -> None:
        """Test non-Windows platforms pass through unchanged."""
        from winml.modelkit.commands.sys import _get_platform_info

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

        mock_platform.system.return_value = "Linux"
        mock_platform.release.return_value = "5.15.0"
        mock_platform.machine.return_value = "x86_64"
        mock_platform.processor.return_value = ""

        result = _get_platform_info()

        assert result["processor"] == "Unknown"
