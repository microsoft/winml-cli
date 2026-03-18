# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for software.py module.

Tests OS class functionality, particularly Windows version detection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.sysinfo.software import OS


class TestOSClass:
    """Test OS class functionality."""

    def test_is_windows_11_with_build_22000(self) -> None:
        """Test Windows 11 detection with build 22000."""
        mock_instance = MagicMock()
        mock_instance.try_get_property.side_effect = lambda key, type_, default: {
            "Caption": "Microsoft Windows 11 Pro",
            "Version": "10.0.22000",
            "OSArchitecture": "64-bit",
            "OperatingSystemSKU": 48,
            "BuildNumber": "22000",
        }.get(key, default)

        os_info = OS(mock_instance)
        assert os_info.is_windows_11() is True
        assert os_info.build_number == "22000"

    def test_is_windows_11_with_build_26200(self) -> None:
        """Test Windows 11 detection with build 26200."""
        mock_instance = MagicMock()
        mock_instance.try_get_property.side_effect = lambda key, type_, default: {
            "Caption": "Microsoft Windows 11 Pro",
            "Version": "10.0.26200",
            "OSArchitecture": "64-bit",
            "OperatingSystemSKU": 48,
            "BuildNumber": "26200",
        }.get(key, default)

        os_info = OS(mock_instance)
        assert os_info.is_windows_11() is True
        assert os_info.build_number == "26200"

    def test_is_windows_10_with_build_19045(self) -> None:
        """Test Windows 10 detection with build 19045."""
        mock_instance = MagicMock()
        mock_instance.try_get_property.side_effect = lambda key, type_, default: {
            "Caption": "Microsoft Windows 10 Pro",
            "Version": "10.0.19045",
            "OSArchitecture": "64-bit",
            "OperatingSystemSKU": 48,
            "BuildNumber": "19045",
        }.get(key, default)

        os_info = OS(mock_instance)
        assert os_info.is_windows_11() is False
        assert os_info.build_number == "19045"

    def test_is_windows_11_boundary_build_21999(self) -> None:
        """Test Windows 10 detection at boundary (build 21999)."""
        mock_instance = MagicMock()
        mock_instance.try_get_property.side_effect = lambda key, type_, default: {
            "Caption": "Microsoft Windows 10 Pro",
            "Version": "10.0.21999",
            "OSArchitecture": "64-bit",
            "OperatingSystemSKU": 48,
            "BuildNumber": "21999",
        }.get(key, default)

        os_info = OS(mock_instance)
        assert os_info.is_windows_11() is False
        assert os_info.build_number == "21999"

    def test_is_windows_11_with_invalid_build_number(self) -> None:
        """Test Windows version detection with invalid build number."""
        mock_instance = MagicMock()
        mock_instance.try_get_property.side_effect = lambda key, type_, default: {
            "Caption": "Microsoft Windows 10 Pro",
            "Version": "10.0.unknown",
            "OSArchitecture": "64-bit",
            "OperatingSystemSKU": 48,
            "BuildNumber": "invalid",
        }.get(key, default)

        os_info = OS(mock_instance)
        assert os_info.is_windows_11() is False
        assert os_info.build_number == "invalid"

    def test_is_windows_11_with_empty_build_number(self) -> None:
        """Test Windows version detection with empty build number."""
        mock_instance = MagicMock()
        mock_instance.try_get_property.side_effect = lambda key, type_, default: {
            "Caption": "Microsoft Windows 10 Pro",
            "Version": "10.0.0",
            "OSArchitecture": "64-bit",
            "OperatingSystemSKU": 48,
            "BuildNumber": "",
        }.get(key, default)

        os_info = OS(mock_instance)
        assert os_info.is_windows_11() is False
        assert os_info.build_number == ""

    def test_os_properties(self) -> None:
        """Test all OS properties are correctly retrieved."""
        mock_instance = MagicMock()
        mock_instance.try_get_property.side_effect = lambda key, type_, default: {
            "Caption": "Microsoft Windows 11 Pro",
            "Version": "10.0.22000",
            "OSArchitecture": "64-bit",
            "OperatingSystemSKU": 48,
            "BuildNumber": "22000",
        }.get(key, default)

        os_info = OS(mock_instance)
        assert os_info.caption == "Microsoft Windows 11 Pro"
        assert os_info.version == "10.0.22000"
        assert os_info.architecture == "64-bit"
        assert os_info.sku == 48
        assert os_info.build_number == "22000"

    def test_os_to_dict(self) -> None:
        """Test OS.to_dict() includes isWindows11 field."""
        mock_instance = MagicMock()
        mock_instance.try_get_property.side_effect = lambda key, type_, default: {
            "Caption": "Microsoft Windows 11 Pro",
            "Version": "10.0.22000",
            "OSArchitecture": "64-bit",
            "OperatingSystemSKU": 48,
            "BuildNumber": "22000",
        }.get(key, default)

        os_info = OS(mock_instance)
        result = os_info.to_dict()

        assert result["caption"] == "Microsoft Windows 11 Pro"
        assert result["version"] == "10.0.22000"
        assert result["architecture"] == "64-bit"
        assert result["sku"] == 48
        assert result["buildNumber"] == "22000"
        assert result["isWindows11"] is True

    @patch("winml.modelkit.sysinfo.helper.CimInstance.get_by_class_name")
    def test_os_get_method(self, mock_get_cim: MagicMock) -> None:
        """Test OS.get() static method."""
        mock_instance = MagicMock()
        mock_instance.try_get_property.side_effect = lambda key, type_, default: {
            "Caption": "Microsoft Windows 11 Pro",
            "Version": "10.0.26200",
            "OSArchitecture": "64-bit",
            "OperatingSystemSKU": 48,
            "BuildNumber": "26200",
        }.get(key, default)

        mock_get_cim.return_value = [mock_instance]

        os_info = OS.get()
        assert os_info.is_windows_11() is True
        assert os_info.build_number == "26200"
        mock_get_cim.assert_called_once_with("Win32_OperatingSystem")

    @patch("winml.modelkit.sysinfo.helper.CimInstance.get_by_class_name")
    def test_os_get_no_instances(self, mock_get_cim: MagicMock) -> None:
        """Test OS.get() raises error when no instances found."""
        mock_get_cim.return_value = []

        with pytest.raises(RuntimeError, match="No Win32_OperatingSystem instance found"):
            OS.get()
