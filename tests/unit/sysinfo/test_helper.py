# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for sysinfo.helper — WMI/PnP PowerShell wrappers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


class TestCimInstancePropertyFilter:
    """CimInstance.get_by_class_name uses -Property to project WMI columns."""

    def test_property_whitelist_is_passed_to_powershell(self) -> None:
        """A property whitelist must appear in the rendered PS command.

        Without -Property, WMI hydrates every column on the class. For
        Win32_Processor (~50 columns) that's a ~6x slowdown (~1.3 s vs
        ~0.2 s warm), so the projection MUST land on the PowerShell side.
        """
        from winml.modelkit.sysinfo.helper import CimInstance

        with patch(
            "winml.modelkit.sysinfo.helper.subprocess.check_output",
            return_value=b"[]",
        ) as mock:
            CimInstance.get_by_class_name(
                "Win32_Processor",
                properties=["Name", "NumberOfCores", "Architecture"],
            )
        assert mock.call_count == 1
        ps_cmd = mock.call_args.args[0][-1]
        assert "-Property Name,NumberOfCores,Architecture" in ps_cmd

    def test_no_property_filter_falls_back_to_old_behavior(self) -> None:
        """Callers that don't whitelist still get every column (back-compat)."""
        from winml.modelkit.sysinfo.helper import CimInstance

        with patch(
            "winml.modelkit.sysinfo.helper.subprocess.check_output",
            return_value=b"[]",
        ) as mock:
            CimInstance.get_by_class_name("Win32_PhysicalMemory")
        ps_cmd = mock.call_args.args[0][-1]
        assert "-Property" not in ps_cmd
        assert "Get-CimInstance -ClassName Win32_PhysicalMemory" in ps_cmd


class TestPnpDevicePrefetchedProperties:
    """PnpDevice can skip the per-device property subprocess when caller
    supplies the property JSON list directly."""

    def test_skips_subprocess_when_properties_prefetched(self) -> None:
        """Construction must not spawn Get-PnpDeviceProperty when the
        caller already obtained the property list out-of-band."""
        from winml.modelkit.sysinfo.helper import PnpDevice

        with patch("winml.modelkit.sysinfo.helper.subprocess.check_output") as mock:
            dev = PnpDevice(
                {"PNPDeviceID": "ACPI\\QCOM0001"},
                prefetched_properties=[
                    {"KeyName": "DEVPKEY_Device_DriverVersion", "Data": "1.2.3"},
                ],
            )
        assert mock.call_count == 0
        assert dev.get_extra_property("DEVPKEY_Device_DriverVersion", str) == "1.2.3"

    def test_empty_prefetched_properties_is_ok(self) -> None:
        """A device that legitimately has no extra properties must work."""
        from winml.modelkit.sysinfo.helper import PnpDevice

        with patch("winml.modelkit.sysinfo.helper.subprocess.check_output") as mock:
            PnpDevice({"PNPDeviceID": "X"}, prefetched_properties=[])
        assert mock.call_count == 0


class TestPnpDeviceGetByClassNameWithProperties:
    """One-shot batched PnP query: Get-PnpDevice + Get-PnpDeviceProperty
    for every device returned, in a single PowerShell subprocess."""

    def _mock_check_output(self, payload: object) -> MagicMock:
        mock = MagicMock()
        mock.return_value = json.dumps(payload).encode("utf-8")
        return mock

    def test_invokes_powershell_exactly_once_for_n_devices(self) -> None:
        """Whole point of this helper: one PS startup regardless of device count."""
        from winml.modelkit.sysinfo.helper import PnpDevice

        payload = {
            "items": [
                {
                    "device": {"PNPDeviceID": "ACPI\\QCOM0001", "Name": "NPU0"},
                    "properties": [
                        {"KeyName": "DEVPKEY_Device_DriverVersion", "Data": "1.0"},
                    ],
                },
                {
                    "device": {"PNPDeviceID": "ACPI\\QCOM0002", "Name": "NPU1"},
                    "properties": [
                        {"KeyName": "DEVPKEY_Device_DriverVersion", "Data": "2.0"},
                    ],
                },
            ]
        }
        mock = self._mock_check_output(payload)
        with patch("winml.modelkit.sysinfo.helper.subprocess.check_output", mock):
            devs = PnpDevice.get_by_class_name_with_properties("ComputeAccelerator")
        assert mock.call_count == 1
        assert len(devs) == 2
        assert devs[0].get_extra_property("DEVPKEY_Device_DriverVersion", str) == "1.0"
        assert devs[1].get_extra_property("DEVPKEY_Device_DriverVersion", str) == "2.0"

    def test_normalizes_single_device_dict_to_list(self) -> None:
        """PowerShell collapses single-element arrays to a bare object; the
        helper must re-list it before iterating."""
        from winml.modelkit.sysinfo.helper import PnpDevice

        payload = {
            "items": {
                "device": {"PNPDeviceID": "ACPI\\X", "Name": "Only"},
                "properties": {"KeyName": "K", "Data": "v"},
            }
        }
        with patch(
            "winml.modelkit.sysinfo.helper.subprocess.check_output",
            self._mock_check_output(payload),
        ):
            devs = PnpDevice.get_by_class_name_with_properties("ComputeAccelerator")
        assert len(devs) == 1
        assert devs[0].get_extra_property("K", str) == "v"

    def test_returns_empty_on_subprocess_failure(self) -> None:
        """A PowerShell crash must surface as an empty list, not bubble up."""
        import subprocess

        from winml.modelkit.sysinfo.helper import PnpDevice

        mock = MagicMock(side_effect=subprocess.CalledProcessError(1, "powershell"))
        with patch("winml.modelkit.sysinfo.helper.subprocess.check_output", mock):
            devs = PnpDevice.get_by_class_name_with_properties("ComputeAccelerator")
        assert devs == []

    def test_returns_empty_when_no_devices(self) -> None:
        """Empty PnP class (no NPUs) returns []."""
        from winml.modelkit.sysinfo.helper import PnpDevice

        with patch(
            "winml.modelkit.sysinfo.helper.subprocess.check_output",
            self._mock_check_output({"items": None}),
        ):
            devs = PnpDevice.get_by_class_name_with_properties("ComputeAccelerator")
        assert devs == []
