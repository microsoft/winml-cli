# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Power reporting contracts using generated command output and native results."""

from __future__ import annotations

import ctypes
import logging
import subprocess
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

import winml.modelkit.commands._sys_power as power
import winml.modelkit.commands.sys as sysmod
from winml.modelkit.sysinfo import DXCoreAdapterInfo, format_pdh_luid


@pytest.fixture
def generated_power_report():
    limits = {
        f"{0:08x}:{index:02x}:{0:02x}.{0:x}": {
            "gpu_current_power_limit_w": float(index * 25),
            "module_current_power_limit_w": float(index * 50),
        }
        for index in range(1, 4)
    }
    lines = []
    for bus_id, values in reversed(limits.items()):
        lines.append(f"GPU {bus_id.upper()}")
        for section, key in (
            ("GPU Power Readings", "gpu_current_power_limit_w"),
            ("Module Power Readings", "module_current_power_limit_w"),
        ):
            lines.extend(
                (
                    f"    {section}",
                    f"        Average Power Draw : {values[key] / 10:.2f} W",
                    f"        Current Power Limit : {values[key]:.2f} W",
                    f"        Requested Power Limit : {values[key] + 1:.2f} W",
                    f"        Default Power Limit : {values[key] + 2:.2f} W",
                )
            )
    return "\n".join(lines), limits


@pytest.mark.parametrize("label", ["Power Scheme GUID", "电源方案 GUID"])
def test_active_scheme_localized_labels(monkeypatch, label):
    guid = str(uuid.uuid4())
    name = f"scheme [{guid[:8]}]"
    check_output = MagicMock(return_value=f"{label}: {guid}  ({name})\n")
    monkeypatch.setattr(power.sys, "platform", "win32")
    monkeypatch.setattr(power.shutil, "which", lambda _: "powercfg.exe")
    monkeypatch.setattr(power.subprocess, "check_output", check_output)

    assert power._get_active_power_scheme() == {"guid": guid, "name": name}
    assert check_output.call_args.args[0] == ["powercfg.exe", "/getactivescheme"]
    assert check_output.call_args.kwargs["timeout"] > 0


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError,
        lambda: subprocess.CalledProcessError(1, "powercfg"),
        lambda: subprocess.TimeoutExpired("powercfg", 10),
        lambda: None,
    ],
)
def test_active_scheme_failure_is_reported(monkeypatch, caplog, failure):
    monkeypatch.setattr(power.sys, "platform", "win32")
    monkeypatch.setattr(power.shutil, "which", lambda _: "powercfg.exe")
    monkeypatch.setattr(
        power.subprocess, "check_output", MagicMock(return_value="", side_effect=failure())
    )
    with caplog.at_level(logging.WARNING):
        assert power._get_active_power_scheme() is None
    assert "Failed to read the active power scheme" in caplog.text


def test_non_windows_does_not_probe_powercfg(monkeypatch):
    monkeypatch.setattr(power.sys, "platform", "linux")
    probe = MagicMock()
    monkeypatch.setattr(power.subprocess, "check_output", probe)
    assert power._get_active_power_scheme() is None
    probe.assert_not_called()


def test_power_query_uses_requested_command(monkeypatch, generated_power_report):
    output, expected = generated_power_report
    monkeypatch.setattr(power.shutil, "which", lambda _: "nvidia-smi.exe")
    check_output = MagicMock(return_value=output)
    monkeypatch.setattr(power.subprocess, "check_output", check_output)
    assert power._get_nvidia_power_limits() == expected
    assert check_output.call_args.args[0] == ["nvidia-smi.exe", "-q", "-d", "POWER"]
    assert check_output.call_args.kwargs["timeout"] > 0


@pytest.mark.parametrize("replacement", ["N/A", "nan W", "inf W", "-1 W", "invalid"])
def test_unsupported_and_invalid_limits_are_not_reported(
    generated_power_report, replacement, caplog
):
    output, expected = generated_power_report
    for values in expected.values():
        output = output.replace(
            f"Current Power Limit : {values['module_current_power_limit_w']:.2f} W",
            f"Current Power Limit : {replacement}",
        )
    actual = power._parse_nvidia_power_limits(output)
    assert all("module_current_power_limit_w" not in values for values in actual.values())
    if replacement != "N/A":
        assert "Invalid NVIDIA current power limit" in caplog.text


def test_missing_nvidia_tool_is_optional(monkeypatch):
    monkeypatch.setattr(power.shutil, "which", lambda _: None)
    probe = MagicMock()
    monkeypatch.setattr(power.subprocess, "check_output", probe)
    assert power._get_nvidia_power_limits() == {}
    probe.assert_not_called()


def test_unavailable_module_limit_preserves_gpu_limit(generated_power_report):
    output, expected = generated_power_report
    lines = output.splitlines()
    in_module = False
    for index, line in enumerate(lines):
        if line.startswith("    ") and not line.startswith("        "):
            in_module = line.strip() == "Module Power Readings"
        elif in_module and "Current Power Limit" in line:
            lines[index] = f"{line.partition(':')[0]}: N/A"
    actual = power._parse_nvidia_power_limits("\n".join(lines))
    assert actual == {
        bus_id: {"gpu_current_power_limit_w": values["gpu_current_power_limit_w"]}
        for bus_id, values in expected.items()
    }


def test_zero_power_limit_is_retained_and_other_sections_ignored(generated_power_report):
    output, expected = generated_power_report
    for values in expected.values():
        for watts in values.values():
            output = output.replace(
                f"Current Power Limit : {watts:.2f} W", "Current Power Limit : 0 W"
            )
    output += "\n    GPU Memory Power Readings\n        Current Power Limit : 999 W"
    actual = power._parse_nvidia_power_limits(output)
    assert actual == {bus_id: dict.fromkeys(values, 0.0) for bus_id, values in expected.items()}


def test_missing_powercfg_is_reported(monkeypatch, caplog):
    monkeypatch.setattr(power.sys, "platform", "win32")
    monkeypatch.setattr(power.shutil, "which", lambda _: None)
    assert power._get_active_power_scheme() is None
    assert "powercfg is unavailable" in caplog.text


def test_invalid_gpu_report_is_reported(caplog):
    assert power._parse_nvidia_power_limits("") == {}
    assert "no GPU PCI identities" in caplog.text


@pytest.mark.parametrize(
    "error", [OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired]
)
def test_nvidia_query_failures_preserve_report(monkeypatch, caplog, error):
    monkeypatch.setattr(power.shutil, "which", lambda _: "nvidia-smi.exe")
    exception = error() if error is OSError else error(1, "nvidia-smi")
    monkeypatch.setattr(power.subprocess, "check_output", MagicMock(side_effect=exception))
    assert power._get_nvidia_power_limits() == {}
    assert "Failed to read NVIDIA power limits" in caplog.text


@pytest.mark.parametrize("query_fails", [False, True])
def test_native_luid_to_pci_bridge_closes_adapter(monkeypatch, query_fails, caplog):
    value = int(uuid.uuid4()) & ((1 << 64) - 1)
    luid = format_pdh_luid(str(value))
    address = power._AdapterAddress(*(int(part, 16) for part in luid.split("_")[0][-3:]))
    handle = value & 0xFFFFFFFF

    def open_adapter(pointer):
        adapter = ctypes.cast(pointer, ctypes.POINTER(power._OpenAdapterFromLUID)).contents
        assert adapter.luid.low == value & 0xFFFFFFFF
        assert adapter.luid.high & 0xFFFFFFFF == value >> 32
        adapter.handle = handle
        return 0

    def query_adapter(pointer):
        query = ctypes.cast(pointer, ctypes.POINTER(power._QueryAdapterInfo)).contents
        assert query.handle == handle
        assert query.query_type == 6
        assert query.size == ctypes.sizeof(address)
        ctypes.memmove(query.data, ctypes.byref(address), ctypes.sizeof(address))
        return -1 if query_fails else 0

    dll = SimpleNamespace(
        D3DKMTOpenAdapterFromLuid=MagicMock(side_effect=open_adapter),
        D3DKMTQueryAdapterInfo=MagicMock(side_effect=query_adapter),
        D3DKMTCloseAdapter=MagicMock(return_value=0),
    )
    monkeypatch.setattr(power.sys, "platform", "win32")
    monkeypatch.setattr(power.ctypes, "WinDLL", lambda _: dll, raising=False)
    result = power._get_gpu_pci_bus_id(luid)
    if query_fails:
        assert result is None
        assert "D3DKMTQueryAdapterInfo" in caplog.text
    else:
        assert result == f"00000000:{address.bus:02x}:{address.device:02x}.{address.function:x}"
    dll.D3DKMTCloseAdapter.assert_called_once()


def test_identical_devices_join_by_identity_and_render_both_limits(
    monkeypatch, generated_power_report
):
    output, limits = generated_power_report
    adapters = [
        DXCoreAdapterInfo(
            device_type="GPU",
            name="Identical GPU",
            luid=format_pdh_luid(str(index)),
            vendor_id=0x10DE,
            device_id=0,
        )
        for index in range(1, len(limits) + 1)
    ]
    identities = dict(zip((adapter.luid for adapter in adapters), limits, strict=True))
    monkeypatch.setattr("winml.modelkit.sysinfo.enumerate_compute_adapters", lambda: adapters[::-1])
    for cls in ("GPU", "NPU", "CPU"):
        monkeypatch.setattr(f"winml.modelkit.sysinfo.{cls}.get_all", list)
    monkeypatch.setattr(
        power, "_get_nvidia_power_limits", lambda: power._parse_nvidia_power_limits(output)
    )
    monkeypatch.setattr(power, "_get_gpu_pci_bus_id", identities.__getitem__)

    devices = sysmod._gather_device_info()
    assert len(devices) == len(adapters)
    for device in devices:
        expected = limits[identities[device["details"]["luid"]]]
        assert expected.items() <= device["details"].items()

    console = Console(record=True, force_terminal=False, width=200)
    monkeypatch.setattr(sysmod, "console", console)
    sysmod._output_device_text(devices)
    rendered = console.export_text()
    for values in limits.values():
        assert f"GPU current power limit: {values['gpu_current_power_limit_w']:g} W" in rendered
        assert (
            f"Module current power limit: {values['module_current_power_limit_w']:g} W" in rendered
        )


def test_missing_device_identity_never_assigns_another_gpu_limit(
    monkeypatch, generated_power_report, caplog
):
    _, limits = generated_power_report
    devices = [{"name": f"GPU {index}", "details": {"luid": None}} for index in range(len(limits))]
    monkeypatch.setattr(power, "_get_nvidia_power_limits", lambda: limits)
    power._enrich_nvidia_power_limits(devices)
    assert all(device["details"] == {"luid": None} for device in devices)
    assert "no device LUID" in caplog.text


def test_unmatched_pci_address_never_assigns_another_gpu_limit(
    monkeypatch, generated_power_report, caplog
):
    _, limits = generated_power_report
    device = {"name": "GPU", "details": {"luid": format_pdh_luid(str(len(limits) + 1))}}
    monkeypatch.setattr(power, "_get_nvidia_power_limits", lambda: limits)
    monkeypatch.setattr(power, "_get_gpu_pci_bus_id", lambda _: f"{len(limits) + 1:08x}:00:00.0")
    power._enrich_nvidia_power_limits([device])
    assert list(device["details"]) == ["luid"]
    assert "No NVIDIA power readings match" in caplog.text


def test_environment_includes_scheme_and_escapes_name(monkeypatch):
    scheme = {"guid": str(uuid.uuid4()), "name": "[bold]scheme[/bold]"}
    monkeypatch.setattr(sysmod, "_get_active_power_scheme", lambda: scheme)
    info = sysmod._gather_system_info()
    info["platform"]["system"] = "Windows"
    assert info["power_scheme"] == scheme
    console = Console(record=True, force_terminal=False, width=200)
    monkeypatch.setattr(sysmod, "console", console)
    sysmod._output_text(info)
    rendered = console.export_text()
    assert "Active power scheme" in rendered
    assert scheme["name"] in rendered
    assert scheme["guid"] in rendered
