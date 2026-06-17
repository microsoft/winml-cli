# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the device-section renderer in ``commands/sys.py``.

Covers the ``device_facts()`` enrichment path: when a registered
:class:`WinMLEP` exposes a matching :class:`WinMLDevice` for a hardware
entry that sysinfo also reported, the renderer should fold
Architecture (and Driver, when sysinfo lacks one) into the per-device
``details`` dict so the *Available Devices* section displays
device-intrinsic facts per
``docs/design/session/4_winml_device.md`` §4.1.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from winml.modelkit.commands.sys import _gather_device_info


def _make_fake_winml_device(
    *, device_type: str, hardware_name: str, architecture: str | None = None,
    driver: str | None = None,
) -> MagicMock:
    """Build a WinMLDevice stand-in with controllable device_facts output."""
    d = MagicMock()
    d.device_type = device_type
    d.hardware_name = hardware_name
    facts: list[str] = []
    if architecture is not None:
        facts.append(f"Architecture: {architecture}")
    if driver is not None:
        facts.append(f"Driver: {driver}")
    d.device_facts.return_value = tuple(facts)
    return d


def _make_fake_winml_ep(devices: list[MagicMock]) -> MagicMock:
    """Build a WinMLEP stand-in carrying the given WinMLDevice list."""
    ep = MagicMock()
    ep.devices = devices
    return ep


class TestDeviceInfoEnrichment:
    """``_gather_device_info`` folds WinMLDevice.device_facts() into details."""

    def test_device_info_enriched_with_winml_device_facts(self) -> None:
        """A registered EP's WinMLDevice contributes architecture to details."""
        # Stub the sysinfo enumerators so _gather_device_info sees a single
        # NPU entry whose name lines up with a fake registered WinMLDevice.
        npu_item = MagicMock(
            name="Intel(R) AI Boost",
            driver_version="32.0.100.4023",
            manufacturer="Intel",
        )
        npu_item.name = "Intel(R) AI Boost"

        fake_winml_device = _make_fake_winml_device(
            device_type="NPU",
            hardware_name="Intel(R) AI Boost",
            architecture="4000",
        )
        fake_winml_ep = _make_fake_winml_ep([fake_winml_device])
        fake_registry = MagicMock()
        fake_registry._registered = {"some/path/openvino.dll": fake_winml_ep}

        with (
            patch(
                "winml.modelkit.sysinfo.NPU.get_all", return_value=[npu_item]
            ),
            patch("winml.modelkit.sysinfo.GPU.get_all", return_value=[]),
            patch("winml.modelkit.sysinfo.CPU.get_all", return_value=[]),
            patch(
                "winml.modelkit.session.WinMLEPRegistry.instance",
                return_value=fake_registry,
            ),
        ):
            result = _gather_device_info()

        assert len(result) == 1
        npu_entry = result[0]
        assert npu_entry["type"] == "NPU"
        assert npu_entry["name"] == "Intel(R) AI Boost"
        # The architecture fact came from WinMLDevice.device_facts().
        assert npu_entry["details"]["architecture"] == "4000"
        # sysinfo-provided driver is preserved (setdefault doesn't clobber).
        assert npu_entry["details"]["driver"] == "32.0.100.4023"

    def test_device_info_no_enrichment_without_match(self) -> None:
        """No hardware_name match → details stay at sysinfo-only values."""
        npu_item = MagicMock(
            name="Intel(R) AI Boost",
            driver_version="32.0.100.4023",
            manufacturer="Intel",
        )
        npu_item.name = "Intel(R) AI Boost"

        # WinMLDevice reports a different hardware_name — no match.
        fake_winml_device = _make_fake_winml_device(
            device_type="NPU",
            hardware_name="Some Other NPU",
            architecture="ignored",
        )
        fake_winml_ep = _make_fake_winml_ep([fake_winml_device])
        fake_registry = MagicMock()
        fake_registry._registered = {"some/path/openvino.dll": fake_winml_ep}

        with (
            patch(
                "winml.modelkit.sysinfo.NPU.get_all", return_value=[npu_item]
            ),
            patch("winml.modelkit.sysinfo.GPU.get_all", return_value=[]),
            patch("winml.modelkit.sysinfo.CPU.get_all", return_value=[]),
            patch(
                "winml.modelkit.session.WinMLEPRegistry.instance",
                return_value=fake_registry,
            ),
        ):
            result = _gather_device_info()

        assert len(result) == 1
        npu_entry = result[0]
        # No architecture key was injected — only sysinfo's keys present.
        assert "architecture" not in npu_entry["details"]
        assert npu_entry["details"]["driver"] == "32.0.100.4023"

    def test_device_info_first_match_wins(self) -> None:
        """When multiple EPs see the same device, first one in registry wins."""
        npu_item = MagicMock(
            name="Intel(R) AI Boost", driver_version=None, manufacturer="Intel",
        )
        npu_item.name = "Intel(R) AI Boost"

        first_device = _make_fake_winml_device(
            device_type="NPU",
            hardware_name="Intel(R) AI Boost",
            architecture="from-first",
        )
        second_device = _make_fake_winml_device(
            device_type="NPU",
            hardware_name="Intel(R) AI Boost",
            architecture="from-second",
        )
        # Insertion order: first DLL precedes second.
        fake_registry = MagicMock()
        fake_registry._registered = {
            "first.dll": _make_fake_winml_ep([first_device]),
            "second.dll": _make_fake_winml_ep([second_device]),
        }

        with (
            patch(
                "winml.modelkit.sysinfo.NPU.get_all", return_value=[npu_item]
            ),
            patch("winml.modelkit.sysinfo.GPU.get_all", return_value=[]),
            patch("winml.modelkit.sysinfo.CPU.get_all", return_value=[]),
            patch(
                "winml.modelkit.session.WinMLEPRegistry.instance",
                return_value=fake_registry,
            ),
        ):
            result = _gather_device_info()

        assert result[0]["details"]["architecture"] == "from-first"

    def test_device_info_registry_failure_is_non_fatal(self) -> None:
        """If the registry blows up, sysinfo results still come through."""
        npu_item = MagicMock(
            name="Intel(R) AI Boost",
            driver_version="32.0.100",
            manufacturer="Intel",
        )
        npu_item.name = "Intel(R) AI Boost"

        with (
            patch(
                "winml.modelkit.sysinfo.NPU.get_all", return_value=[npu_item]
            ),
            patch("winml.modelkit.sysinfo.GPU.get_all", return_value=[]),
            patch("winml.modelkit.sysinfo.CPU.get_all", return_value=[]),
            patch(
                "winml.modelkit.session.WinMLEPRegistry.instance",
                side_effect=RuntimeError("registry exploded"),
            ),
        ):
            result = _gather_device_info()

        # Sysinfo-only details survive the registry failure.
        assert len(result) == 1
        assert result[0]["details"]["driver"] == "32.0.100"
        assert "architecture" not in result[0]["details"]
