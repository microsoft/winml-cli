# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the device-section renderer in ``commands/sys.py``.

Covers the ``device_facts`` enrichment path: when the EP inventory
carries a matching per-device entry for a hardware device that sysinfo
also reported, the renderer folds Architecture (and Driver, when sysinfo
lacks one) into the per-device ``details`` dict so the *Available
Devices* section displays device-intrinsic facts per
``docs/design/session/4_winml_device.md`` §4.1.

Post-refactor, ``_gather_device_info`` reads from an ``ep_info``
argument (the output of :func:`_gather_ep_info`) rather than from
``WinMLEPRegistry._registered`` — because filesystem-backed EPs are
registered in isolated subprocesses whose live handles never exist in
the parent's registry.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import click
import pytest

from winml.modelkit.commands.sys import (
    _gather,
    _gather_device_info,
    _get_memory_info,
    _render_compact,
)


def _fake_ep_info(
    *,
    device_type: str,
    hardware_name: str,
    architecture: str | None = None,
    driver: str | None = None,
    luid: str | None = None,
    ep_name: str = "OpenVINOExecutionProvider",
) -> dict[str, dict[str, Any]]:
    """Build an ``ep_info``-shaped dict with a single per-source device row.

    Mirrors the shape :func:`_gather_ep_info` returns:
    ``{ep_name: {"entries": [{"devices": [{...}]}]}}``. Each device
    entry carries the ``device_facts`` list ``_gather_device_info`` reads.
    """
    facts: list[str] = []
    if architecture is not None:
        facts.append(f"Architecture: {architecture}")
    if driver is not None:
        facts.append(f"Driver: {driver}")
    return {
        ep_name: {
            "entries": [
                {
                    "devices": [
                        {
                            "device_type": device_type,
                            "hardware_name": hardware_name,
                            "luid": luid,
                            "device_facts": facts,
                        }
                    ]
                }
            ]
        }
    }


class TestDeviceInfoEnrichment:
    """``_gather_device_info`` folds device_facts from ep_info into details."""

    def test_device_info_enriched_with_winml_device_facts(self) -> None:
        """A matching per-source device contributes architecture to details."""
        npu_item = MagicMock(
            name="Intel(R) AI Boost",
            driver_version="32.0.100.4023",
            manufacturer="Intel",
        )
        npu_item.name = "Intel(R) AI Boost"

        ep_info = _fake_ep_info(
            device_type="NPU",
            hardware_name="Intel(R) AI Boost",
            architecture="4000",
            luid="0x00000000_0x00018393",
        )

        with (
            patch("winml.modelkit.sysinfo.NPU.get_all", return_value=[npu_item]),
            patch("winml.modelkit.sysinfo.GPU.get_all", return_value=[]),
            patch("winml.modelkit.sysinfo.CPU.get_all", return_value=[]),
        ):
            result = _gather_device_info(ep_info)

        assert len(result) == 1
        npu_entry = result[0]
        assert npu_entry["type"] == "NPU"
        assert npu_entry["name"] == "Intel(R) AI Boost"
        # The architecture came from the ep_info device row's device_facts.
        assert npu_entry["details"]["architecture"] == "4000"
        assert npu_entry["details"]["luid"] == "0x00000000_0x00018393"
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

        ep_info = _fake_ep_info(
            device_type="NPU",
            hardware_name="Some Other NPU",
            architecture="ignored",
        )

        with (
            patch("winml.modelkit.sysinfo.NPU.get_all", return_value=[npu_item]),
            patch("winml.modelkit.sysinfo.GPU.get_all", return_value=[]),
            patch("winml.modelkit.sysinfo.CPU.get_all", return_value=[]),
        ):
            result = _gather_device_info(ep_info)

        assert len(result) == 1
        npu_entry = result[0]
        assert "architecture" not in npu_entry["details"]
        assert npu_entry["details"]["luid"] is None
        assert npu_entry["details"]["driver"] == "32.0.100.4023"

    def test_device_info_first_match_wins(self) -> None:
        """When multiple sources see the same device, first one in ep_info wins."""
        npu_item = MagicMock(
            name="Intel(R) AI Boost", driver_version=None, manufacturer="Intel",
        )
        npu_item.name = "Intel(R) AI Boost"

        # Two entries under the same EP name — the first source wins per
        # the enrichment's first-match-wins contract (device_facts are
        # device-intrinsic, so all sources should agree; if they don't,
        # taking the first one is a defensible tiebreak).
        ep_info: dict[str, dict[str, Any]] = {
            "OpenVINOExecutionProvider": {
                "entries": [
                    {
                        "devices": [
                            {
                                "device_type": "NPU",
                                "hardware_name": "Intel(R) AI Boost",
                                "device_facts": ["Architecture: from-first"],
                            }
                        ]
                    },
                    {
                        "devices": [
                            {
                                "device_type": "NPU",
                                "hardware_name": "Intel(R) AI Boost",
                                "device_facts": ["Architecture: from-second"],
                            }
                        ]
                    },
                ]
            }
        }

        with (
            patch("winml.modelkit.sysinfo.NPU.get_all", return_value=[npu_item]),
            patch("winml.modelkit.sysinfo.GPU.get_all", return_value=[]),
            patch("winml.modelkit.sysinfo.CPU.get_all", return_value=[]),
        ):
            result = _gather_device_info(ep_info)

        assert result[0]["details"]["architecture"] == "from-first"

    def test_device_info_no_ep_info_is_non_fatal(self) -> None:
        """No ep_info arg (default None) → sysinfo results still come through."""
        npu_item = MagicMock(
            name="Intel(R) AI Boost",
            driver_version="32.0.100",
            manufacturer="Intel",
        )
        npu_item.name = "Intel(R) AI Boost"

        with (
            patch("winml.modelkit.sysinfo.NPU.get_all", return_value=[npu_item]),
            patch("winml.modelkit.sysinfo.GPU.get_all", return_value=[]),
            patch("winml.modelkit.sysinfo.CPU.get_all", return_value=[]),
        ):
            result = _gather_device_info()

        # Sysinfo-only details survive without any ep_info to enrich from.
        assert len(result) == 1
        assert result[0]["details"]["driver"] == "32.0.100"
        assert result[0]["details"]["luid"] is None
        assert "architecture" not in result[0]["details"]

class TestMemoryInfo:
    """System memory metadata is fast, explicit, and best-effort."""

    def test_physical_total_is_reported_in_mib(self) -> None:
        memory = MagicMock(total=24 * 1024 * 1024 * 1024)
        with patch("psutil.virtual_memory", return_value=memory):
            assert _get_memory_info() == {"physical_total_mib": 24 * 1024}

    def test_probe_failure_returns_unknown_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            patch("psutil.virtual_memory", side_effect=RuntimeError("unavailable")),
            caplog.at_level(logging.WARNING, logger="winml.modelkit.commands.sys"),
        ):
            assert _get_memory_info() == {"physical_total_mib": None}

        assert "Failed to get physical memory details: unavailable" in caplog.text

    def test_non_positive_total_returns_unknown_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        memory = MagicMock(total=0)
        with (
            patch("psutil.virtual_memory", return_value=memory),
            caplog.at_level(logging.WARNING, logger="winml.modelkit.commands.sys"),
        ):
            assert _get_memory_info() == {"physical_total_mib": None}

        assert "physical memory total must be greater than zero" in caplog.text


def test_compact_device_output_includes_luid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _render_compact(
        {
            "devices": [
                {
                    "type": "GPU",
                    "name": "Test GPU",
                    "details": {"luid": "0x00000000_0x00018393"},
                },
                {
                    "type": "CPU",
                    "name": "Test CPU",
                    "details": {"luid": None},
                },
            ]
        },
        False,
    )

    output = capsys.readouterr().out
    assert "GPU: Test GPU (LUID: 0x00000000_0x00018393)" in output
    assert "CPU: Test CPU (LUID: N/A)" in output


class TestGatherDeviceSectionEnrichment:
    """``_gather`` must enrich devices even when the EP section isn't emitted.

    ``winml sys --list-device`` asks for devices only, but ``ep_info`` is the
    sole carrier of ``device_facts``; gathering it only for ``eps=True`` made
    that view drop ``details`` keys the default view shows for the very same
    hardware.
    """

    def test_device_only_view_still_gathers_ep_info(self) -> None:
        ep_info = _fake_ep_info(
            device_type="NPU",
            hardware_name="Intel(R) AI Boost",
            architecture="4000",
        )

        with (
            patch("winml.modelkit.commands.sys._gather_ep_info", return_value=ep_info) as mock_eps,
            patch(
                "winml.modelkit.commands.sys._gather_device_info", return_value=[]
            ) as mock_devices,
        ):
            info = _gather(devices=True, tolerant=False)

        mock_eps.assert_called_once_with()
        mock_devices.assert_called_once_with(ep_info)
        # The EP section is gathered for enrichment only, never emitted.
        assert set(info) == {"devices"}

    def test_devices_match_across_default_and_device_only_views(self) -> None:
        npu_item = MagicMock(driver_version="32.0.100.4023", manufacturer="Intel")
        npu_item.name = "Intel(R) AI Boost"
        ep_info = _fake_ep_info(
            device_type="NPU",
            hardware_name="Intel(R) AI Boost",
            architecture="4000",
        )

        with (
            patch("winml.modelkit.commands.sys._gather_ep_info", return_value=ep_info),
            patch("winml.modelkit.commands.sys._gather_system_info", return_value={}),
            patch("winml.modelkit.sysinfo.NPU.get_all", return_value=[npu_item]),
            patch("winml.modelkit.sysinfo.GPU.get_all", return_value=[]),
            patch("winml.modelkit.sysinfo.CPU.get_all", return_value=[]),
        ):
            default = _gather(system=True, devices=True, eps=True, tolerant=True)
            device_only = _gather(devices=True, tolerant=False)

        assert default["devices"] == device_only["devices"]
        assert device_only["devices"][0]["details"]["architecture"] == "4000"

    def test_ep_failure_is_non_fatal_for_device_only_view(self) -> None:
        """Enrichment is best-effort: a broken EP probe must not fail devices."""
        with (
            patch(
                "winml.modelkit.commands.sys._gather_ep_info",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "winml.modelkit.commands.sys._gather_device_info", return_value=[]
            ) as mock_devices,
        ):
            info = _gather(devices=True, tolerant=False)

        mock_devices.assert_called_once_with({})
        assert info == {"devices": []}

    def test_ep_failure_still_raises_when_ep_section_requested(self) -> None:
        """The strict contract for an explicitly pinned EP section is unchanged."""
        with (
            patch(
                "winml.modelkit.commands.sys._gather_ep_info",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(click.ClickException, match="Error detecting execution providers"),
        ):
            _gather(eps=True, tolerant=False)
