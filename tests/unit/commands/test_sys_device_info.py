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

from typing import Any
from unittest.mock import MagicMock, patch

from winml.modelkit.commands.sys import _gather_device_info


def _fake_ep_info(
    *,
    device_type: str,
    hardware_name: str,
    architecture: str | None = None,
    driver: str | None = None,
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
        assert "architecture" not in result[0]["details"]
