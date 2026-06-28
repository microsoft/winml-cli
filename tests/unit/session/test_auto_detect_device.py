# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for session.auto_detect_device."""

from __future__ import annotations

import logging
from unittest.mock import patch

from winml.modelkit.session import auto_detect_device
from winml.modelkit.session.ep_registry import WinMLEPRegistry


class TestAutoDetectDevice:
    """Tests for auto_detect_device()."""

    def test_auto_detect_npu_with_ep(self) -> None:
        """NPU hardware + QNN EP -> returns "npu"."""
        with (
            patch(
                "winml.modelkit.sysinfo.get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch.object(
                WinMLEPRegistry, "available_eps",
                return_value=frozenset(
                    {
                        "QNNExecutionProvider",
                        "DmlExecutionProvider",
                        "CPUExecutionProvider",
                    }
                ),
            ),
        ):
            device = auto_detect_device()

        assert device == "npu"

    def test_auto_detect_npu_without_ep_falls_through_to_gpu(self) -> None:
        """NPU hardware + no QNN EP -> falls through to GPU."""
        with (
            patch(
                "winml.modelkit.sysinfo.get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch.object(
                WinMLEPRegistry, "available_eps",
                return_value=frozenset(
                    {
                        "DmlExecutionProvider",
                        "CPUExecutionProvider",
                    }
                ),
            ),
        ):
            device = auto_detect_device()

        assert device == "gpu"

    def test_auto_detect_cpu_fallback_when_gpu_ep_missing(self) -> None:
        """GPU hardware but no GPU EP -> falls through to CPU."""
        with (
            patch(
                "winml.modelkit.sysinfo.get_available_devices",
                return_value=["gpu", "cpu"],
            ),
            patch.object(
                WinMLEPRegistry, "available_eps",
                return_value=frozenset({"CPUExecutionProvider"}),
            ),
        ):
            device = auto_detect_device()

        assert device == "cpu"

    def test_auto_detect_no_eps_falls_back_to_cpu(self) -> None:
        """No EPs at all -> falls back to CPU."""
        with (
            patch(
                "winml.modelkit.sysinfo.get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch.object(
                WinMLEPRegistry, "available_eps",
                return_value=frozenset(),
            ),
        ):
            device = auto_detect_device()

        assert device == "cpu"

    def test_auto_detect_empty_eps_warns(self, caplog) -> None:
        """When no EPs are detected, a warning is logged."""
        with (
            patch(
                "winml.modelkit.sysinfo.get_available_devices",
                return_value=["cpu"],
            ),
            patch.object(
                WinMLEPRegistry, "available_eps",
                return_value=frozenset(),
            ),
            caplog.at_level(logging.WARNING, logger="winml.modelkit.session.ep_device"),
        ):
            auto_detect_device()

        assert any("No execution providers detected" in record.message for record in caplog.records)


def test_auto_detect_device_returns_string() -> None:
    """Smoke: function returns a non-empty device-category string."""
    device = auto_detect_device()
    assert isinstance(device, str)
    assert device in {"npu", "gpu", "cpu"}


def test_auto_detect_device_falls_back_to_cpu_on_vendor_detection_failure(
    caplog,
) -> None:
    """RuntimeError from EP_CATALOG.is_compatible must not crash CLI.

    On headless servers _get_detected_vendors raises RuntimeError which
    propagates through is_compatible. auto_detect_device must catch and
    fall back to "cpu" with a warning, never let the traceback reach
    the click command layer.
    """
    from winml.modelkit.ep_path import EPCatalog

    with (
        patch(
            "winml.modelkit.sysinfo.get_available_devices",
            return_value=["npu", "gpu", "cpu"],
        ),
        patch.object(
            WinMLEPRegistry, "available_eps",
            return_value=frozenset(
                {"QNNExecutionProvider", "CPUExecutionProvider"}
            ),
        ),
        patch.object(
            EPCatalog, "is_compatible",
            side_effect=RuntimeError("WMI unavailable"),
        ),
        caplog.at_level(logging.WARNING, logger="winml.modelkit.session.ep_device"),
    ):
        device = auto_detect_device()

    assert device == "cpu"
    assert any(
        "vendor detection" in record.message.lower()
        or "wmi" in record.message.lower()
        or "hardware detection" in record.message.lower()
        for record in caplog.records
    )


def test_default_ep_for_device_returns_none_on_vendor_detection_failure() -> None:
    """RuntimeError from EP_CATALOG.is_compatible must yield None per contract.

    default_ep_for_device's return type already encodes "no compatible
    EP found" as None; a hardware-detection failure is functionally
    equivalent and must not propagate as an exception.
    """
    from winml.modelkit.ep_path import EPCatalog
    from winml.modelkit.session.ep_device import default_ep_for_device

    with (
        patch.object(
            WinMLEPRegistry, "available_eps",
            return_value=frozenset({"QNNExecutionProvider"}),
        ),
        patch.object(
            EPCatalog, "is_compatible",
            side_effect=RuntimeError("WMI unavailable"),
        ),
    ):
        result = default_ep_for_device("npu")

    assert result is None
