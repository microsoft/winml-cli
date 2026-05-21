# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for session.auto_detect_device."""

from __future__ import annotations

import logging
from unittest.mock import patch

from winml.modelkit.session import auto_detect_device


class TestAutoDetectDevice:
    """Tests for auto_detect_device()."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test."""
        from winml.modelkit.session.ep_registry import available_eps

        available_eps.cache_clear()

    def test_auto_detect_npu_with_ep(self) -> None:
        """NPU hardware + QNN EP -> returns "npu"."""
        with (
            patch(
                "winml.modelkit.sysinfo.hardware.get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch(
                "winml.modelkit.session.ep_registry.available_eps",
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
                "winml.modelkit.sysinfo.hardware.get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch(
                "winml.modelkit.session.ep_registry.available_eps",
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
                "winml.modelkit.sysinfo.hardware.get_available_devices",
                return_value=["gpu", "cpu"],
            ),
            patch(
                "winml.modelkit.session.ep_registry.available_eps",
                return_value=frozenset({"CPUExecutionProvider"}),
            ),
        ):
            device = auto_detect_device()

        assert device == "cpu"

    def test_auto_detect_no_eps_falls_back_to_cpu(self) -> None:
        """No EPs at all -> falls back to CPU."""
        with (
            patch(
                "winml.modelkit.sysinfo.hardware.get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch(
                "winml.modelkit.session.ep_registry.available_eps",
                return_value=frozenset(),
            ),
        ):
            device = auto_detect_device()

        assert device == "cpu"

    def test_auto_detect_empty_eps_warns(self, caplog) -> None:
        """When no EPs are detected, a warning is logged."""
        with (
            patch(
                "winml.modelkit.sysinfo.hardware.get_available_devices",
                return_value=["cpu"],
            ),
            patch(
                "winml.modelkit.session.ep_registry.available_eps",
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
