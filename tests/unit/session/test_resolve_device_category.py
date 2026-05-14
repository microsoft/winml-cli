# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for session.resolve_device_category."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from winml.modelkit.session import resolve_device_category


class TestResolveDevice:
    """Tests for resolve_device_category()."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test."""
        from winml.modelkit.session.ep_registry import available_eps

        available_eps.cache_clear()

    def test_resolve_device_auto_npu_with_ep(self) -> None:
        """Auto mode: NPU hardware + QNN EP -> returns "npu"."""
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
            device, available = resolve_device_category("auto")

        assert device == "npu"
        assert available == ["npu", "gpu", "cpu"]

    def test_resolve_device_auto_npu_without_ep(self) -> None:
        """Auto mode: NPU hardware + no QNN EP -> falls through to GPU or CPU."""
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
            device, available = resolve_device_category("auto")

        assert device == "gpu"
        assert available == ["npu", "gpu", "cpu"]

    def test_resolve_device_auto_cpu_fallback(self) -> None:
        """Auto mode: GPU hardware but no GPU EP -> falls through to CPU."""
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
            device, available = resolve_device_category("auto")

        assert device == "cpu"
        assert available == ["gpu", "cpu"]

    def test_resolve_device_auto_no_eps(self) -> None:
        """Auto mode: no EPs at all -> falls back to CPU."""
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
            device, _available = resolve_device_category("auto")

        assert device == "cpu"

    def test_resolve_device_explicit_valid(self) -> None:
        """Explicit device "gpu" -> returns "gpu"."""
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
            device, available = resolve_device_category("gpu")

        assert device == "gpu"
        assert available == ["npu", "gpu", "cpu"]

    def test_resolve_device_explicit_invalid(self) -> None:
        """Unrecognized device "tpu" -> raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device 'tpu'"):
            resolve_device_category("tpu")

    def test_resolve_device_explicit_no_ep_warns(self, caplog) -> None:
        """Explicit "npu" but no QNN EP -> returns "npu" with warning."""
        with (
            patch(
                "winml.modelkit.sysinfo.hardware.get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch(
                "winml.modelkit.session.ep_registry.available_eps",
                return_value=frozenset({"CPUExecutionProvider"}),
            ),
            caplog.at_level(logging.WARNING, logger="winml.modelkit.session.ep_device"),
        ):
            device, available = resolve_device_category("npu")

        assert device == "npu"
        assert available == ["npu", "gpu", "cpu"]
        assert any("no compatible EP found" in record.message for record in caplog.records)

    def test_resolve_device_case_insensitive(self) -> None:
        """Device argument should be case-insensitive."""
        with (
            patch(
                "winml.modelkit.sysinfo.hardware.get_available_devices",
                return_value=["cpu"],
            ),
            patch(
                "winml.modelkit.session.ep_registry.available_eps",
                return_value=frozenset({"CPUExecutionProvider"}),
            ),
        ):
            device, _ = resolve_device_category("CPU")

        assert device == "cpu"

    def test_resolve_device_empty_eps_warns(self, caplog) -> None:
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
            resolve_device_category("auto")

        assert any("No execution providers detected" in record.message for record in caplog.records)


def test_resolve_device_category_returns_category_and_eps() -> None:
    """Smoke: function still returns a (category, list) tuple under new name."""
    category, eps = resolve_device_category("auto")
    assert isinstance(category, str)
    assert isinstance(eps, list)
