# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.sysinfo.device module."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.sysinfo.device import (
    _get_available_devices,
    resolve_device_category,
)


class TestGetAvailableDevices:
    """Tests for _get_available_devices()."""

    def test_with_npu_and_gpu(self) -> None:
        """When NPU and GPU are present, returns ["npu", "gpu", "cpu"]."""
        mock_npu = MagicMock()
        mock_gpu = MagicMock()

        with (
            patch(
                "winml.modelkit.sysinfo.hardware.NPU.get_all",
                return_value=[mock_npu],
            ),
            patch(
                "winml.modelkit.sysinfo.hardware.GPU.get_all",
                return_value=[mock_gpu],
            ),
        ):
            devices = _get_available_devices()

        assert devices == ["npu", "gpu", "cpu"]

    def test_no_npu_with_gpu(self) -> None:
        """When no NPU but GPU present, returns ["gpu", "cpu"]."""
        mock_gpu = MagicMock()

        with (
            patch(
                "winml.modelkit.sysinfo.hardware.NPU.get_all",
                return_value=[],
            ),
            patch(
                "winml.modelkit.sysinfo.hardware.GPU.get_all",
                return_value=[mock_gpu],
            ),
        ):
            devices = _get_available_devices()

        assert devices == ["gpu", "cpu"]

    def test_no_npu_no_gpu(self) -> None:
        """When no NPU and no GPU, returns ["cpu"]."""
        with (
            patch(
                "winml.modelkit.sysinfo.hardware.NPU.get_all",
                return_value=[],
            ),
            patch(
                "winml.modelkit.sysinfo.hardware.GPU.get_all",
                return_value=[],
            ),
        ):
            devices = _get_available_devices()

        assert devices == ["cpu"]

    def test_cpu_always_present(self) -> None:
        """CPU is always in the result list, even if detection fails."""
        with (
            patch(
                "winml.modelkit.sysinfo.hardware.NPU.get_all",
                side_effect=RuntimeError("WMI failed"),
            ),
            patch(
                "winml.modelkit.sysinfo.hardware.GPU.get_all",
                side_effect=RuntimeError("WMI failed"),
            ),
        ):
            devices = _get_available_devices()

        assert "cpu" in devices
        assert devices == ["cpu"]

    def test_npu_detection_failure_falls_through(self) -> None:
        """If NPU detection raises, GPU and CPU still appear."""
        mock_gpu = MagicMock()

        with (
            patch(
                "winml.modelkit.sysinfo.hardware.NPU.get_all",
                side_effect=RuntimeError("WMI failed"),
            ),
            patch(
                "winml.modelkit.sysinfo.hardware.GPU.get_all",
                return_value=[mock_gpu],
            ),
        ):
            devices = _get_available_devices()

        assert devices == ["gpu", "cpu"]
        assert "npu" not in devices


class TestResolveDevice:
    """Tests for resolve_device_category()."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test."""
        from winml.modelkit.sysinfo.device import _get_available_eps

        _get_available_eps.cache_clear()

    def test_resolve_device_auto_npu_with_ep(self) -> None:
        """Auto mode: NPU hardware + QNN EP -> returns "npu"."""
        with (
            patch(
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
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
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
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
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=["gpu", "cpu"],
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
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
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
                return_value=frozenset(),
            ),
        ):
            device, _available = resolve_device_category("auto")

        assert device == "cpu"

    def test_resolve_device_explicit_valid(self) -> None:
        """Explicit device "gpu" -> returns "gpu"."""
        with (
            patch(
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
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
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=["npu", "gpu", "cpu"],
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
                return_value=frozenset({"CPUExecutionProvider"}),
            ),
            caplog.at_level(logging.WARNING, logger="winml.modelkit.sysinfo.device"),
        ):
            device, available = resolve_device_category("npu")

        assert device == "npu"
        assert available == ["npu", "gpu", "cpu"]
        assert any("no compatible EP found" in record.message for record in caplog.records)

    def test_resolve_device_case_insensitive(self) -> None:
        """Device argument should be case-insensitive."""
        with (
            patch(
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=["cpu"],
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
                return_value=frozenset({"CPUExecutionProvider"}),
            ),
        ):
            device, _ = resolve_device_category("CPU")

        assert device == "cpu"

    def test_resolve_device_empty_eps_warns(self, caplog) -> None:
        """When no EPs are detected, a warning is logged."""
        with (
            patch(
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=["cpu"],
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
                return_value=frozenset(),
            ),
            caplog.at_level(logging.WARNING, logger="winml.modelkit.sysinfo.device"),
        ):
            resolve_device_category("auto")

        assert any("No execution providers detected" in record.message for record in caplog.records)


def test_resolve_device_category_returns_category_and_eps() -> None:
    """Smoke: function still returns a (category, list) tuple under new name."""
    category, eps = resolve_device_category("auto")
    assert isinstance(category, str)
    assert isinstance(eps, list)
