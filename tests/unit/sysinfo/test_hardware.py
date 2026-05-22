# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.sysinfo.hardware module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from winml.modelkit.sysinfo.hardware import get_available_devices


class TestGetAvailableDevices:
    """Tests for get_available_devices()."""

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
            devices = get_available_devices()

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
            devices = get_available_devices()

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
            devices = get_available_devices()

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
            devices = get_available_devices()

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
            devices = get_available_devices()

        assert devices == ["gpu", "cpu"]
        assert "npu" not in devices
