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
    _DEVICE_EP_MAP,
    _EP_DEVICE_MAP,
    _get_available_devices,
    resolve_device,
)
from winml.modelkit.utils.constants import EP_NAMES


class TestGetAvailableDevices:
    """Tests for _get_available_devices()."""

    def test_with_npu_and_gpu(self) -> None:
        """When NPU and GPU are present, returns ("npu", "gpu", "cpu")."""
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

        assert devices == ("npu", "gpu", "cpu")

    def test_no_npu_with_gpu(self) -> None:
        """When no NPU but GPU present, returns ("gpu", "cpu")."""
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

        assert devices == ("gpu", "cpu")

    def test_no_npu_no_gpu(self) -> None:
        """When no NPU and no GPU, returns ("cpu",)."""
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

        assert devices == ("cpu",)

    def test_cpu_always_present(self) -> None:
        """CPU is always in the result, even if detection fails."""
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
        assert devices == ("cpu",)

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

        assert devices == ("gpu", "cpu")
        assert "npu" not in devices


class TestMappingConstants:
    """Tests for _EP_DEVICE_MAP and _DEVICE_EP_MAP constants."""

    def test_ep_device_map_has_required_entries(self) -> None:
        """_EP_DEVICE_MAP has all standard EPs."""
        # NVIDIA
        assert "NvTensorRTRTXExecutionProvider" in _EP_DEVICE_MAP
        # AMD
        assert "MIGraphXExecutionProvider" in _EP_DEVICE_MAP
        assert "VitisAIExecutionProvider" in _EP_DEVICE_MAP
        # Qualcomm
        assert "QNNExecutionProvider" in _EP_DEVICE_MAP
        # Microsoft
        assert "DmlExecutionProvider" in _EP_DEVICE_MAP
        # Intel
        assert "OpenVINOExecutionProvider" in _EP_DEVICE_MAP
        # CPU
        assert "CPUExecutionProvider" in _EP_DEVICE_MAP

    def test_ep_device_map_covers_every_ep_name_literal(self) -> None:
        """Every value in the ``EPName`` Literal must have an _EP_DEVICE_MAP entry.

        Adding a new canonical EP to the Literal without wiring its device
        mapping here would silently break device resolution; this test guards
        against that drift.
        """
        missing = set(EP_NAMES) - set(_EP_DEVICE_MAP)
        assert not missing, f"_EP_DEVICE_MAP is missing entries for: {sorted(missing)}"

    def test_ep_device_map_no_extra_keys_outside_literal(self) -> None:
        """_EP_DEVICE_MAP must not contain keys absent from the ``EPName`` Literal."""
        extra = set(_EP_DEVICE_MAP) - set(EP_NAMES)
        assert not extra, f"_EP_DEVICE_MAP has unexpected canonical names: {sorted(extra)}"

    def test_ep_device_map_values_are_lowercase(self) -> None:
        """All _EP_DEVICE_MAP values should be lowercase."""
        for ep, device in _EP_DEVICE_MAP.items():
            assert device == device.lower(), f"{ep} maps to non-lowercase '{device}'"

    def test_device_ep_map_includes_multi_device_eps(self) -> None:
        """Multi-device EPs (QNN, OpenVINO) should appear in each device."""
        assert "QNNExecutionProvider" in _DEVICE_EP_MAP["npu"]
        assert "QNNExecutionProvider" in _DEVICE_EP_MAP["gpu"]
        assert "OpenVINOExecutionProvider" in _DEVICE_EP_MAP["npu"]
        assert "OpenVINOExecutionProvider" in _DEVICE_EP_MAP["gpu"]
        assert "OpenVINOExecutionProvider" in _DEVICE_EP_MAP["cpu"]

    def test_device_ep_map_derived_from_ep_device_map(self) -> None:
        """_DEVICE_EP_MAP should be consistent with _EP_DEVICE_MAP."""
        for device, eps in _DEVICE_EP_MAP.items():
            for ep in eps:
                assert ep in _EP_DEVICE_MAP, (
                    f"EP '{ep}' in _DEVICE_EP_MAP but not in _EP_DEVICE_MAP"
                )
                assert device in _EP_DEVICE_MAP[ep].split("/")

    def test_nv_tensorrt_rtx_is_gpu_ep(self) -> None:
        """NvTensorRTRTXExecutionProvider should map to gpu."""
        assert _EP_DEVICE_MAP["NvTensorRTRTXExecutionProvider"] == "gpu"
        assert "NvTensorRTRTXExecutionProvider" in _DEVICE_EP_MAP["gpu"]


class TestResolveDevice:
    """Tests for resolve_device()."""

    def setup_method(self) -> None:
        """Mock _get_available_eps so the real function never runs.

        Tests stack their own ``with patch(...)`` on top of this default mock
        to set test-specific return values; the inner patch shadows ours for
        the duration of the with-block and restores back on exit.
        """
        self._eps_patcher = patch(
            "winml.modelkit.sysinfo.device._get_available_eps",
            return_value=frozenset(),
        )
        self._eps_patcher.start()

    def teardown_method(self) -> None:
        self._eps_patcher.stop()

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
            device, available = resolve_device("auto")

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
            device, available = resolve_device("auto")

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
            device, available = resolve_device("auto")

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
            device, _available = resolve_device("auto")

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
            device, available = resolve_device("gpu")

        assert device == "gpu"
        assert available == ["npu", "gpu", "cpu"]

    def test_resolve_device_explicit_invalid(self) -> None:
        """Unrecognized device "tpu" -> raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device 'tpu'"):
            resolve_device("tpu")

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
            device, available = resolve_device("npu")

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
            device, _ = resolve_device("CPU")

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
            resolve_device("auto")

        assert any("No execution providers detected" in record.message for record in caplog.records)


class TestResolveDeviceWithEp:
    """Tests for resolve_device(ep=...) — EP-aware filtering of available_devices/eps."""

    def setup_method(self) -> None:
        """Mock _get_available_eps so the real function never runs."""
        self._eps_patcher = patch(
            "winml.modelkit.sysinfo.device._get_available_eps",
            return_value=frozenset(),
        )
        self._eps_patcher.start()

    def teardown_method(self) -> None:
        self._eps_patcher.stop()

    def test_ep_qnn_filters_devices_to_npu_and_gpu(self) -> None:
        """ep='qnn' narrows available_devices to QNN's compatible devices (npu/gpu)."""
        with (
            patch(
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=("npu", "gpu", "cpu"),
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
            device, available = resolve_device("auto", ep="qnn")

        assert device == "npu"
        assert available == ["npu", "gpu"]

    def test_ep_qnn_auto_picks_gpu_when_no_npu(self) -> None:
        """ep='qnn' on a GPU-only system auto-selects gpu."""
        with (
            patch(
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=("gpu", "cpu"),
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
                return_value=frozenset(
                    {"QNNExecutionProvider", "CPUExecutionProvider"},
                ),
            ),
        ):
            device, available = resolve_device("auto", ep="qnn")

        assert device == "gpu"
        assert available == ["gpu"]

    def test_ep_dml_filters_to_gpu_only(self) -> None:
        """ep='dml' narrows available_devices to gpu (DML is gpu-only)."""
        with (
            patch(
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=("npu", "gpu", "cpu"),
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
                return_value=frozenset(
                    {"DmlExecutionProvider", "CPUExecutionProvider"},
                ),
            ),
        ):
            device, available = resolve_device("auto", ep="dml")

        assert device == "gpu"
        assert available == ["gpu"]

    def test_ep_filters_available_eps(self, caplog) -> None:
        """ep='vitisai' filters available_eps; if EP isn't actually available,
        no compatible EP remains, so the no-EPs warning fires."""
        with (
            patch(
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=("npu", "gpu", "cpu"),
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
            caplog.at_level(logging.WARNING, logger="winml.modelkit.sysinfo.device"),
        ):
            # vitisai is not in the available set above; after filtering,
            # available_eps becomes empty.
            device, _ = resolve_device("auto", ep="vitisai")

        assert device == "cpu"  # auto-fallback when no EP matches
        assert any("No execution providers detected" in record.message for record in caplog.records)

    def test_ep_unknown_raises(self) -> None:
        """Unknown ep short name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown EP 'tpu'"):
            resolve_device("auto", ep="tpu")

    def test_ep_case_insensitive(self) -> None:
        """ep argument is case-insensitive."""
        with (
            patch(
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=("npu", "gpu", "cpu"),
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
                return_value=frozenset(
                    {"QNNExecutionProvider", "CPUExecutionProvider"},
                ),
            ),
        ):
            device, available = resolve_device("auto", ep="QNN")

        assert device == "npu"
        assert available == ["npu", "gpu"]

    def test_ep_explicit_device_filtered_out(self, caplog) -> None:
        """ep='qnn' + device='cpu' returns 'cpu' but available_devices excludes cpu."""
        with (
            patch(
                "winml.modelkit.sysinfo.device._get_available_devices",
                return_value=("npu", "gpu", "cpu"),
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
                return_value=frozenset(
                    {"QNNExecutionProvider", "CPUExecutionProvider"},
                ),
            ),
            caplog.at_level(logging.WARNING, logger="winml.modelkit.sysinfo.device"),
        ):
            device, available = resolve_device("cpu", ep="qnn")

        assert device == "cpu"
        assert available == ["npu", "gpu"]
        assert any("no compatible EP found" in record.message for record in caplog.records)
