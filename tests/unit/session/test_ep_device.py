# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# tests/unit/session/test_ep_device.py
"""Unit tests for EPDevice descriptor and resolution helpers."""

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.session.ep_device import (
    AmbiguousMatch,
    DeviceNotFound,
    EPDevice,
    expand_ep_name,
    resolve_device,
    short_ep_name,
)


def test_ep_device_round_trip() -> None:
    """EPDevice -> to_dict -> from_dict yields an equal instance."""
    original = EPDevice(
        ep="QNNExecutionProvider",
        device="npu",
        vendor_id=0x4D4F,
        device_id=0x0001,
        vendor="Qualcomm",
    )
    rehydrated = EPDevice.from_dict(original.to_dict())
    assert rehydrated == original
    assert rehydrated.ep == "QNNExecutionProvider"
    assert rehydrated.device == "npu"
    assert rehydrated.vendor_id == 0x4D4F
    assert rehydrated.device_id == 0x0001
    assert rehydrated.vendor == "Qualcomm"


def test_ep_device_lowercase_invariant() -> None:
    """`device` field is forced to lowercase by __post_init__."""
    ep_device = EPDevice(
        ep="QNNExecutionProvider",
        device="NPU",
        vendor_id=0x4D4F,
        device_id=0x0001,
    )
    assert ep_device.device == "npu"


def test_expand_ep_name_short_form() -> None:
    assert expand_ep_name("qnn") == "QNNExecutionProvider"
    assert expand_ep_name("openvino") == "OpenVINOExecutionProvider"
    assert expand_ep_name("vitisai") == "VitisAIExecutionProvider"
    assert expand_ep_name("migraphx") == "MIGraphXExecutionProvider"
    assert expand_ep_name("nv_tensorrt_rtx") == "NvTensorRtRtxExecutionProvider"
    assert expand_ep_name("dml") == "DmlExecutionProvider"
    assert expand_ep_name("cpu") == "CPUExecutionProvider"


def test_expand_ep_name_passthrough() -> None:
    """Already-canonical names flow through unchanged."""
    assert expand_ep_name("QNNExecutionProvider") == "QNNExecutionProvider"
    assert expand_ep_name("CPUExecutionProvider") == "CPUExecutionProvider"


def test_expand_ep_name_alias_casing() -> None:
    """Mixed-case canonical aliases are normalized."""
    assert expand_ep_name("NvTensorRTRTXExecutionProvider") == "NvTensorRtRtxExecutionProvider"


# --- resolve_device tests ---------------------------------------------------


def _fake_ort_dev(dev_type: str, vendor_id: int, device_id: int) -> MagicMock:
    d = MagicMock()
    d.device.type.name = dev_type
    d.device.vendor_id = vendor_id
    d.device.device_id = device_id
    d.device.vendor = "Qualcomm"
    return d


def test_resolve_device_qnn_npu() -> None:
    """resolve_device selects the NPU entry when device='npu'."""
    devices = [
        _fake_ort_dev("NPU", 0x4D4F, 0x0001),
        _fake_ort_dev("GPU", 0x4D4F, 0x0002),
        _fake_ort_dev("CPU", 0x4D4F, 0x0003),
    ]
    with patch("winml.modelkit.session.ep_device.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = devices
        result = resolve_device("qnn", "npu")
    assert result.ep == "QNNExecutionProvider"
    assert result.device == "npu"
    assert result.vendor_id == 0x4D4F
    assert result.device_id == 0x0001
    assert result.vendor == "Qualcomm"


def test_resolve_device_dedup_qnn_gpu() -> None:
    """Two OrtEpDevices with identical (vendor_id, device_id) collapse to one."""
    devices = [
        _fake_ort_dev("GPU", 0x4D4F, 0x0002),
        _fake_ort_dev("GPU", 0x4D4F, 0x0002),
    ]
    with patch("winml.modelkit.session.ep_device.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = devices
        result = resolve_device("qnn", "gpu")
    assert result.device == "gpu"
    assert result.device_id == 0x0002


def test_resolve_device_device_not_found_raises() -> None:
    """DeviceNotFound is raised when no OrtEpDevice matches the requested type."""
    devices = [_fake_ort_dev("NPU", 0x4D4F, 0x0001)]
    with patch("winml.modelkit.session.ep_device.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = devices
        with pytest.raises(DeviceNotFound):
            resolve_device("qnn", "gpu")


def test_resolve_device_ambiguous_raises() -> None:
    """Two distinct GPU entries (different device_id) cannot be auto-resolved."""
    devices = [
        _fake_ort_dev("GPU", 0x4D4F, 0x0002),
        _fake_ort_dev("GPU", 0x4D4F, 0x0003),
    ]
    with patch("winml.modelkit.session.ep_device.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = devices
        with pytest.raises(AmbiguousMatch):
            resolve_device("qnn", "gpu")


# --- short_ep_name tests ---------------------------------------------------


def test_short_ep_name_known_canonical() -> None:
    """Canonical EP names map back to their short forms."""
    assert short_ep_name("QNNExecutionProvider") == "qnn"
    assert short_ep_name("OpenVINOExecutionProvider") == "openvino"
    assert short_ep_name("DmlExecutionProvider") == "dml"
    assert short_ep_name("CPUExecutionProvider") == "cpu"


def test_short_ep_name_unknown_falls_back() -> None:
    """Unknown canonical names fall back to a stripped lowercase form."""
    assert short_ep_name("SomeFutureExecutionProvider") == "somefuture"
    assert short_ep_name("AlreadyShort") == "alreadyshort"


def test_short_ep_name_round_trip_with_expand() -> None:
    """short_ep_name and expand_ep_name are inverses for canonical names."""
    for short in ("qnn", "openvino", "vitisai", "dml", "cpu"):
        assert short_ep_name(expand_ep_name(short)) == short
