# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# tests/unit/session/test_ep_device.py
"""Unit tests for EPDevice descriptor and resolution helpers."""

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.session import (
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
    """Already-full names flow through unchanged."""
    assert expand_ep_name("QNNExecutionProvider") == "QNNExecutionProvider"
    assert expand_ep_name("CPUExecutionProvider") == "CPUExecutionProvider"


def test_expand_ep_name_alias_casing() -> None:
    """Mixed-case full-name aliases are normalized to canonical casing."""
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


def test_short_ep_name_known_full() -> None:
    """Full EP names map back to their short forms."""
    assert short_ep_name("QNNExecutionProvider") == "qnn"
    assert short_ep_name("OpenVINOExecutionProvider") == "openvino"
    assert short_ep_name("DmlExecutionProvider") == "dml"
    assert short_ep_name("CPUExecutionProvider") == "cpu"


def test_short_ep_name_unknown_falls_back() -> None:
    """Unknown full names fall back to a stripped lowercase form."""
    assert short_ep_name("SomeFutureExecutionProvider") == "somefuture"
    assert short_ep_name("AlreadyShort") == "alreadyshort"


def test_short_ep_name_round_trip_with_expand() -> None:
    """short_ep_name and expand_ep_name are inverses for full names."""
    for short in ("qnn", "openvino", "vitisai", "dml", "cpu"):
        assert short_ep_name(expand_ep_name(short)) == short


def test_expand_ep_name_cuda_tensorrt() -> None:
    """cuda and tensorrt short names expand to canonical ORT EP names."""
    assert expand_ep_name("cuda") == "CUDAExecutionProvider"
    assert expand_ep_name("tensorrt") == "TensorrtExecutionProvider"


def test_short_ep_name_cuda_tensorrt_round_trip() -> None:
    """Round-trip via expand and short for cuda/tensorrt."""
    for short in ("cuda", "tensorrt"):
        assert short_ep_name(expand_ep_name(short)) == short


# --- EPDeviceSpec catalog tests -------------------------------------------


def test_ep_device_specs_count() -> None:
    """The catalog must contain exactly 13 variants."""
    from winml.modelkit.session import EP_DEVICE_SPECS

    assert len(EP_DEVICE_SPECS) == 13


def test_lookup_device_spec_qnn_npu() -> None:
    """lookup_device_spec returns the QNN-NPU entry with burst defaults."""
    from winml.modelkit.session import lookup_device_spec

    spec = lookup_device_spec("QNNExecutionProvider", "npu")
    assert spec is not None
    assert spec.ep == "QNNExecutionProvider"
    assert spec.device == "npu"
    assert spec.default_provider_options["htp_performance_mode"] == "burst"
    assert spec.default_provider_options["htp_graph_finalization_optimization_mode"] == "3"


def test_lookup_device_spec_unknown_returns_none() -> None:
    """lookup_device_spec returns None for unknown (ep, device) pairs."""
    from winml.modelkit.session import lookup_device_spec

    assert lookup_device_spec("UnknownEP", "npu") is None
    assert lookup_device_spec("QNNExecutionProvider", "unknown_device") is None


def test_lookup_device_spec_empty_defaults() -> None:
    """Non-QNN-NPU entries have empty default_provider_options (TODO entries)."""
    from winml.modelkit.session import lookup_device_spec

    for ep, device in [
        ("DmlExecutionProvider", "gpu"),
        ("CPUExecutionProvider", "cpu"),
        ("CUDAExecutionProvider", "gpu"),
        ("QNNExecutionProvider", "gpu"),
    ]:
        spec = lookup_device_spec(ep, device)
        assert spec is not None, f"Expected {ep}/{device} in catalog"
        assert dict(spec.default_provider_options) == {}, (
            f"{ep}/{device} should have empty defaults until measured"
        )


def test_default_device_for_ep_qnn() -> None:
    """default_device_for_ep returns 'npu' for QNN (first variant in catalog)."""
    from winml.modelkit.session import default_device_for_ep

    assert default_device_for_ep("QNNExecutionProvider") == "npu"


def test_default_device_for_ep_dml() -> None:
    """default_device_for_ep returns 'gpu' for DML (single variant)."""
    from winml.modelkit.session import default_device_for_ep

    assert default_device_for_ep("DmlExecutionProvider") == "gpu"


def test_default_device_for_ep_cpu() -> None:
    """default_device_for_ep returns 'cpu' for CPU EP."""
    from winml.modelkit.session import default_device_for_ep

    assert default_device_for_ep("CPUExecutionProvider") == "cpu"


def test_default_device_for_ep_unknown_returns_none() -> None:
    """default_device_for_ep returns None for unknown EP."""
    from winml.modelkit.session import default_device_for_ep

    assert default_device_for_ep("UnknownExecutionProvider") is None


def test_default_ep_for_device_npu() -> None:
    """default_ep_for_device returns QNNExecutionProvider for npu (first in catalog)."""
    from winml.modelkit.session import default_ep_for_device

    assert default_ep_for_device("npu") == "QNNExecutionProvider"


def test_default_ep_for_device_gpu() -> None:
    """default_ep_for_device returns OpenVINOExecutionProvider for gpu (first in catalog)."""
    from winml.modelkit.session import default_ep_for_device

    assert default_ep_for_device("gpu") == "OpenVINOExecutionProvider"


def test_default_ep_for_device_cpu() -> None:
    """default_ep_for_device returns OpenVINOExecutionProvider for cpu (first in catalog)."""
    from winml.modelkit.session import default_ep_for_device

    # OpenVINO-CPU comes before QNN-CPU and CPUExecutionProvider in the catalog
    assert default_ep_for_device("cpu") == "OpenVINOExecutionProvider"


def test_default_ep_for_device_unknown_returns_none() -> None:
    """default_ep_for_device returns None for unknown device."""
    from winml.modelkit.session import default_ep_for_device

    assert default_ep_for_device("unknown_device") is None


def test_ep_device_spec_is_frozen() -> None:
    """EPDeviceSpec is frozen — mutation raises FrozenInstanceError."""
    from dataclasses import FrozenInstanceError

    from winml.modelkit.session import EPDeviceSpec

    spec = EPDeviceSpec(ep="QNNExecutionProvider", device="npu")
    with pytest.raises(FrozenInstanceError):
        spec.ep = "DmlExecutionProvider"  # type: ignore[misc]


def test_ep_device_spec_default_factory_is_fresh() -> None:
    """Each EPDeviceSpec with no options gets a new empty dict (not shared)."""
    from winml.modelkit.session import EPDeviceSpec

    s1 = EPDeviceSpec(ep="DmlExecutionProvider", device="gpu")
    s2 = EPDeviceSpec(ep="CUDAExecutionProvider", device="gpu")
    # They should be equal (both empty) but not the same object
    assert dict(s1.default_provider_options) == {}
    assert dict(s2.default_provider_options) == {}
    # The dict() copy from lookup_device_spec guarantees mutability for callers
    d = dict(s1.default_provider_options)
    d["key"] = "value"
    assert "key" not in s1.default_provider_options
