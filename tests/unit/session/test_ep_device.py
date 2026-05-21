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

from .conftest import QNN_VENDOR_ID


def test_ep_device_round_trip() -> None:
    """EPDevice -> to_dict -> from_dict yields an equal instance."""
    original = EPDevice(
        ep="QNNExecutionProvider",
        device="npu",
        vendor_id=QNN_VENDOR_ID,
        device_id=0x0001,
        vendor="Qualcomm",
    )
    rehydrated = EPDevice.from_dict(original.to_dict())
    assert rehydrated == original
    assert rehydrated.ep == "QNNExecutionProvider"
    assert rehydrated.device == "npu"
    assert rehydrated.vendor_id == QNN_VENDOR_ID
    assert rehydrated.device_id == 0x0001
    assert rehydrated.vendor == "Qualcomm"


def test_ep_device_lowercase_invariant() -> None:
    """`device` field is forced to lowercase by __post_init__."""
    ep_device = EPDevice(
        ep="QNNExecutionProvider",
        device="NPU",
        vendor_id=QNN_VENDOR_ID,
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


# --- short<->full loopback (catalog invariant) ------------------------------


def _short_to_full_items() -> list[tuple[str, str]]:
    """Read _SHORT_TO_FULL directly so this test catches every entry."""
    from winml.modelkit.session.ep_device import _SHORT_TO_FULL

    return list(_SHORT_TO_FULL.items())


@pytest.mark.parametrize("short, full", _short_to_full_items())
def test_short_full_loopback(short: str, full: str) -> None:
    """Every (short, full) catalog entry round-trips both directions.

    Forward:   expand_ep_name(short) == full.
    Reverse:   short_ep_name(full) gives back a short that expands to the
               same full. (Allows for many-to-one mapping if multiple
               shorts ever alias the same full — the canonical winner
               just has to be ONE of them.)

    This invariant pins the 1:1 (or many:1) shape of _SHORT_TO_FULL /
    _FULL_TO_SHORT. Catalog drift (e.g. typoing a value, adding an alias
    that breaks the inverse) fails this test loudly.
    """
    # Forward
    assert expand_ep_name(short) == full, (
        f"expand_ep_name({short!r}) returned {expand_ep_name(short)!r}, expected {full!r}"
    )
    # Reverse: short_ep_name returns SOME valid short for this full.
    canonical_short = short_ep_name(full)
    assert expand_ep_name(canonical_short) == full, (
        f"short_ep_name({full!r}) returned {canonical_short!r}, but "
        f"expand_ep_name({canonical_short!r}) = "
        f"{expand_ep_name(canonical_short)!r} != {full!r}"
    )


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
        _fake_ort_dev("NPU", QNN_VENDOR_ID, 0x0001),
        _fake_ort_dev("GPU", QNN_VENDOR_ID, 0x0002),
        _fake_ort_dev("CPU", QNN_VENDOR_ID, 0x0003),
    ]
    with patch("winml.modelkit.session.ep_device.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = devices
        result = resolve_device("qnn", "npu")
    assert result.ep == "QNNExecutionProvider"
    assert result.device == "npu"
    assert result.vendor_id == QNN_VENDOR_ID
    assert result.device_id == 0x0001
    assert result.vendor == "Qualcomm"


def test_resolve_device_dedup_qnn_gpu() -> None:
    """Two OrtEpDevices with identical (vendor_id, device_id) collapse to one."""
    devices = [
        _fake_ort_dev("GPU", QNN_VENDOR_ID, 0x0002),
        _fake_ort_dev("GPU", QNN_VENDOR_ID, 0x0002),
    ]
    with patch("winml.modelkit.session.ep_device.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = devices
        result = resolve_device("qnn", "gpu")
    assert result.device == "gpu"
    assert result.device_id == 0x0002


def test_resolve_device_device_not_found_raises() -> None:
    """DeviceNotFound is raised when no OrtEpDevice matches the requested type."""
    devices = [_fake_ort_dev("NPU", QNN_VENDOR_ID, 0x0001)]
    with patch("winml.modelkit.session.ep_device.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = devices
        with pytest.raises(DeviceNotFound):
            resolve_device("qnn", "gpu")


def test_resolve_device_ambiguous_raises() -> None:
    """Two distinct GPU entries (different device_id) cannot be auto-resolved."""
    devices = [
        _fake_ort_dev("GPU", QNN_VENDOR_ID, 0x0002),
        _fake_ort_dev("GPU", QNN_VENDOR_ID, 0x0003),
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
    """default_ep_for_device returns DmlExecutionProvider for gpu.

    The EP_DEVICE_SPECS catalog comment claims "gpu-first: DmlExecutionProvider"
    (ep_device.py:154). This test pins that invariant. A failure here means
    either the catalog ordering drifted (DML/gpu was moved off the primary
    position) or `default_ep_for_device` is walking entries it shouldn't.

    Related: docs/design/session/3_design_ep.md §6.4 — the same comment also
    promises QNN's gpu/cpu *secondary* entries must not shadow the primary
    GPU/CPU EPs in deduction.
    """
    from winml.modelkit.session import default_ep_for_device

    assert default_ep_for_device("gpu") == "DmlExecutionProvider"


def test_default_ep_for_device_cpu() -> None:
    """default_ep_for_device returns CPUExecutionProvider for cpu.

    Catalog comment (ep_device.py:155) promises "cpu-first: CPUExecutionProvider".
    A failure means QNN's secondary cpu row shadowed the primary CPU EP.
    See test_default_ep_for_device_gpu for the related GPU case.
    """
    from winml.modelkit.session import default_ep_for_device

    assert default_ep_for_device("cpu") == "CPUExecutionProvider"


def test_ep_device_specs_primary_per_device_invariant() -> None:
    """EP_DEVICE_SPECS must order documented primaries before secondary EPs.

    The catalog's own comment (ep_device.py:152-157) declares:
        npu-first: QNNExecutionProvider
        gpu-first: DmlExecutionProvider
        cpu-first: CPUExecutionProvider

    For each device, the FIRST catalog entry for that device must match the
    documented primary. A failure means the comment and the code disagree —
    same root cause as the registration-aware deduction gap in §6.4: callers
    that walk EP_DEVICE_SPECS in order get an answer the docstring did not
    promise them.
    """
    from winml.modelkit.session import EP_DEVICE_SPECS

    expected_primary = {
        "npu": "QNNExecutionProvider",
        "gpu": "DmlExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    first_per_device: dict[str, str] = {}
    for spec in EP_DEVICE_SPECS:
        first_per_device.setdefault(spec.device, spec.ep)

    for device, expected_ep in expected_primary.items():
        assert first_per_device.get(device) == expected_ep, (
            f"EP_DEVICE_SPECS first entry for {device!r} is "
            f"{first_per_device.get(device)!r}, but the catalog comment "
            f"declares the primary is {expected_ep!r}. Either move the "
            "primary to positions 0-2 or update the docstring/comments."
        )


def test_default_ep_for_device_unknown_returns_none() -> None:
    """default_ep_for_device returns None for unknown device."""
    from winml.modelkit.session import default_ep_for_device

    assert default_ep_for_device("unknown_device") is None


# --- registration-aware deduction (spec §6.4 in docs/design/session/3_design_ep.md) -----


def _patch_available_eps(eps: frozenset[str]) -> tuple:
    """Patch every plausible binding of `available_eps` so that the fix is free
    to import it from `ep_registry` directly or re-bind it into `ep_device`.

    Returns a tuple of context managers — use ``with contextlib.ExitStack()``
    or nest them. ``create=True`` covers the case where the fix has not yet
    introduced the import at the ep_device module level.
    """
    mock = MagicMock(return_value=eps)
    return (
        patch("winml.modelkit.session.ep_registry.available_eps", mock),
        patch("winml.modelkit.session.ep_device.available_eps", mock, create=True),
    )


def test_default_ep_for_device_filters_by_available_eps_npu() -> None:
    """On an OpenVINO-only box, default_ep_for_device('npu') must NOT return QNN.

    Spec: docs/design/session/3_design_ep.md §6.4 — registration-aware deduction.

    The static catalog (EP_DEVICE_SPECS) orders QNNExecutionProvider first
    for 'npu', but QNN is not registered on this host. A correct deduction
    walks the catalog and returns the first EP that is also in
    `available_eps()` — here, OpenVINOExecutionProvider.
    """
    import contextlib

    from winml.modelkit.session import default_ep_for_device

    available = frozenset({"OpenVINOExecutionProvider", "CPUExecutionProvider"})
    with contextlib.ExitStack() as stack:
        for cm in _patch_available_eps(available):
            stack.enter_context(cm)
        result = default_ep_for_device("npu")

    assert result == "OpenVINOExecutionProvider", (
        f"Expected OpenVINOExecutionProvider (only registered NPU EP on this host) "
        f"but got {result!r}. default_ep_for_device must filter by available_eps()."
    )


def test_default_ep_for_device_filters_by_available_eps_gpu() -> None:
    """On a MIGraphX-only host (no QNN, no DML), default_ep_for_device('gpu') must
    return MIGraphXExecutionProvider — the first registered EP in the catalog for gpu.
    """
    import contextlib

    from winml.modelkit.session import default_ep_for_device

    available = frozenset({"MIGraphXExecutionProvider", "CPUExecutionProvider"})
    with contextlib.ExitStack() as stack:
        for cm in _patch_available_eps(available):
            stack.enter_context(cm)
        result = default_ep_for_device("gpu")

    assert result == "MIGraphXExecutionProvider", (
        f"Expected MIGraphXExecutionProvider (only registered GPU EP) but got {result!r}. "
        "default_ep_for_device must filter by available_eps()."
    )


def test_default_ep_for_device_returns_none_when_no_registered_ep_for_device() -> None:
    """When no registered EP exists for the requested device, return None.

    The contract per §6.4: caller decides what to do (raise, fall back to CPU).
    The helper must not return an unregistered EP just because the catalog has one.
    """
    import contextlib

    from winml.modelkit.session import default_ep_for_device

    # CPU-only host — no NPU EP is registered.
    available = frozenset({"CPUExecutionProvider"})
    with contextlib.ExitStack() as stack:
        for cm in _patch_available_eps(available):
            stack.enter_context(cm)
        result = default_ep_for_device("npu")

    assert result is None, (
        f"Expected None (no NPU EP registered) but got {result!r}. "
        "default_ep_for_device must not return unregistered EPs."
    )


def test_resolve_device_device_only_picks_registered_ep() -> None:
    """resolve_device(device='npu') with no ep must pick a REGISTERED EP.

    Spec: §6.4. Today the device-only branch (ep_device.py:379) calls
    default_ep_for_device which returns QNN unconditionally — so on an
    OpenVINO-only box this attempts to register QNN, then fails.

    Correct behavior: deduce OpenVINOExecutionProvider, register it,
    resolve to the OpenVINO NPU OrtEpDevice.
    """
    import contextlib

    from winml.modelkit.session import resolve_device

    available = frozenset({"OpenVINOExecutionProvider", "CPUExecutionProvider"})
    intel_vendor_id = 0x8086
    openvino_npu_device_id = 0x643E

    ov_npu = _fake_ort_dev("NPU", intel_vendor_id, openvino_npu_device_id)
    ov_npu.device.vendor = "Intel Corporation"

    with contextlib.ExitStack() as stack:
        for cm in _patch_available_eps(available):
            stack.enter_context(cm)
        mock_reg = stack.enter_context(
            patch("winml.modelkit.session.ep_device.WinMLEPRegistry")
        )
        mock_reg.get_instance.return_value.register_ep.return_value = [ov_npu]

        result = resolve_device(device="npu")

        # The fix MUST request OpenVINOExecutionProvider from the registry,
        # not QNNExecutionProvider — otherwise it's still walking the static
        # catalog first.
        registered = mock_reg.get_instance.return_value.register_ep.call_args
        assert registered is not None, "register_ep was never called"
        assert registered.args[0] == "OpenVINOExecutionProvider", (
            f"register_ep was called with {registered.args[0]!r}; "
            "expected 'OpenVINOExecutionProvider' (the only registered NPU EP). "
            "Static-catalog deduction returned an unregistered EP."
        )

    assert result.ep == "OpenVINOExecutionProvider"
    assert result.device == "npu"


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
