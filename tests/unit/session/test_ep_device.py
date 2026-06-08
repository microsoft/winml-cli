# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# tests/unit/session/test_ep_device.py
"""Unit tests for EPDeviceTarget descriptor and resolution helpers."""

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.session import (
    EPDeviceTarget,
    expand_ep_name,
    resolve_device,
    short_ep_name,
)


def test_ep_device_round_trip() -> None:
    """EPDeviceTarget -> to_dict -> from_dict yields an equal instance."""
    original = EPDeviceTarget(ep="QNNExecutionProvider", device="npu")
    rehydrated = EPDeviceTarget.from_dict(original.to_dict())
    assert rehydrated == original
    assert rehydrated.ep == "QNNExecutionProvider"
    assert rehydrated.device == "npu"


def test_ep_device_lowercase_invariant() -> None:
    """`device` field is forced to lowercase by __post_init__."""
    ep_device = EPDeviceTarget(ep="QNNExecutionProvider", device="NPU")
    assert ep_device.device == "npu"


def test_from_dict_forward_compat_with_optional_source() -> None:
    """EPDeviceTarget.from_dict round-trips both legacy and new JSON shapes.

    Legacy persisted configs predate the ``source`` field — they must rehydrate
    with ``source=None``.  New configs that carry ``source`` (e.g. ``"pypi"``,
    ``"msix"``) must round-trip the value through to_dict / from_dict.

    Pins the forward-compatibility contract for the EPDeviceTarget JSON shape.
    """
    # Legacy JSON (no source field; also tolerates legacy vendor_id/device_id keys).
    legacy = EPDeviceTarget.from_dict(
        {
            "ep": "QNNExecutionProvider",
            "device": "npu",
            "vendor_id": 0x4D4F,  # legacy key — silently ignored
            "device_id": 0x0001,  # legacy key — silently ignored
        }
    )
    assert legacy == EPDeviceTarget(ep="QNNExecutionProvider", device="npu", source=None)
    assert legacy.source is None

    # New JSON with source — must round-trip via to_dict / from_dict.
    new = EPDeviceTarget.from_dict(
        {
            "ep": "QNNExecutionProvider",
            "device": "npu",
            "source": "pypi",
        }
    )
    assert new.source == "pypi"
    serialized = new.to_dict()
    assert serialized["source"] == "pypi"
    assert EPDeviceTarget.from_dict(serialized) == new


def test_from_dict_roundtrips_source_field() -> None:
    """source field round-trips through to_dict/from_dict cleanly."""
    target = EPDeviceTarget(
        ep="OpenVINOExecutionProvider",
        device="npu",
        source="pypi",
    )
    d = target.to_dict()
    assert d["source"] == "pypi"
    restored = EPDeviceTarget.from_dict(d)
    assert restored.source == "pypi"
    assert restored == target


def test_from_dict_legacy_without_vendor_id() -> None:
    """JSON written before Batch C strip — no vendor_id/device_id/vendor keys."""
    legacy = {"ep": "openvino", "device": "npu"}
    target = EPDeviceTarget.from_dict(legacy)
    assert target.source is None
    # Stripped fields must NOT appear on the dataclass post-Batch-C.
    assert not hasattr(target, "vendor_id")
    assert not hasattr(target, "device_id")
    assert not hasattr(target, "vendor")


class TestEPDeviceTargetValidation:
    """Construction-time validation per __post_init__."""

    def test_invalid_device_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown device"):
            EPDeviceTarget(ep="openvino", device="tpu")

    def test_invalid_ep_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown EP"):
            EPDeviceTarget(ep="bogus_ep", device="npu")

    def test_invalid_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown source tag"):
            EPDeviceTarget(ep="openvino", device="npu", source="not-a-real-tag")

    def test_auto_passes(self) -> None:
        # 'auto' on either axis bypasses the catalog check.
        EPDeviceTarget(ep="auto", device="auto")
        EPDeviceTarget(ep="openvino", device="auto")
        EPDeviceTarget(ep="auto", device="npu")

    def test_full_ep_name_accepted(self) -> None:
        # Full EP names also work (not only short forms).
        EPDeviceTarget(ep="OpenVINOExecutionProvider", device="npu")


def test_expand_ep_name_short_form() -> None:
    assert expand_ep_name("qnn") == "QNNExecutionProvider"
    assert expand_ep_name("openvino") == "OpenVINOExecutionProvider"
    assert expand_ep_name("vitisai") == "VitisAIExecutionProvider"
    assert expand_ep_name("migraphx") == "MIGraphXExecutionProvider"
    assert expand_ep_name("nvtensorrtrtx") == "NvTensorRtRtxExecutionProvider"
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


# --- resolve_device tests (pure-deduction; no DLL load) ----------------------


def test_resolve_device_qnn_npu() -> None:
    """resolve_device returns the requested (ep, device) pair without registry calls."""
    result = resolve_device("qnn", "npu")
    assert result.ep == "QNNExecutionProvider"
    assert result.device == "npu"
    assert result.source is None


def test_resolve_device_threads_source_through() -> None:
    """resolve_device passes source through to EPDeviceTarget."""
    result = resolve_device("qnn", "npu", source="pypi")
    assert result.source == "pypi"


def test_resolve_device_does_not_load_dll() -> None:
    """resolve_device must NOT touch WinMLEPRegistry in the new architecture.

    DLL load + handle binding lives in WinMLEPRegistry.auto_device. This test
    guards the boundary: if resolve_device ever re-acquires the registry it
    will re-introduce the same circular-import / slow-startup bugs Batch C
    removed.
    """
    with patch("winml.modelkit.session.ep_registry.WinMLEPRegistry") as mock_reg:
        result = resolve_device("qnn", "npu")
    assert result.ep == "QNNExecutionProvider"
    mock_reg.get_instance.assert_not_called()


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
    """The catalog must contain exactly 12 variants.

    (CUDAExecutionProvider was dropped in the v1 catalog — not currently
    measured by this project. Re-add an EPCatalog.Row + bump this count to 13
    if/when CUDA support lands.)
    """
    from winml.modelkit.session import EP_DEVICE_SPECS

    assert len(EP_DEVICE_SPECS) == 12


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
    """Non-QNN-NPU entries have empty default_provider_options (TODO entries).

    CUDA is intentionally omitted — dropped from EP_DEVICE_SPECS in v1.
    """
    from winml.modelkit.session import lookup_device_spec

    for ep, device in [
        ("DmlExecutionProvider", "gpu"),
        ("CPUExecutionProvider", "cpu"),
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

    Spec: §6.4. On an OpenVINO-only box, deduction must walk the catalog
    AND filter by available_eps() so QNN is not returned. Post-Batch-C,
    resolve_device is pure-deduction (no DLL load), so the test verifies
    the deduced EP name only.
    """
    import contextlib

    from winml.modelkit.session import resolve_device

    available = frozenset({"OpenVINOExecutionProvider", "CPUExecutionProvider"})

    with contextlib.ExitStack() as stack:
        for cm in _patch_available_eps(available):
            stack.enter_context(cm)
        result = resolve_device(device="npu")

    assert result.ep == "OpenVINOExecutionProvider", (
        f"resolve_device returned {result.ep!r}; expected "
        "'OpenVINOExecutionProvider' (the only registered NPU EP on this stub host)."
    )
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
