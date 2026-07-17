# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# tests/unit/session/test_ep_device.py
"""Unit tests for EPDeviceTarget descriptor and resolution helpers."""

from unittest.mock import patch

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
    from winml.modelkit.session import _SHORT_TO_FULL

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
    result = resolve_device(EPDeviceTarget(ep="qnn", device="npu"))
    assert result.ep == "QNNExecutionProvider"
    assert result.device == "npu"
    assert result.source is None


def test_resolve_device_passes_source_through_unchanged() -> None:
    """resolve_device is pure deduction — source passes through verbatim.

    Source-tag validation lives in WinMLEPRegistry.auto_device (Path A
    registration step), not here. resolve_device must do no filesystem
    scan, no DLL load, and no registry I/O on the source axis.
    """
    result = resolve_device(EPDeviceTarget(ep="qnn", device="npu", source="pypi"))
    assert result.ep == "QNNExecutionProvider"
    assert result.device == "npu"
    assert result.source == "pypi"


def test_resolve_device_does_not_load_dll() -> None:
    """resolve_device must NOT touch WinMLEPRegistry in the new architecture.

    DLL load + handle binding lives in WinMLEPRegistry.auto_device. This test
    guards the boundary: if resolve_device ever re-acquires the registry it
    will re-introduce the same circular-import / slow-startup bugs Batch C
    removed.
    """
    with patch("winml.modelkit.session.ep_registry.WinMLEPRegistry") as mock_reg:
        result = resolve_device(EPDeviceTarget(ep="qnn", device="npu"))
    assert result.ep == "QNNExecutionProvider"
    mock_reg.instance.assert_not_called()


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


def test_expand_ep_name_tensorrt() -> None:
    """``tensorrt`` short name expands to ``TensorrtExecutionProvider``.

    Post-T-15: ``cuda`` is no longer a recognized alias (the catalog has
    no ``CUDAExecutionProvider`` row; ``cuda`` was dropped from
    ``_SHORT_TO_FULL`` so ``EPDeviceTarget(ep="cuda", ...)`` is rejected
    at construction time, matching the catalog's authoritative scope).
    """
    assert expand_ep_name("tensorrt") == "TensorrtExecutionProvider"


def test_short_ep_name_tensorrt_round_trip() -> None:
    """Round-trip via expand and short for ``tensorrt``."""
    assert short_ep_name(expand_ep_name("tensorrt")) == "tensorrt"


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


def test_default_ep_for_device_gpu_prefers_plugin_over_builtin() -> None:
    """default_ep_for_device returns OpenVINO GPU (first plugin) for gpu.

    Built-in EPs (DML) are ordered LAST in EP_DEVICE_SPECS per the design
    intent stated in ``ep_registry.py:302-306`` ("built-ins are lowest
    priority — only used when no plugin provided the EP"). When OpenVINO
    is available, it wins over DML.

    A failure here means either the catalog ordering drifted or
    ``default_ep_for_device`` is skipping plugin entries it shouldn't.
    """
    from winml.modelkit.session import default_ep_for_device

    assert default_ep_for_device("gpu") == "OpenVINOExecutionProvider"


def test_default_ep_for_device_gpu_falls_back_to_dml(monkeypatch) -> None:
    """DML wins for gpu when no plugin GPU EP is registered — fallback path."""
    from winml.modelkit.session import default_ep_for_device
    from winml.modelkit.session.ep_registry import WinMLEPRegistry

    monkeypatch.setattr(
        WinMLEPRegistry,
        "available_eps",
        lambda self: frozenset({"DmlExecutionProvider", "CPUExecutionProvider"}),
    )
    assert default_ep_for_device("gpu") == "DmlExecutionProvider"


def test_default_ep_for_device_cpu_prefers_plugin_over_builtin() -> None:
    """default_ep_for_device returns OpenVINO CPU (first plugin) for cpu.

    Same design intent as the gpu case: built-in CPU EP is the fallback.
    """
    from winml.modelkit.session import default_ep_for_device

    assert default_ep_for_device("cpu") == "OpenVINOExecutionProvider"


def test_default_ep_for_device_cpu_falls_back_to_cpu_ep(monkeypatch) -> None:
    """Built-in CPU EP wins for cpu when no plugin CPU EP is registered."""
    from winml.modelkit.session import default_ep_for_device
    from winml.modelkit.session.ep_registry import WinMLEPRegistry

    monkeypatch.setattr(
        WinMLEPRegistry,
        "available_eps",
        lambda self: frozenset({"DmlExecutionProvider", "CPUExecutionProvider"}),
    )
    assert default_ep_for_device("cpu") == "CPUExecutionProvider"


def test_ep_device_specs_first_plugin_per_device_invariant() -> None:
    """EP_DEVICE_SPECS must order vendor-optimal plugin EPs before built-ins.

    Per the design intent stated in ``ep_registry.py:302-306`` — "built-ins
    are lowest priority — only used when no plugin provided the EP" — the
    catalog's first entry for each device must be a plugin (vendor-specific)
    EP, not a built-in fallback. Built-ins (DML/CPU/Azure) trail as
    fallbacks.

    Expected first-plugin per device (from EP_DEVICE_SPECS ordering):
        npu: QNNExecutionProvider          (Snapdragon HTP)
        gpu: OpenVINOExecutionProvider     (Intel GPU / Intel Arc)
        cpu: OpenVINOExecutionProvider     (Intel CPU-optimized runtime)
    """
    from winml.modelkit.session import EP_DEVICE_SPECS

    expected_first = {
        "npu": "QNNExecutionProvider",
        "gpu": "OpenVINOExecutionProvider",
        "cpu": "OpenVINOExecutionProvider",
    }
    first_per_device: dict[str, str] = {}
    for spec in EP_DEVICE_SPECS:
        first_per_device.setdefault(spec.device, spec.ep)

    for device, expected_ep in expected_first.items():
        assert first_per_device.get(device) == expected_ep, (
            f"EP_DEVICE_SPECS first entry for {device!r} is "
            f"{first_per_device.get(device)!r}; expected {expected_ep!r} "
            "per the built-ins-as-fallback design intent. Either move the "
            "plugin EP to the head of its device group or update this test."
        )


def test_ep_device_specs_builtins_come_last() -> None:
    """Built-in EPs (DML/CPU/Azure) must appear AFTER every plugin entry.

    Enforces the trailing-fallback position that matches
    ``ep_registry.py:302-306``'s design intent.
    """
    from winml.modelkit.session import EP_DEVICE_SPECS

    builtin_names = frozenset(
        {
            "DmlExecutionProvider",
            "CPUExecutionProvider",
            "AzureExecutionProvider",
        }
    )
    seen_builtin = False
    for spec in EP_DEVICE_SPECS:
        is_builtin = spec.ep in builtin_names
        if is_builtin:
            seen_builtin = True
        else:
            assert not seen_builtin, (
                f"Plugin EP {spec.ep!r}/{spec.device!r} appears AFTER a "
                "built-in — built-ins must come last per the design intent."
            )


def test_default_ep_for_device_unknown_returns_none() -> None:
    """default_ep_for_device returns None for unknown device."""
    from winml.modelkit.session import default_ep_for_device

    assert default_ep_for_device("unknown_device") is None


# --- registration-aware deduction (spec §6.4 in docs/design/session/3_design_ep.md) -----


def _patch_available_eps(eps: frozenset[str]) -> tuple:
    """Patch ``WinMLEPRegistry.available_eps`` at the class level.

    After the v2.8 refactor, ``available_eps`` is a method on the singleton
    registry rather than a module-level free function. Class-level patching
    flows through the singleton's bound-method lookup, so any callsite
    (``default_ep_for_device``, ``auto_detect_device``) sees the stub.
    """
    from winml.modelkit.session.ep_registry import WinMLEPRegistry

    return (
        patch.object(
            WinMLEPRegistry,
            "available_eps",
            return_value=eps,
        ),
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
        result = resolve_device(EPDeviceTarget(ep="auto", device="npu"))

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


# ---------------------------------------------------------------------------
# L2 vendor-compatibility filter in EP deduction (Bug #1 — RED tests).
#
# The bug: on an Intel host, QNNExecutionProvider registers successfully
# with a CPU fallback device. It ends up in available_eps() (which only
# checks L1 = "discovered + registerable"), and default_ep_for_device("npu")
# then picks QNN — the catalog primary — even though QNN's vendor
# requirement (Qualcomm) is not satisfied on this host. The user-visible
# effect: `winml perf` with no --ep / --device on an Intel box would route
# to QNN-on-CPU-fallback instead of OpenVINO-on-NPU.
#
# These tests pin the desired behavior: EP deduction must skip rows whose
# EP_CATALOG.is_compatible(ep_name) returns False, regardless of where the
# fix lands (available_eps, default_ep_for_device, resolve_device, etc.).
# All five tests are RED today.
# ---------------------------------------------------------------------------


def _patch_ep_catalog_compat(compatible_map: dict[str, bool]) -> tuple:
    """Patch EPCatalog.is_compatible at the CLASS level.

    The catalog instance (``EP_CATALOG``) overrides ``__setattr__`` to enforce
    immutability — patching the bound attribute fails. Patching the underlying
    method on the class works because the bound-method lookup hits the patched
    class slot. Unknown EPs default to True (forward-compat — matches the
    catalog's behavior for EPs without vendor_requirements).
    """

    def fake_is_compatible(self, ep_name: str) -> bool:
        return compatible_map.get(ep_name, True)

    return (
        patch(
            "winml.modelkit.ep_path.EPCatalog.is_compatible",
            fake_is_compatible,
        ),
    )


def test_default_ep_for_device_skips_l2_incompatible_for_npu() -> None:
    """RED — default_ep_for_device('npu') must skip QNN when L2-incompatible.

    Simulates an Intel host: QNN is discovered + registered (L1 ok) but its
    vendor requirement (Qualcomm) isn't satisfied (L2 fail). The fix must
    walk past QNN and return the next L2-compatible NPU EP in
    EP_DEVICE_SPECS — here, OpenVINOExecutionProvider.
    """
    import contextlib

    from winml.modelkit.session import default_ep_for_device

    available = frozenset(
        {
            "QNNExecutionProvider",  # L1 ok (registered)
            "OpenVINOExecutionProvider",  # L1 ok
            "CPUExecutionProvider",
        }
    )
    compatibility = {
        "QNNExecutionProvider": False,  # L2 fail — wrong vendor
        "OpenVINOExecutionProvider": True,  # L2 ok
        "CPUExecutionProvider": True,
    }

    with contextlib.ExitStack() as stack:
        for cm in _patch_available_eps(available):
            stack.enter_context(cm)
        for cm in _patch_ep_catalog_compat(compatibility):
            stack.enter_context(cm)
        result = default_ep_for_device("npu")

    assert result == "OpenVINOExecutionProvider", (
        f"Expected OpenVINOExecutionProvider (next L2-compatible NPU EP after "
        f"QNN was filtered out for vendor mismatch), but got {result!r}. "
        f"default_ep_for_device must respect EP_CATALOG.is_compatible."
    )


def test_default_ep_for_device_skips_l2_incompatible_for_gpu() -> None:
    """RED — same skip-on-L2-fail behavior for the GPU axis.

    Simulates a non-Intel/non-Nvidia GPU box: DML and QNN are registered
    but vendor-incompatible; OpenVINO is L2-compatible. The next compatible
    GPU EP per EP_DEVICE_SPECS precedence (DML, QNN-secondary, OpenVINO,
    MIGraphX, Tensorrt, NvTensorRtRtx) is OpenVINO.
    """
    import contextlib

    from winml.modelkit.session import default_ep_for_device

    available = frozenset(
        {
            "DmlExecutionProvider",
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        }
    )
    compatibility = {
        "DmlExecutionProvider": False,
        "QNNExecutionProvider": False,
        "OpenVINOExecutionProvider": True,
        "CPUExecutionProvider": True,
    }

    with contextlib.ExitStack() as stack:
        for cm in _patch_available_eps(available):
            stack.enter_context(cm)
        for cm in _patch_ep_catalog_compat(compatibility):
            stack.enter_context(cm)
        result = default_ep_for_device("gpu")

    assert result == "OpenVINOExecutionProvider", (
        f"Expected OpenVINOExecutionProvider (next L2-compatible GPU EP), but got {result!r}."
    )


def test_default_ep_for_device_returns_none_when_all_npu_eps_l2_incompatible() -> None:
    """RED — when no NPU EP is BOTH registered AND L2-compatible, return None.

    Edge: every NPU-targeting EP in EP_DEVICE_SPECS (QNN, OpenVINO, VitisAI)
    is L1-registered but L2-incompatible. The function must NOT silently
    return one of them — None lets the caller fall back to CPU or raise
    cleanly, per its documented contract.
    """
    import contextlib

    from winml.modelkit.session import default_ep_for_device

    available = frozenset(
        {
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "VitisAIExecutionProvider",
            "CPUExecutionProvider",
        }
    )
    compatibility = {
        "QNNExecutionProvider": False,
        "OpenVINOExecutionProvider": False,
        "VitisAIExecutionProvider": False,
        "CPUExecutionProvider": True,
    }

    with contextlib.ExitStack() as stack:
        for cm in _patch_available_eps(available):
            stack.enter_context(cm)
        for cm in _patch_ep_catalog_compat(compatibility):
            stack.enter_context(cm)
        result = default_ep_for_device("npu")

    assert result is None, (
        f"Expected None (no L2-compatible NPU EP on this stub host), "
        f"but got {result!r}. Returning an incompatible EP silently is the bug."
    )


def test_default_ep_for_device_unchanged_when_all_l2_compatible() -> None:
    """Regression guard — when every registered EP is L2-compatible, the
    L2 filter is a no-op and the catalog primary still wins.

    Without this guard, a fix that always returns the second EP (or the last)
    would also satisfy the skip-tests above. This pins precedence semantics.
    """
    import contextlib

    from winml.modelkit.session import default_ep_for_device

    available = frozenset(
        {
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        }
    )
    compatibility = {
        "QNNExecutionProvider": True,
        "OpenVINOExecutionProvider": True,
        "CPUExecutionProvider": True,
    }

    with contextlib.ExitStack() as stack:
        for cm in _patch_available_eps(available):
            stack.enter_context(cm)
        for cm in _patch_ep_catalog_compat(compatibility):
            stack.enter_context(cm)
        result = default_ep_for_device("npu")

    assert result == "QNNExecutionProvider", (
        f"Expected QNNExecutionProvider (catalog primary for NPU when L2-ok), "
        f"but got {result!r}. The L2 filter must not change behavior for "
        f"compatible EPs."
    )


def test_resolve_device_both_auto_skips_l2_incompatible_full_chain() -> None:
    """RED — full chain: resolve_device(ep=auto, device=auto) on an Intel-like
    host must not return QNN.

    This is the end-to-end pin for the user-visible bug: ``winml perf`` with
    no --ep / --device on an Intel box. The chain is auto_detect_device ->
    default_ep_for_device. If the L2 filter is missing anywhere in the chain
    (registry's available_eps, default_ep_for_device, resolve_device itself),
    this test catches it.

    Stub layout:
      - auto_detect_device picks "npu" (Intel AI Boost detected)
      - QNN, OpenVINO both registered (L1 ok)
      - QNN L2-incompatible (wrong vendor), OpenVINO L2-ok
      - Expected: resolve_device returns OpenVINO, not QNN
    """
    import contextlib

    available = frozenset(
        {
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        }
    )
    compatibility = {
        "QNNExecutionProvider": False,
        "OpenVINOExecutionProvider": True,
        "CPUExecutionProvider": True,
    }

    with contextlib.ExitStack() as stack:
        for cm in _patch_available_eps(available):
            stack.enter_context(cm)
        for cm in _patch_ep_catalog_compat(compatibility):
            stack.enter_context(cm)
        stack.enter_context(
            patch("winml.modelkit.session.ep_device.auto_detect_device", return_value="npu")
        )
        result = resolve_device(EPDeviceTarget(ep="auto", device="auto"))

    assert result.ep == "OpenVINOExecutionProvider", (
        f"resolve_device(ep=auto, device=auto) returned {result.ep!r}. "
        f"Expected OpenVINOExecutionProvider (the L2-compatible NPU EP on "
        f"this stub Intel host). Returning QNNExecutionProvider is the bug — "
        f"QNN's CPU fallback registration shouldn't make it the auto pick."
    )
    assert result.device == "npu"


def test_default_ep_for_device_composes_l1_and_l2_filters() -> None:
    """RED — L1 and L2 filters must AND together, not OR.

    Mixed-mode pin per agent review: primary catalog EP (QNN) is L1-unavailable
    (not in available_eps); secondary (OpenVINO) is L1-available but
    L2-incompatible (wrong vendor on this stub); tertiary (VitisAI) is both
    L1-available AND L2-compatible. The function must walk past the first
    two failure modes — one per filter — and return the tertiary.

    A buggy fix that applies only L2 (skipping L1) would wrongly return
    OpenVINO. A buggy fix that applies only L1 (skipping L2) would also
    return OpenVINO. Only a correct fix that AND-composes the filters
    returns VitisAI here.
    """
    import contextlib

    from winml.modelkit.session import default_ep_for_device

    # QNN: L1 fail (not registered). OpenVINO: L1 ok, L2 fail. VitisAI: both ok.
    available = frozenset(
        {
            "OpenVINOExecutionProvider",
            "VitisAIExecutionProvider",
            "CPUExecutionProvider",
        }
    )
    compatibility = {
        "OpenVINOExecutionProvider": False,  # L2 fail
        "VitisAIExecutionProvider": True,  # L2 ok
        "CPUExecutionProvider": True,
    }

    with contextlib.ExitStack() as stack:
        for cm in _patch_available_eps(available):
            stack.enter_context(cm)
        for cm in _patch_ep_catalog_compat(compatibility):
            stack.enter_context(cm)
        result = default_ep_for_device("npu")

    assert result == "VitisAIExecutionProvider", (
        f"Expected VitisAIExecutionProvider (the only EP passing both L1 AND L2 "
        f"filters for NPU), but got {result!r}. L1 and L2 must AND-compose, "
        f"not OR-compose."
    )


# ---------------------------------------------------------------------------
# T-13: ep_short_or_none — dedup driver for build.py + precision.py
# ---------------------------------------------------------------------------


def test_valid_eps_matches_known_short_names() -> None:
    """``VALID_EPS`` and ``known_ep_short_names()`` must enumerate the same EPs.

    Pre-T-15 drift: ``_SHORT_TO_FULL`` carried ``"cuda"`` even though the
    EP catalog has no row for ``CUDAExecutionProvider`` (CUDA was dropped
    when the v1 catalog landed — see ``test_ep_device_specs_count``'s
    comment). ``EPDeviceTarget(ep="cuda", ...)`` passed validation but
    ``default_device_for_ep("CUDAExecutionProvider")`` returned ``None``,
    setting up a silent crash downstream.

    Post-T-15 contract: the two sets are equal — the catalog
    (``EP_DEVICE_SPECS`` → ``VALID_EPS``) is the single source of truth;
    ``_SHORT_TO_FULL`` only contains names the catalog also recognizes.
    """
    from winml.modelkit.session import VALID_EPS, known_ep_short_names

    assert known_ep_short_names() == VALID_EPS


class TestEpShortOrNone:
    """Pin the ``ep_short_or_none`` contract.

    Both ``config/build.py`` and ``config/precision.py`` carried the same
    expression ``short_ep_name(canonical) if canonical is not None else None``
    followed by ``_short if _short != "cpu" else None``. T-13 centralizes
    the ``"cpu" -> None`` collapse so the rule lives in one place.
    """

    def test_returns_short_name_for_non_cpu(self) -> None:
        """A non-CPU full EP name maps to its short form."""
        from winml.modelkit.session import ep_short_or_none

        assert ep_short_or_none("QNNExecutionProvider") == "qnn"
        assert ep_short_or_none("OpenVINOExecutionProvider") == "openvino"

    def test_returns_none_for_cpu_execution_provider(self) -> None:
        """``CPUExecutionProvider`` collapses to ``None``.

        Load-bearing per ``config/build.py`` audit notes: without the
        collapse, a CPU-only build would emit a non-None compile stage
        where the design contract says "no compile stage for CPU".
        """
        from winml.modelkit.session import ep_short_or_none

        assert ep_short_or_none("CPUExecutionProvider") is None
