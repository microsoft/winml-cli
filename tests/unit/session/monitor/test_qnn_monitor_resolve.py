# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""v2.4 QNNMonitor: ``_resolve_op_type`` fallback chain + ``_heuristic_op_type``.

Phase 2 of the v2.4 simplification.  Pins the four-layer fallback chain
described in spec §5 (op-trace parser interface):

* L1 — ONNX ``node.name -> node.op_type`` lookup (primary).
* L2 — EP-authoritative (e.g. QHAS ``qnn_op_type``).
* L3 — EP-heuristic (leaf-split with strip safety).
* L4 — raw ``op_path`` verbatim.

Plus the strip / token-suffix / trailing-slash safety semantics for the
heuristic helper (FR-15 token-strip bridge, coreloop §4.3 Phase 0 fix).
"""

from __future__ import annotations

from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor


def test_resolve_l1_onnx_hit_wins(tmp_path):
    """L1 wins: ONNX op-type map hit overrides everything below it."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    mon.set_onnx_op_types({"/encoder/Conv": "Conv"})
    assert mon._resolve_op_type("/encoder/Conv", ep_authoritative="Conv2d") == "Conv"


def test_resolve_l2_ep_authoritative_wins_when_onnx_misses(tmp_path):
    """L2 wins: empty ONNX map falls through to EP-authoritative."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    mon.set_onnx_op_types({})  # no ONNX map
    assert mon._resolve_op_type("/encoder/Conv", ep_authoritative="Conv2d") == "Conv2d"


def test_resolve_l2_wins_over_heuristic_when_path_missing_from_map(tmp_path):
    """L2 wins over L3 even when the ONNX map is populated but missing this path."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    mon.set_onnx_op_types({"/some/other/Path": "Add"})  # populated but irrelevant
    # heuristic would yield "Conv"; L2 must beat it
    assert mon._resolve_op_type("/encoder/Conv", ep_authoritative="Conv2d") == "Conv2d"


def test_resolve_l3_heuristic_wins_when_no_authoritative(tmp_path):
    """L3 wins: empty ONNX map + no ep_authoritative falls through to heuristic."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    mon.set_onnx_op_types({})
    # CSV basic mode: no ep_authoritative passed
    assert mon._resolve_op_type("/encoder/Conv") == "Conv"


def test_resolve_l4_raw_op_path_when_all_fall_through(tmp_path):
    """L4 wins: bare event with no slash and no token suffix → return verbatim."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    mon.set_onnx_op_types({})
    # Bare event with no slash and no _token suffix to strip
    assert mon._resolve_op_type("Gelu") == "Gelu"


def test_resolve_default_state_is_empty_onnx_map(tmp_path):
    """A fresh monitor has an empty ONNX map (no implicit population)."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    # No call to set_onnx_op_types; resolver should still work and fall through.
    assert mon._resolve_op_type("/encoder/Conv", ep_authoritative="Conv2d") == "Conv2d"
    assert mon._resolve_op_type("/encoder/Conv") == "Conv"


def test_heuristic_strip_safety_outer_whitespace(tmp_path):
    """Outer whitespace is stripped before splitting."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    assert mon._heuristic_op_type("  /encoder/Conv  ") == "Conv"


def test_heuristic_strip_safety_inner_whitespace_around_leaf(tmp_path):
    """Inner whitespace around the leaf is stripped after the split."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    assert mon._heuristic_op_type("/encoder/  Conv  ") == "Conv"


def test_heuristic_trailing_slash_falls_back_to_full(tmp_path):
    """Trailing-slash input yields an empty leaf → fall back to cleaned input."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    # The legacy contract: never return an empty string for non-empty input.
    assert mon._heuristic_op_type("/encoder/") == "/encoder/"


def test_heuristic_token_suffix_stripped_before_split(tmp_path):
    """The QNN compiler's ``_token_\\d+(?:_\\d+)?`` suffix is stripped first."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    assert mon._heuristic_op_type("/encoder/Conv_token_3_1") == "Conv"


def test_heuristic_bare_event_id_returns_verbatim(tmp_path):
    """A bare op-type event id (no slash) is returned as-is after strip."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    assert mon._heuristic_op_type("Gelu") == "Gelu"


def test_heuristic_empty_input_does_not_crash(tmp_path):
    """Empty input is degenerate but must not raise."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    # Empty after strip → no slash → returns ""
    assert mon._heuristic_op_type("") == ""


def test_set_onnx_op_types_copies_input(tmp_path):
    """Mutating the caller's dict after injection must not affect the monitor."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    src = {"/encoder/Conv": "Conv"}
    mon.set_onnx_op_types(src)
    src["/encoder/Conv"] = "Add"  # caller mutates after injection
    src["/encoder/New"] = "Mul"  # caller adds after injection
    # Monitor's resolver still uses the original snapshot.
    assert mon._resolve_op_type("/encoder/Conv") == "Conv"
    # And the new key was never seen by the monitor → falls through.
    assert mon._resolve_op_type("/encoder/New") == "New"


def test_set_onnx_op_types_overwrites_previous_call(tmp_path):
    """Subsequent ``set_onnx_op_types`` replaces (not merges) the prior map."""
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    mon.set_onnx_op_types({"/a/Conv": "Conv"})
    mon.set_onnx_op_types({"/b/Add": "Add"})
    # First key gone after second call.
    assert mon._resolve_op_type("/a/Conv") == "Conv"  # heuristic kicks in
    assert mon._resolve_op_type("/b/Add") == "Add"  # L1 hit on second map


def test_resolve_falls_through_when_onnx_map_has_empty_string_value(tmp_path):
    """CRIT-2: empty-string ONNX op_type values must NOT short-circuit L1.

    If a malformed ONNX file produced an empty op_type (theoretically
    possible — onnx.checker rejects it, but ``_build_op_type_map`` uses
    ``onnx.load`` without ``check_model``), the chain should fall
    through to L2/L3/L4 instead of returning empty string as the op type.

    This is double-defense: ``_build_op_type_map`` also filters
    empty-op_type entries at construction time.
    """
    mon = QNNMonitor(level="basic", output_dir=tmp_path)
    mon.set_onnx_op_types({"/encoder/Conv": ""})  # pathological empty value

    # L2 wins because L1 hit-but-empty falls through
    result = mon._resolve_op_type("/encoder/Conv", ep_authoritative="Conv2d")
    assert result == "Conv2d", f"Expected L2 fallback; got {result!r}"

    # No ep_authoritative + empty L1 → heuristic (L3) wins
    result = mon._resolve_op_type("/encoder/Conv")
    assert result == "Conv", f"Expected L3 heuristic leaf-split; got {result!r}"
