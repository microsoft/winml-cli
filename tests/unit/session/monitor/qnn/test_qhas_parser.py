# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test QHAS JSON parser."""

import json
from pathlib import Path

import pytest

from winml.modelkit.session.monitor.qnn import parse_qhas


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_qhas():
    with (FIXTURE_DIR / "qhas_resnet50.json").open() as f:
        return json.load(f)


def test_parse_qhas_returns_dict():
    result = parse_qhas(_load_qhas())
    assert "summary" in result
    assert "operators" in result


def test_parse_qhas_summary():
    """Summary keys MUST match what report._display_detail_report reads.

    Pre-Bundle-A bug (I-9): the parser produced raw QHAS-source keys
    (``time_us``, ``graph_execute_us``, ``total_dram_read``) while the
    renderer read user-facing keys (``inference_us``, ``execute_us``,
    ``dram_read_bytes``).  5 of 6 keys were disjoint so the detail-mode
    summary line silently rendered empty for real production data.
    """
    result = parse_qhas(_load_qhas())
    s = result["summary"]
    assert s["inference_us"] > 0
    assert s["execute_us"] > 0
    assert s["dram_read_bytes"] > 0
    assert s["qnn_nodes"] > 0
    assert s["utilization_pct"] > 0


def test_parse_qhas_operators():
    result = parse_qhas(_load_qhas())
    ops = result["operators"]
    assert len(ops) > 0
    first = ops[0]
    assert first["duration_us"] > 0
    assert first["percent_of_total"] > 0
    assert "dram_read_bytes" in first
    assert "vtcm_read_bytes" in first
    assert "name" in first
    assert "op_path" in first


def test_parse_qhas_dominant_path():
    result = parse_qhas(_load_qhas())
    ops = result["operators"]
    has_dp = any(op.get("dominant_path_us") is not None for op in ops)
    assert has_dp


def test_parse_qhas_vtcm_hit_ratio():
    result = parse_qhas(_load_qhas())
    ops = result["operators"]
    # At least some ops should have vtcm_hit_ratio
    has_ratio = any(op.get("vtcm_hit_ratio") is not None for op in ops)
    assert has_ratio


def test_extract_summary_produces_renderer_expected_keys():
    """The parser's ``_extract_summary`` keys MUST match what the
    :func:`winml.modelkit.session.monitor.report._display_detail_report`
    renderer reads.

    Pre-Bundle-A bug (I-9): 5 of 6 user-facing keys were disjoint
    between parser and renderer.  Renderer expected
    ``inference_us`` / ``execute_us`` / ``dram_read_bytes`` /
    ``dram_write_bytes`` / ``vtcm_peak_bytes``; parser produced
    raw QHAS ``time_us`` / ``graph_execute_us`` / ``total_dram_read``
    / ``total_dram_write`` / ``peak_vtcm_alloc``.  Only
    ``utilization_pct`` matched.  Result: detail-mode summary line
    silently rendered empty for real production data.

    Snapshot test pinning the exact expected key set so any future
    drift on either side breaks loudly.
    """
    parsed = parse_qhas(_load_qhas())
    summary = parsed["summary"]

    expected_renderer_keys = {
        "inference_us",
        "execute_us",
        "utilization_pct",
        "dram_read_bytes",
        "dram_write_bytes",
        "vtcm_peak_bytes",
    }
    missing = expected_renderer_keys - set(summary.keys())
    assert not missing, (
        f"Parser-produced summary keys MUST be a superset of what the renderer "
        f"reads.  Got {sorted(summary.keys())}; renderer expects "
        f"{sorted(expected_renderer_keys)}; missing: {sorted(missing)}.  "
        f"Renderer is in src/winml/modelkit/session/monitor/report.py "
        f"::_display_detail_report."
    )


def test_parse_qhas_uses_authoritative_qnn_op_type_for_name():
    """QHAS-sourced ops MUST use ``qnn_op_type`` for ``name``, not a
    leaf-split of the ``qnn_op`` framework path.

    This is the load-bearing distinction between the two vocabularies:

    - QHAS ``qnn_op_type`` is the authoritative QNN op type
      (``"Conv2d"``, ``"ElementWiseAdd"``, ``"PoolMax2d"``, ...).
    - The leaf segment of ``qnn_op`` is the ONNX op symbol
      (``"Conv"``, ``"Add"``, ``"MaxPool"``, ...).

    A previous fix (``c3ac3d45``) applied the CSV-only leaf-split heuristic
    uniformly to QHAS as well, which silently degraded ``name`` to the
    ONNX symbol and would have invited a hardcoded translation table to
    reconcile vocabularies — a Cardinal Rule #1 violation.  This test
    pins the regression by asserting canonical QNN op type values from
    the resnet50 fixture.
    """
    result = parse_qhas(_load_qhas())
    ops = result["operators"]

    # Per the resnet50 fixture, the first op is a Conv with
    # qnn_op="/resnet/embedder/embedder/convolution/Conv_token_1_2".  A
    # leaf-split of qnn_op would yield "Conv_token_1_2" (or "Conv" after
    # token stripping); the authoritative qnn_op_type is "Conv2d".
    first = ops[0]
    assert first["name"] == "Conv2d", (
        f"first op name must be authoritative QNN op type 'Conv2d'; "
        f"got {first['name']!r} (likely a leaf-split of qnn_op)"
    )
    # And the framework path is preserved with ``_token_N_M`` stripped
    # (CRIT-1 fix): the QHAS path's ``op_path`` is normalised so it
    # matches the clean ONNX ``node.name`` keys produced by
    # :py:meth:`WinMLSession._build_op_type_map`.  Without this strip
    # the FR-14 L1 ONNX-primary lookup is silently inert in detail
    # mode.  Strip is idempotent on already-clean strings.
    assert first["op_path"] == "/resnet/embedder/embedder/convolution/Conv"

    # The QNN op type vocabulary set must NOT contain ONNX op symbols.
    names = {op["name"] for op in ops}
    qnn_canonical_seen = names & {"Conv2d", "ElementWiseAdd", "PoolMax2d", "PoolAvg2d", "Transpose"}
    assert qnn_canonical_seen, f"expected canonical QNN op types in fixture; got {sorted(names)}"
    onnx_symbols_in_names = names & {"Conv", "Add", "MaxPool", "AveragePool"}
    assert not onnx_symbols_in_names, (
        f"QHAS path leaked ONNX op symbols into name; got {sorted(onnx_symbols_in_names)}. "
        f"This indicates the leaf-split heuristic was wrongly applied to QHAS data."
    )


# ---------------------------------------------------------------------------
# A2: named KeyError on missing QHAS fields (Bundle A)
# ---------------------------------------------------------------------------


def test_qnn_internal_missing_key_named_in_error():
    """A2: a bare dict[key] access raises an opaque KeyError that the outer
    _try_qhas except silently swallows, degrading to 'basic_fallback' with
    no record of which field was missing.

    After the fix, missing required fields raise KeyError with the field name
    in the message so the outer handler can log it for SDK schema drift
    diagnosis.

    Test: feed _extract_summary a QHAS structure whose htp_overall_summary
    row is missing the required 'time_us' field, and assert the raised
    KeyError message contains the field name.
    """
    import copy

    from winml.modelkit.session.monitor.qnn._internal import _extract_summary

    data = _load_qhas()["data"]
    broken = copy.deepcopy(data)
    # Remove a required field from the first summary row.
    del broken["htp_overall_summary"]["data"][0]["time_us"]

    with pytest.raises(KeyError) as exc_info:
        _extract_summary(broken)

    msg = str(exc_info.value)
    assert "time_us" in msg, (
        f"KeyError message must name the missing field 'time_us' so SDK schema "
        f"drift is diagnosable; got: {msg!r}"
    )


def test_qnn_internal_missing_transform_key_named_in_error():
    """A2 (transform path): _transform_op raises a named KeyError when a
    required field is absent from a qnn_op_instances_nodes entry.

    Covers the 'cycles' / 'qnn_op_type' / 'qnn_op' / 'percent_active_cycles'
    group of required fields in _transform_op.
    """
    import pytest

    from winml.modelkit.session.monitor.qnn._internal import _transform_op

    # A minimal valid op dict with one required field removed.
    op = {
        "qnn_op_type": "Conv2d",
        "qnn_op": "/resnet/Conv_token_1_2",
        "percent_active_cycles": 12.5,
        # "cycles" intentionally omitted
        "num_htp_ops": 1,
    }

    with pytest.raises(KeyError) as exc_info:
        _transform_op(op, cycle_to_us=0.005)

    msg = str(exc_info.value)
    assert "cycles" in msg, f"KeyError message must name the missing field 'cycles'; got: {msg!r}"
