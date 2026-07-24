# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QNN CSV parser must retain per-sample timings for each operator.

The parser keys ops by ``op_id`` and stores per-sample timing in
``cycles``.  The aggregator must additionally retain a per-sample list
under ``samples_cycles`` so that downstream layers can compute p90 /
total / count without re-parsing the CSV.

The cycle->microsecond conversion happens in ``QNNMonitor`` (which
owns the ``cycle_to_us`` ratio derived from ROOT-level metadata), so
the parser deliberately stays in the cycles domain.
"""

from __future__ import annotations

from winml.modelkit.session.monitor.qnn._internal import _aggregate_operators


def test_per_sample_retention_preserves_order() -> None:
    """``samples_cycles`` is a list of per-sample cycle counts in input order."""
    sample_1 = [
        {"op_path": "Conv2d", "op_id": 1, "cycles": 100},
        {"op_path": "Relu", "op_id": 2, "cycles": 5},
    ]
    sample_2 = [
        {"op_path": "Conv2d", "op_id": 1, "cycles": 110},
        {"op_path": "Relu", "op_id": 2, "cycles": 6},
    ]
    ops = _aggregate_operators([sample_1, sample_2])
    by_id = {op["op_id"]: op for op in ops}

    assert by_id[1]["samples_cycles"] == [100, 110]
    assert by_id[2]["samples_cycles"] == [5, 6]


def test_per_sample_back_compat_avg_cycles() -> None:
    """``cycles`` still equals avg across samples (back-compat for callers)."""
    sample_1 = [{"op_path": "X", "op_id": 7, "cycles": 100}]
    sample_2 = [{"op_path": "X", "op_id": 7, "cycles": 300}]
    ops = _aggregate_operators([sample_1, sample_2])
    assert len(ops) == 1
    assert ops[0]["cycles"] == 200.0  # avg, unchanged
    assert ops[0]["samples_cycles"] == [100, 300]


def test_per_sample_single_sample() -> None:
    """Single-sample input still yields a single-element ``samples_cycles``."""
    ops = _aggregate_operators([[{"op_path": "X", "op_id": 1, "cycles": 42}]])
    assert ops[0]["samples_cycles"] == [42]
    assert ops[0]["cycles"] == 42.0


def test_per_sample_empty_input() -> None:
    """Empty input produces no operators (unchanged behaviour)."""
    assert _aggregate_operators([]) == []


def test_per_sample_op_missing_in_some_samples() -> None:
    """An op that appears in only one sample has only one entry in its list.

    This matches the existing aggregator's count-based avg semantics:
    if op X shows up once across two samples, its avg uses divisor=1
    (current behaviour), and ``samples_cycles`` likewise has length 1.
    """
    sample_1 = [
        {"op_path": "Conv2d", "op_id": 1, "cycles": 100},
        {"op_path": "OneShot", "op_id": 99, "cycles": 7},
    ]
    sample_2 = [
        {"op_path": "Conv2d", "op_id": 1, "cycles": 200},
    ]
    ops = _aggregate_operators([sample_1, sample_2])
    by_id = {op["op_id"]: op for op in ops}

    assert by_id[1]["samples_cycles"] == [100, 200]
    assert by_id[99]["samples_cycles"] == [7]
