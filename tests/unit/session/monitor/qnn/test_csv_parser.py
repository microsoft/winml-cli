# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test QNN profiling CSV parser."""

from pathlib import Path

from winml.modelkit.session.monitor.qnn import parse_qnn_profiling_csv


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_csv_returns_sample_list():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all({"metadata", "samples"} <= entry.keys() for entry in result)


def test_parse_csv_sample_metadata():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    for sample in result:
        meta = sample["metadata"]
        assert meta["hvx_threads"] == 4
        assert meta["accel_execute_cycles"] > 0
        assert meta["accel_execute_us"] > 0


def test_parse_csv_sample_operators():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    ops = result[0]["samples"]
    assert len(ops) > 0
    first = ops[0]
    assert "op_path" in first
    assert "op_id" in first
    assert "cycles" in first
    assert first["cycles"] > 0


def test_parse_csv_multi_sample():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    # The fixture captures several inference samples.
    assert len(result) > 1


def test_parse_csv_per_sample_cycles_differ():
    """Per-sample accel cycles are captured independently, not a shared snapshot."""
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    per_sample_cycles = [s["metadata"]["accel_execute_cycles"] for s in result]
    # The fixture has distinct accelerator cycle counts across its samples.
    assert len(set(per_sample_cycles)) > 1


def test_parse_csv_each_sample_has_operators():
    """No sample is retained without operator rows."""
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    for sample in result:
        assert len(sample["samples"]) > 0
        assert all({"name", "op_id", "cycles"} <= op.keys() for op in sample["samples"])
