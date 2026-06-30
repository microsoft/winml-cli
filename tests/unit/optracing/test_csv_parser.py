# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test QNN profiling CSV parser."""

from pathlib import Path

from winml.modelkit.optracing.qnn.csv_parser import parse_qnn_profiling_csv


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_csv_returns_dict():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    assert isinstance(result, dict)
    assert "metadata" in result
    assert "operators" in result
    assert "samples" in result


def test_parse_csv_metadata():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    meta = result["metadata"]
    assert meta["hvx_threads"] == 4
    assert meta["accel_execute_cycles"] > 0
    assert meta["num_samples"] >= 1


def test_parse_csv_operators():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    ops = result["operators"]
    assert len(ops) > 0
    first = ops[0]
    assert "name" in first
    assert "op_id" in first
    assert "cycles" in first
    assert first["cycles"] > 0


def test_parse_csv_operators_sorted_by_cycles():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    ops = result["operators"]
    cycles = [op["cycles"] for op in ops]
    assert cycles == sorted(cycles, reverse=True)


def test_parse_csv_multi_sample():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    assert result["metadata"]["num_samples"] >= 1
    assert len(result["samples"]) >= 1


def test_parse_csv_sample_carries_own_metadata():
    """Each sample exposes its own metadata + operator list."""
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    assert len(result["samples"]) == result["metadata"]["num_samples"]

    for sample in result["samples"]:
        meta = sample["metadata"]
        assert meta["hvx_threads"] == 4
        assert meta["accel_execute_cycles"] > 0
        assert meta["accel_execute_us"] > 0
        # The per-sample operator list is non-empty and well-formed.
        assert len(sample["samples"]) > 0
        assert all({"name", "op_id", "cycles"} <= op.keys() for op in sample["samples"])


def test_parse_csv_per_sample_cycles_differ():
    """Per-sample accel cycles are captured independently, not a shared snapshot."""
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    per_sample_cycles = [s["metadata"]["accel_execute_cycles"] for s in result["samples"]]
    # The fixture has distinct accelerator cycle counts across its samples.
    assert len(set(per_sample_cycles)) > 1
