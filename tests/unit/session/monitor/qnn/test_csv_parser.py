# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test QNN profiling CSV parser."""

import csv
import io
from pathlib import Path

from winml.modelkit.session.monitor.qnn import parse_qnn_profiling_csv


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _write_profile(path: Path, samples: list[dict[str, int]], empty_samples: int = 0) -> None:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "Msg Timestamp",
            "Message",
            "Time",
            "Unit of Measurement",
            "Timing Source",
            "Event Level",
            "Event Identifier",
        ]
    )
    for index, sample in enumerate(samples):
        writer.writerow(
            [
                0,
                "BACKEND",
                sample["hvx_threads"],
                "COUNT",
                "BACKEND",
                "ROOT",
                "Number of HVX threads used",
            ]
        )
        writer.writerow(
            [
                0,
                "BACKEND",
                sample["accel_execute_cycles"],
                "CYCLES",
                "BACKEND",
                "ROOT",
                "Accelerator (execute) time (cycles)",
            ]
        )
        if index >= empty_samples:
            writer.writerow(
                [
                    0,
                    "NODE",
                    sample["operator_cycles"],
                    "CYCLES",
                    "BACKEND",
                    "SUB-EVENT",
                    "GeneratedOp:OpId_1 (cycles)",
                ]
            )
        writer.writerow(
            [
                0,
                "BACKEND",
                sample["accel_execute_us"],
                "US",
                "BACKEND",
                "ROOT",
                "Accelerator (execute) time",
            ]
        )
    path.write_text(output.getvalue(), encoding="utf-8")


def test_parse_csv_returns_structured_dict():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    assert isinstance(result, dict)
    assert set(result) == {"metadata", "operators", "samples"}
    assert all({"metadata", "samples"} <= sample.keys() for sample in result["samples"])


def test_parse_csv_sample_metadata():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    for sample in result["samples"]:
        meta = sample["metadata"]
        assert meta["hvx_threads"] == 4
        assert meta["accel_execute_cycles"] > 0
        assert meta["accel_execute_us"] > 0


def test_parse_csv_sample_operators():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    ops = result["samples"][0]["samples"]
    assert len(ops) > 0
    first = ops[0]
    assert "op_path" in first
    assert "op_id" in first
    assert "cycles" in first
    assert first["cycles"] > 0


def test_parse_csv_multi_sample():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    assert result["metadata"]["num_samples"] == 5
    assert len(result["samples"]) == 5


def test_parse_csv_per_sample_cycles_differ():
    """Per-sample accel cycles are captured independently, not a shared snapshot."""
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    per_sample_cycles = [sample["metadata"]["accel_execute_cycles"] for sample in result["samples"]]
    per_sample_us = [sample["metadata"]["accel_execute_us"] for sample in result["samples"]]
    assert len(set(per_sample_cycles)) > 1
    assert len(set(per_sample_us)) > 1


def test_parse_csv_each_sample_has_operators():
    """No sample is retained without operator rows."""
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    for sample in result["samples"]:
        assert len(sample["samples"]) > 0
        assert all({"op_path", "op_id", "cycles"} <= op.keys() for op in sample["samples"])


def test_parse_csv_aggregates_each_sample_operator_list():
    result = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    first_op_id = result["samples"][0]["samples"][0]["op_id"]
    expected_cycles = [
        op["cycles"]
        for sample in result["samples"]
        for op in sample["samples"]
        if op["op_id"] == first_op_id
    ]
    aggregate = next(op for op in result["operators"] if op["op_id"] == first_op_id)
    assert aggregate["samples_cycles"] == expected_cycles
    assert aggregate["cycles"] == sum(expected_cycles) / len(expected_cycles)


def test_parse_csv_drops_boundaries_without_operator_rows_and_averages_metadata(tmp_path):
    samples = [
        {
            "hvx_threads": 2,
            "accel_execute_cycles": 100,
            "accel_execute_us": 10,
            "operator_cycles": 50,
        },
        {
            "hvx_threads": 4,
            "accel_execute_cycles": 300,
            "accel_execute_us": 60,
            "operator_cycles": 150,
        },
        {
            "hvx_threads": 6,
            "accel_execute_cycles": 500,
            "accel_execute_us": 150,
            "operator_cycles": 250,
        },
    ]
    path = tmp_path / "profile.csv"
    _write_profile(path, samples, empty_samples=1)

    result = parse_qnn_profiling_csv(path)

    measured = samples[1:]
    assert len(result["samples"]) == len(measured)
    assert result["metadata"] == {
        "hvx_threads": sum(sample["hvx_threads"] for sample in measured) / len(measured),
        "accel_execute_cycles": sum(sample["accel_execute_cycles"] for sample in measured)
        / len(measured),
        "accel_execute_us": sum(sample["accel_execute_us"] for sample in measured) / len(measured),
        "num_samples": len(measured),
    }
