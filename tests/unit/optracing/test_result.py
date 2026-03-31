# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test OpTraceResult dataclass and serialization."""

import json

from winml.modelkit.optracing import OperatorMetrics, OpTraceResult


def test_operator_metrics_to_dict():
    op = OperatorMetrics(name="Conv2d", op_path="/layer1/conv/Conv", duration_us=45.2)
    d = op.to_dict()
    assert d["name"] == "Conv2d"
    assert d["op_path"] == "/layer1/conv/Conv"
    assert d["duration_us"] == 45.2
    assert d["dram_read_bytes"] is None


def test_operator_metrics_with_detail_fields():
    op = OperatorMetrics(
        name="Conv2d",
        op_path="/conv",
        duration_us=100.0,
        dram_read_bytes=1024,
        vtcm_read_bytes=4096,
        vtcm_hit_ratio=0.8,
        dominant_path_us=50.0,
    )
    d = op.to_dict()
    assert d["dram_read_bytes"] == 1024
    assert d["vtcm_hit_ratio"] == 0.8
    assert d["dominant_path_us"] == 50.0


def test_op_trace_result_to_dict():
    result = OpTraceResult(
        model="resnet-50",
        device="npu",
        tracing_level="basic",
        operators=[OperatorMetrics(name="Conv2d", op_path="/conv", duration_us=10.0)],
    )
    d = result.to_dict()
    assert d["metadata"]["model"] == "resnet-50"
    assert d["metadata"]["device"] == "npu"
    assert d["metadata"]["tracing_level"] == "basic"
    assert len(d["operators"]) == 1
    assert d["operators"][0]["name"] == "Conv2d"


def test_op_trace_result_to_json():
    result = OpTraceResult(
        model="resnet-50",
        device="npu",
        tracing_level="detail",
        ep="QNNExecutionProvider",
        operators=[
            OperatorMetrics(name="Conv2d", op_path="/conv", duration_us=10.0),
            OperatorMetrics(name="Add", op_path="/add", duration_us=5.0),
        ],
        summary={"time_us": 1343, "utilization_pct": 99.59},
    )
    j = result.to_json()
    parsed = json.loads(j)
    assert parsed["metadata"]["model"] == "resnet-50"
    assert parsed["metadata"]["ep"] == "QNNExecutionProvider"
    assert len(parsed["operators"]) == 2
    assert parsed["summary"]["time_us"] == 1343


def test_op_trace_result_empty():
    result = OpTraceResult(model="test", device="cpu", tracing_level="basic")
    d = result.to_dict()
    assert d["operators"] == []
    assert d["summary"] == {}
