# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests using real QNN profiling data."""
import json
from pathlib import Path

from winml.modelkit.optracing.qnn.csv_parser import parse_qnn_profiling_csv
from winml.modelkit.optracing.qnn.qhas_parser import parse_qhas
from winml.modelkit.optracing.report import write_op_trace_json
from winml.modelkit.optracing.result import OperatorMetrics, OpTraceResult


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_basic_pipeline_csv_to_json(tmp_path):
    """Full basic mode: CSV -> OpTraceResult -> JSON file."""
    csv_data = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")

    total_cycles = sum(op["cycles"] for op in csv_data["operators"])

    operators = [
        OperatorMetrics(
            name=op["name"],
            op_path=op["name"],  # CSV doesn't distinguish type vs path
            op_id=op["op_id"],
            duration_us=op["cycles"],  # keep raw cycles as duration placeholder
            percent_of_total=(
                (op["cycles"] / total_cycles * 100) if total_cycles else 0
            ),
        )
        for op in csv_data["operators"]
    ]

    result = OpTraceResult(
        model="resnet-50",
        device="npu",
        tracing_level="basic",
        operators=operators,
        num_samples=csv_data["metadata"]["num_samples"],
        summary=csv_data["metadata"],
    )

    out = tmp_path / "basic_op_trace.json"
    write_op_trace_json(result, out)

    assert out.exists()
    data = json.loads(out.read_text())
    assert data["metadata"]["tracing_level"] == "basic"
    assert len(data["operators"]) > 0
    assert data["operators"][0]["duration_us"] > 0


def test_detail_pipeline_qhas_to_json(tmp_path):
    """Full detail mode: QHAS -> OpTraceResult -> JSON file."""
    qhas_raw = json.loads((FIXTURE_DIR / "qhas_resnet50.json").read_text())

    parsed = parse_qhas(qhas_raw)

    operators = [
        OperatorMetrics(
            name=op["name"],
            op_path=op["op_path"],
            duration_us=op["duration_us"],
            percent_of_total=op["percent_of_total"],
            dominant_path_us=op.get("dominant_path_us"),
            dram_read_bytes=op.get("dram_read_bytes"),
            dram_write_bytes=op.get("dram_write_bytes"),
            vtcm_read_bytes=op.get("vtcm_read_bytes"),
            vtcm_write_bytes=op.get("vtcm_write_bytes"),
            vtcm_hit_ratio=op.get("vtcm_hit_ratio"),
            num_htp_ops=op.get("num_htp_ops"),
        )
        for op in parsed["operators"]
    ]

    result = OpTraceResult(
        model="resnet-50",
        device="npu",
        tracing_level="detail",
        ep="QNNExecutionProvider",
        operators=operators,
        summary=parsed["summary"],
    )

    out = tmp_path / "detail_op_trace.json"
    write_op_trace_json(result, out)

    data = json.loads(out.read_text())
    assert data["metadata"]["tracing_level"] == "detail"
    assert data["summary"]["time_us"] > 0
    # At least one operator should have DRAM read data populated
    assert any(
        op["dram_read_bytes"] is not None for op in data["operators"]
    )


def test_json_schema_basic():
    """Verify basic mode JSON has required keys."""
    result = OpTraceResult(
        model="test",
        device="npu",
        tracing_level="basic",
        operators=[
            OperatorMetrics(name="Conv", op_path="/conv", duration_us=10.0)
        ],
    )
    data = result.to_dict()

    assert "metadata" in data
    assert "operators" in data
    assert "summary" in data
    assert "statistics" in data
    assert "artifacts" in data

    meta = data["metadata"]
    for key in ("model", "device", "tracing_level", "timestamp"):
        assert key in meta


def test_json_schema_detail():
    """Verify detail mode JSON has P0-P3 fields."""
    result = OpTraceResult(
        model="test",
        device="npu",
        tracing_level="detail",
        operators=[
            OperatorMetrics(
                name="Conv2d",
                op_path="/conv",
                duration_us=100.0,
                dram_read_bytes=1024,
                vtcm_read_bytes=4096,
                vtcm_hit_ratio=0.8,
                dominant_path_us=50.0,
            )
        ],
    )
    data = result.to_dict()
    op = data["operators"][0]

    # P0: Temporal Localization
    assert "duration_us" in op
    # P1: Roofline Analysis
    assert "dominant_path_us" in op
    # P2: DMA Traffic
    assert "dram_read_bytes" in op
    assert "vtcm_read_bytes" in op
    # P3: Cache Efficiency
    assert "vtcm_hit_ratio" in op


def test_round_trip_json():
    """OpTraceResult -> JSON -> parse back -> verify fields match."""
    original = OpTraceResult(
        model="resnet-50",
        device="npu",
        tracing_level="detail",
        ep="QNNExecutionProvider",
        operators=[
            OperatorMetrics(
                name="Conv2d",
                op_path="/layer1/conv",
                duration_us=123.4,
                percent_of_total=45.6,
                dram_read_bytes=2048,
                vtcm_hit_ratio=0.95,
            ),
            OperatorMetrics(
                name="ReLU",
                op_path="/layer1/relu",
                duration_us=10.0,
                percent_of_total=3.7,
            ),
        ],
        summary={"time_us": 270.5, "utilization_pct": 83.5},
    )

    json_str = original.to_json()
    parsed = json.loads(json_str)

    # Metadata round-trip
    assert parsed["metadata"]["model"] == "resnet-50"
    assert parsed["metadata"]["tracing_level"] == "detail"
    assert parsed["metadata"]["ep"] == "QNNExecutionProvider"

    # Operators round-trip
    assert len(parsed["operators"]) == 2
    op0 = parsed["operators"][0]
    assert op0["name"] == "Conv2d"
    assert op0["op_path"] == "/layer1/conv"
    assert op0["duration_us"] == 123.4
    assert op0["percent_of_total"] == 45.6
    assert op0["dram_read_bytes"] == 2048
    assert op0["vtcm_hit_ratio"] == 0.95

    op1 = parsed["operators"][1]
    assert op1["name"] == "ReLU"
    assert op1["dram_read_bytes"] is None  # not set => None preserved

    # Summary round-trip
    assert parsed["summary"]["time_us"] == 270.5


def test_csv_parser_operator_count():
    """CSV parser finds the expected number of operators."""
    data = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    # ResNet-50 produces ~79 aggregated QNN ops from the fixture
    assert len(data["operators"]) > 50


def test_qhas_parser_operator_count():
    """QHAS parser finds operators in fixture."""
    qhas = json.loads((FIXTURE_DIR / "qhas_resnet50.json").read_text())
    parsed = parse_qhas(qhas)
    assert len(parsed["operators"]) > 0


def test_cross_parser_top_operator_is_conv():
    """Both parsers should show Conv as the top operator for ResNet."""
    # CSV: operators are sorted by cycles descending
    csv_data = parse_qnn_profiling_csv(FIXTURE_DIR / "optrace_resnet50.csv")
    top_csv = csv_data["operators"][0]["name"].lower()

    # QHAS: operators are not pre-sorted; find the one with max duration
    qhas_raw = json.loads((FIXTURE_DIR / "qhas_resnet50.json").read_text())
    parsed = parse_qhas(qhas_raw)
    top_qhas = max(parsed["operators"], key=lambda op: op["duration_us"])
    top_qhas_name = top_qhas["name"].lower()

    # The top op for ResNet should contain "conv" (the large 7x7 convolution)
    assert "conv" in top_csv, f"Expected 'conv' in top CSV op: {top_csv}"
    assert "conv" in top_qhas_name, (
        f"Expected 'conv' in top QHAS op: {top_qhas_name}"
    )
