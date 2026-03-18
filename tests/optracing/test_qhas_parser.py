"""Test QHAS JSON parser."""
import json
from pathlib import Path

from winml.modelkit.optracing.qnn.qhas_parser import parse_qhas


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_qhas():
    with (FIXTURE_DIR / "qhas_resnet50.json").open() as f:
        return json.load(f)


def test_parse_qhas_returns_dict():
    result = parse_qhas(_load_qhas())
    assert "summary" in result
    assert "operators" in result


def test_parse_qhas_summary():
    result = parse_qhas(_load_qhas())
    s = result["summary"]
    assert s["time_us"] > 0
    assert s["graph_execute_us"] > 0
    assert s["total_dram_read"] > 0
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
