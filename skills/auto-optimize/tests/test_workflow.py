# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Regression tests for evidence-bound delivery and resume."""

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / "scripts/workflow.py"
    spec = importlib.util.spec_from_file_location("workflow", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_changed_evidence_invalidates_downstream(tmp_path):
    m = module()
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    state = tmp_path / "state.json"
    for gate in m.GATES:
        m.record(state, gate, [evidence])
    assert m.status(state)["next_gate"] == "deliver"
    evidence.write_text("changed")
    assert m.status(state)["next_gate"] == m.GATES[0]


def test_cannot_skip_gate(tmp_path):
    m = module()
    p = tmp_path / "evidence.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="correctness"):
        m.record(tmp_path / "state.json", "replay", [p])


def test_prepare_uses_raw_samples_and_maps_analyzer():
    m = module()
    report = {"baseline": {"missing_reasons": {"p90_ms": "missing"}}, "leader": {}, "evidence": {}}
    session = {
        "raw_samples_ms": [1, 2, 3],
        "benchmark_info": {
            "running_model_path": "m.onnx",
            "ep": "qnn",
            "device": "npu",
            "ep_options": {},
            "batch_size": 1,
            "effective_batch_size": 1,
            "precision": "auto",
        },
    }
    analyzer = {
        "results": [{"ep_type": "qnn", "classification": {"supported": ["Conv"]}}],
        "optimization_output_support": {
            "optimizations": [
                {"name": "fusion", "worst_support": "supported", "support_counts": {"supported": 2}}
            ]
        },
    }
    result = m.prepare(report, [session], [session], analyzer)
    assert result["baseline"]["p90_ms"] == pytest.approx(2.8)
    assert "p90_ms" not in result["baseline"]["missing_reasons"]
    assert result["evidence"]["analyzer"]["coverage"] == [
        {"classification": "supported", "count": 1}
    ]
    assert result["evidence"]["analyzer"]["optimizations"][0]["instances"] == 2
    assert report["baseline"]["missing_reasons"]["p90_ms"] == "missing"


def test_delivery_refuses_incomplete_state_before_writing(tmp_path):
    m = module()
    with pytest.raises(ValueError, match="correctness"):
        m.deliver(tmp_path / "state.json", {}, tmp_path / "delivery")
    assert not (tmp_path / "delivery").exists()


def test_delivery_real_helpers_emit_receipt_and_preserve_existing(tmp_path):
    m = module()
    fixture_path = Path(__file__).with_name("test_output_bundle.py")
    spec = importlib.util.spec_from_file_location("fixtures", fixture_path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    report, champion, config, companion = fixtures._inputs(tmp_path)
    rebuild, script, lock, assets = fixtures._repro_inputs(tmp_path)
    context = tmp_path / "context.json"
    context.write_text("{}")
    cfg = {
        "report": str(report),
        "champion": str(champion),
        "winml_config": str(config),
        "companions": [str(companion)],
        "rebuild_config": str(rebuild),
        "repro_script": str(script),
        "repro_lock": str(lock),
        "repro_assets": [str(x) for x in assets],
        "promotion_context": str(context),
    }
    evidence = [report, champion, config, companion, rebuild, script, lock, *assets, context]
    state = tmp_path / "state.json"
    for gate in m.GATES:
        m.record(state, gate, evidence)
    receipt = m.deliver(state, cfg, tmp_path / "delivery")
    assert receipt["status"] == "validated"
    assert (tmp_path / "delivery/bundle/report.html").exists()
    with pytest.raises(ValueError, match="exists"):
        m.deliver(state, cfg, tmp_path / "delivery")
    champion.write_bytes(b"changed")
    with pytest.raises(ValueError, match="correctness"):
        m.deliver(state, cfg, tmp_path / "another")


def test_prepare_rejects_cross_provider_comparison():
    m = module()
    s = {
        "raw_samples_ms": [1],
        "benchmark_info": {
            "running_model_path": "m",
            "ep": "qnn",
            "device": "npu",
            "ep_options": {},
            "precision": "auto",
            "batch_size": 1,
            "effective_batch_size": 1,
        },
    }
    import copy

    other = copy.deepcopy(s)
    other["benchmark_info"]["ep"] = "cpu"
    with pytest.raises(ValueError, match="comparison"):
        m.prepare({"baseline": {}, "leader": {}, "evidence": {}}, [s], [other], {})


def test_duplicate_paths_rejected(tmp_path):
    m = module()
    p = tmp_path / "a.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="duplicate"):
        m.read_sessions([p, p])
