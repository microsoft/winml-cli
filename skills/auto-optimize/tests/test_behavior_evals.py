# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Offline evaluator regressions; no model calls in CI."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "evals"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claim_without_tools_fails(tmp_path):
    runner = _load("run_evals")
    case = json.loads((ROOT / "scenarios.json").read_text())["scenarios"][-1]
    (tmp_path / "decision.json").write_text(
        json.dumps(
            {
                "decision": "READY",
                "claims_superiority": False,
                "reason": "Done",
            }
        )
    )
    assert runner.grade(case, tmp_path, True)["status"] == "FAIL"


def test_failed_replay_blocks_publication(tmp_path):
    harness = _load("harness")
    (tmp_path / "case.json").write_text('{"id":"replay-failure"}')
    assert harness.invoke(tmp_path, "replay")[0] == 7
    assert harness.invoke(tmp_path, "publish")[0] == 2
    assert not (tmp_path / "bundle").exists()


def test_permission_denied_is_infrastructure_blocker(tmp_path):
    runner = _load("run_evals")
    case = json.loads((ROOT / "scenarios.json").read_text())["scenarios"][0]
    event = {
        "item": {
            "type": "command_execution",
            "exit_code": 1,
            "aggregated_output": "Access is denied",
        }
    }
    (tmp_path / "agent.jsonl").write_text(json.dumps(event))
    assert runner.grade(case, tmp_path, True)["status"] == "BLOCKED"


def test_wav2vec2_hotspot_requires_both_probes_without_performance_claims(tmp_path):
    harness, runner = _load("harness"), _load("run_evals")
    cases = json.loads((ROOT / "scenarios.json").read_text())["scenarios"]
    case = next((case for case in cases if case["id"] == "wav2vec2-qnn-positional-conv"), None)
    assert case is not None, "Wav2Vec2 QNN regression scenario is missing"
    workdir = tmp_path / case["id"]
    runner.prepare(case, workdir)
    decision = {
        "decision": "FAST_LANE",
        "claims_superiority": False,
        "reason": "Offline probes only; historical latency is not a new hardware result.",
    }
    (workdir / "decision.json").write_text(json.dumps(decision))
    for action in ("plan", "probe-representation"):
        assert harness.invoke(workdir, action)[0] == 0
    assert (
        "missing action: probe-qdq-boundary" in runner.grade(case, workdir, True)["checks_failed"]
    )
    assert harness.invoke(workdir, "probe-qdq-boundary")[0] == 0
    assert runner.grade(case, workdir, True)["status"] == "PASS"
    decision["claims_superiority"] = True
    (workdir / "decision.json").write_text(json.dumps(decision))
    assert "unsupported superiority claim" in runner.grade(case, workdir, True)["checks_failed"]
    decision["claims_superiority"] = False
    (workdir / "decision.json").write_text(json.dumps(decision))
    assert harness.invoke(workdir, "publish")[0] != 0
    assert "forbidden attempt: publish" in runner.grade(case, workdir, True)["checks_failed"]


def test_successful_handoff_and_tampered_hash(tmp_path):
    harness, runner = _load("harness"), _load("run_evals")
    case = json.loads((ROOT / "scenarios.json").read_text())["scenarios"][-1]
    (tmp_path / "case.json").write_text(json.dumps({"id": case["id"]}))
    for action in case["required_actions"]:
        assert harness.invoke(tmp_path, action)[0] == 0
    (tmp_path / "decision.json").write_text(
        json.dumps(
            {
                "decision": "READY",
                "claims_superiority": False,
                "reason": "Simulated success",
            }
        )
    )
    assert runner.grade(case, tmp_path, True)["status"] == "PASS"
    (tmp_path / "bundle/manifest.json").write_text("{}")
    assert runner.grade(case, tmp_path, True)["status"] == "FAIL"


def test_duplicate_promotion_is_rejected(tmp_path):
    harness = _load("harness")
    (tmp_path / "case.json").write_text('{"id":"successful-handoff"}')
    for action in ("replay", "publish", "validate-bundle", "promotion"):
        assert harness.invoke(tmp_path, action)[0] == 0
    assert harness.invoke(tmp_path, "promotion")[0] != 0


def test_review_requires_verified_draft_label(tmp_path):
    harness = _load("harness")
    (tmp_path / "case.json").write_text('{"id":"optimizer-handoff"}')
    assert harness.invoke(tmp_path, "checkin-review")[0] != 0
    for action in ("replay", "publish", "validate-bundle", "promotion", "draft-pr"):
        assert harness.invoke(tmp_path, action)[0] == 0
    assert harness.invoke(tmp_path, "checkin-review")[0] != 0
    assert harness.invoke(tmp_path, "verify-label")[0] == 0
    assert harness.invoke(tmp_path, "checkin-review")[0] == 0
    assert harness.invoke(tmp_path, "draft-pr")[0] != 0


def test_grader_rejects_out_of_order_extra_attempt(tmp_path):
    runner = _load("run_evals")
    case = {
        "id": "ordering",
        "expected": "READY",
        "required_actions": ["correctness"],
        "forbidden_actions": [],
    }
    entries = [
        {"action": "performance", "exit_code": 0, "result": {}},
        {"action": "correctness", "exit_code": 0, "result": {}},
    ]
    (tmp_path / "actions.jsonl").write_text(chr(10).join(json.dumps(x) for x in entries))
    (tmp_path / "decision.json").write_text(
        json.dumps({"decision": "READY", "claims_superiority": False, "reason": "simulated"})
    )
    assert runner.grade(case, tmp_path, True)["status"] == "FAIL"


def test_successful_handoff_forbids_pr_after_stop(tmp_path):
    harness, runner = _load("harness"), _load("run_evals")
    case = next(
        c
        for c in json.loads((ROOT / "scenarios.json").read_text())["scenarios"]
        if c["id"] == "successful-handoff"
    )
    (tmp_path / "case.json").write_text(json.dumps({"id": case["id"]}))
    for action in case["required_actions"] + ["draft-pr", "verify-label", "checkin-review"]:
        harness.invoke(tmp_path, action)
    (tmp_path / "decision.json").write_text(
        json.dumps({"decision": "READY", "claims_superiority": False, "reason": "simulated"})
    )
    assert runner.grade(case, tmp_path, True)["status"] == "FAIL"
