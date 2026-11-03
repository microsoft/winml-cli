# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import statistics
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]


def test_eval_inputs_are_complete_and_ungraded() -> None:
    suite = json.loads((SKILL_ROOT / "evals" / "evals.json").read_text(encoding="utf-8"))
    cases = suite["evals"]
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({case["name"] for case in cases}) == len(cases)
    for case in cases:
        assert case["mode"] in {"implementation", "response"}
        assert case["prompt"] and case["assertions"]
        for assertion in case["assertions"]:
            assert assertion["text"]
            assert assertion["passed"] is None and assertion["evidence"] == ""
        for name in case["files"]:
            path = SKILL_ROOT / name
            assert path.resolve().is_relative_to(SKILL_ROOT.resolve()) and path.is_file()
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
    triggers = json.loads((SKILL_ROOT / "evals" / "trigger_eval.json").read_text(encoding="utf-8"))[
        "queries"
    ]
    assert len({item["query"] for item in triggers}) == len(triggers)
    assert all(item["query"] and type(item["should_trigger"]) is bool for item in triggers)


def test_recorded_handoffs_are_internally_consistent() -> None:
    fixture = json.loads(
        (SKILL_ROOT / "evals" / "fixtures" / "pr_description_handoffs.json").read_text(
            encoding="utf-8"
        )
    )
    perf = fixture["measurements"]["perf"]
    assert perf["mean_ms"] == statistics.mean(perf["samples_ms"])
    assert perf["p50_ms"] == statistics.median(perf["samples_ms"])
    ops = fixture["analysis"]["op_level_analysis"]
    assert ops["total_operators"] == sum(ops["operator_counts"].values())
    for classes in ops["per_ep_classification"].values():
        assert {op for group in classes.values() for op in group} == set(ops["operator_counts"])
