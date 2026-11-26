# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (SKILL_ROOT / relative).read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("role", ["planner", "producer", "tester", "reviewer"])
def test_executing_roles_require_the_shared_gate(role: str) -> None:
    text = _text(f"agents/{role}.md")
    assert "./orchestrator.md#custom-code-execution-gate" in text
    assert "explicit user authorization" in text or "explicitly authorized by the user" in text
    assert "credential-free" in text
    assert "blocked" in text


def test_gate_requires_independent_authorization_and_isolation() -> None:
    text = _text("agents/orchestrator.md")
    gate = text.split("## custom-code execution gate\n", 1)[1].split("\n## ", 1)[0]
    for requirement in (
        "reviewed immutable model and executable-code commit revisions",
        "separately referenced code repository",
        "explicit user authorization",
        "trusted user conversation",
        "a changed revision requires a new review and authorization",
        "credential-free isolated execution",
        "credential helpers",
        "privileged host sockets",
        "host credential services",
        "disable network access",
        "a worktree, virtual environment, or fresh subagent alone is not isolation",
        "do not fall back to mutable references",
        "stop as blocked",
        "do not retry automatically",
        "github publication outside the execution worker",
        "overrides standing loop authorization",
        "planning, config, build, perf, eval, testing, and review",
    ):
        assert requirement in gate


def test_dispatcher_and_loop_authorization_preserve_the_gate() -> None:
    skill = _text("SKILL.md")
    orchestrator = _text("agents/orchestrator.md")
    assert "./agents/orchestrator.md#custom-code-execution-gate" in skill
    assert "before any probe or run" in skill
    assert "required custom-code trust decision" in orchestrator
    assert "mandatory exception to automatic continuation" in orchestrator
    assert "you may pause **only**" not in orchestrator


def test_missing_flag_is_not_automatic_execution_permission() -> None:
    assert "pass `--trust-remote-code` on every" not in _text("agents/producer.md")
    assert "missing-flag operator error" not in _text("agents/tester.md")
    assert "commands must carry the cli flag" not in _text("agents/reviewer.md")


def test_explanation_only_queries_remain_negative_without_excluding_repairs() -> None:
    skill = _text("SKILL.md")
    assert "explanation-only questions" in skill
    assert "implement or repair model support" in skill
    cases = json.loads((SKILL_ROOT / "evals/trigger_eval.json").read_text(encoding="utf-8"))
    queries = {item["query"]: item["should_trigger"] for item in cases["queries"]}
    assert (
        queries[
            "winml analyze says Conv is unsupported on QNN for my resnet that already has a recipe "
            "\u2014 why is op coverage reporting that"
        ]
        is False
    )
    assert (
        queries[
            "the QNN compile backend segfaults on my model, "
            "can you explain how the EPContext compile path is structured"
        ]
        is False
    )


def test_response_regression_covers_all_trust_gate_states() -> None:
    cases = json.loads((SKILL_ROOT / "evals/evals.json").read_text(encoding="utf-8"))["evals"]
    case = next(item for item in cases if item["name"] == "custom-code-authorization-and-isolation")
    assert len({item["id"] for item in cases}) == len(cases)
    for label in ("A)", "B)", "C)", "D)"):
        assert label in case["prompt"]
    assert "do not run commands" in case["prompt"]
    assert all(item["passed"] is None for item in case["assertions"])
    assert len(case["assertions"]) == 8
