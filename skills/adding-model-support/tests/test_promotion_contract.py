# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (SKILL_ROOT / relative).read_text(encoding="utf-8").lower()


def test_promotion_entry_uses_the_auto_optimize_validator() -> None:
    skill = _text("SKILL.md")
    orchestrator = _text("agents/orchestrator.md")

    for required in (
        "--promotion-handoff <absolute path>",
        "promotion.py validate --handoff",
        'routes.recipe == "eligible"',
        "blocked_on_optimizer",
        "adding-model-support owns the recipe pr",
        "separately installed `auto-optimize` skill",
        "normal model-support entry is unaffected",
    ):
        assert required in orchestrator, required
    assert "promotion handoff" in skill
    assert "agents/orchestrator.md#promotion-handoff-entry" in skill


def test_consumer_updates_only_the_recipe_route() -> None:
    orchestrator = _text("agents/orchestrator.md")

    for required in (
        "--route recipe --status in_progress --pr-url",
        "--route recipe --status approved",
        "model-scale-by-skill",
        "never updates the optimizer route",
    ):
        assert required in orchestrator, required
    assert "convertfrom-json" not in orchestrator
    assert "invoke-strictjsonparser" not in orchestrator


def test_promotion_planner_is_recipe_only() -> None:
    planner = _text("agents/planner.md")

    for required in (
        "promotion mode",
        "recipe-only",
        "code_paths",
        "never implement optimizer code",
        "validated route summary",
    ):
        assert required in planner, required


def test_orchestrator_requires_fresh_subagent_dispatch() -> None:
    skill = _text("SKILL.md")
    orchestrator = _text("agents/orchestrator.md")

    for required in (
        "agent runtime must support fresh subagent delegation",
        "if delegation is unavailable, stop as blocked",
    ):
        assert required in skill, required
    for required in (
        "agent/subagent delegation capability",
        "start a fresh agent",
        "start the reviewer from a fresh context",
        "never emulate all roles in one context",
    ):
        assert required in orchestrator, required


def test_reviewer_eval_matches_the_active_verdict_contract() -> None:
    reviewer = _text("agents/reviewer.md")
    evals = _text("evals/evals.json")

    for verdict in ("approve", "request_changes", "reject"):
        assert verdict in reviewer
        assert verdict in evals
    assert "a fixable compatibility break is `request_changes`" in reviewer
    for stale_verdict in ("acceptable/changes_needed/unsound", "formal approved"):
        assert stale_verdict not in evals
