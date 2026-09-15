# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import shutil
import subprocess
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


def test_cleanup_dry_run_measures_empty_and_nonempty_directories(tmp_path: Path) -> None:
    script = SKILL_ROOT / "scripts" / "cleanup-run-cache.ps1"
    powershell = shutil.which("powershell")
    assert powershell is not None

    for name, contents in (("empty", b""), ("nonempty", b"bytecode")):
        case_root = tmp_path / name
        allowed_root = case_root / "run-owned"
        candidate = allowed_root / "__pycache__"
        candidate.mkdir(parents=True)
        if contents:
            (candidate / "module.pyc").write_bytes(contents)
        manifest_path = case_root / "manifest.json"
        record_path = case_root / "record.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": f"{name}-directory-test",
                    "terminal_state": "APPROVE",
                    "quiescent": True,
                    "dependencies_complete": True,
                    "allowed_roots": [str(allowed_root)],
                    "repository_roots": [],
                    "protected_paths": [],
                    "candidates": [
                        {
                            "path": str(candidate),
                            "kind": "python_bytecode",
                            "regenerable": True,
                            "dependent_checks_complete": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        result = subprocess.run(  # noqa: S603
            [
                powershell,
                "-NoProfile",
                "-File",
                str(script),
                "-ManifestPath",
                str(manifest_path),
                "-RecordPath",
                str(record_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["candidates"][0]["bytes"] == len(contents)
