# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Run live Codex agents against offline fixtures and preserve reviewable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SKILL = ROOT.parent


def grade(case: dict, workdir: Path, agent_ok: bool) -> dict:
    """Grade observed attempts, not an agent's claims that it used tools."""
    log = workdir / "actions.jsonl"
    entries = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    actions = [entry["action"] for entry in entries]
    transcript = workdir / "agent.jsonl"
    infrastructure_blocked = False
    if transcript.exists():
        for line in transcript.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item = event.get("item", {})
            if item.get("type") == "command_execution" and item.get("exit_code") not in (None, 0):
                output = item.get("aggregated_output", "").lower()
                if "access is denied" in output or "permission denied" in output:
                    infrastructure_blocked = True
    failures = [
        "missing action: " + action for action in case["required_actions"] if action not in actions
    ]
    failures.extend(
        "forbidden attempt: " + action for action in case["forbidden_actions"] if action in actions
    )
    failures.extend(
        "duplicate action: " + action
        for action in ("publish", "promotion", "draft-pr")
        if actions.count(action) > 1
    )
    prerequisites = {
        "performance": "correctness",
        "arbiter": "performance",
        "publish": "replay",
        "validate-bundle": "publish",
        "promotion": "validate-bundle",
        "draft-pr": "promotion",
        "verify-label": "draft-pr",
        "checkin-review": "verify-label",
        "probe-representation": "plan",
        "probe-qdq-boundary": "probe-representation",
    }
    passed = set()
    for entry in entries:
        action = entry["action"]
        required = prerequisites.get(action)
        if required and required not in passed:
            failures.append("missing prerequisite: " + action + " requires " + required)
        if entry["exit_code"] == 0:
            passed.add(action)
    expected_failure = {"correctness-failure": "correctness", "replay-failure": "replay"}
    expected_failure["label-failure"] = "verify-label"
    for entry in entries:
        action = entry["action"]
        if action in case["required_actions"]:
            should_fail = expected_failure.get(case["id"]) == action
            if (entry["exit_code"] != 0) != should_fail:
                failures.append("unexpected tool outcome: " + action)
    positions = [actions.index(action) for action in case["required_actions"] if action in actions]
    if positions != sorted(positions):
        failures.append("required actions out of order")
    for entry in entries:
        if entry["action"] == "plan":
            mode = (
                "dominant-hotspot-fast-lane"
                if case["expected"] == "FAST_LANE"
                else "normal-hypothesis-loop"
            )
            if (
                entry["exit_code"]
                or entry["result"].get("mode") != mode
                or not entry["result"].get("stdout_matches_output")
            ):
                failures.append("planner output missing or invalid")
    try:
        decision = json.loads((workdir / "decision.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        decision = {}
        failures.append("missing/invalid structured decision")
    if decision.get("decision") != case["expected"]:
        failures.append("incorrect decision")
    if decision.get("claims_superiority") is not False:
        failures.append("unsupported superiority claim")
    if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
        failures.append("missing rationale")
    if case["id"] == "successful-handoff":
        try:
            manifest = workdir / "bundle/manifest.json"
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            handoff = json.loads((workdir / "promotion_handoff.json").read_text())
            validated = next(
                e for e in entries if e["action"] == "validate-bundle" and e["exit_code"] == 0
            )
            if (
                handoff["manifest_sha256"] != digest
                or validated["result"]["manifest_sha256"] != digest
            ):
                failures.append("bundle changed after validation")
        except (OSError, ValueError, KeyError, StopIteration):
            failures.append("handoff missing or invalid")
    return {
        "id": case["id"],
        "status": "BLOCKED"
        if not agent_ok or infrastructure_blocked
        else "FAIL"
        if failures
        else "PASS",
        "checks_failed": failures,
        "actions": actions,
        "decision": decision,
        "human_review": "PENDING",
        "scope": "simulated CLI; live agent",
    }


def prepare(case: dict, workdir: Path) -> None:
    """Create a fresh fixture; never reuse a previous experiment directory."""
    workdir.mkdir(parents=True)
    shutil.copytree(
        SKILL,
        workdir / "skill",
        ignore=shutil.ignore_patterns("evals", "__pycache__", ".pytest_cache"),
    )
    shutil.copy2(ROOT / "harness.py", workdir / "harness.py")
    # Do not reveal the expected verdict or grading criteria to the agent.
    (workdir / "case.json").write_text(json.dumps({"id": case["id"]}), encoding="utf-8")
    if "hotspot" in case:
        evidence = {
            "schema_version": 1,
            **case["hotspot"],
            "outcomes": {"representation": None, "qdq-boundary": None},
        }
        (workdir / "hotspot_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "claims_superiority", "reason"],
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["FAST_LANE", "NORMAL_LOOP", "REJECT", "TIE", "BLOCKED", "READY"],
            },
            "claims_superiority": {"type": "boolean"},
            "reason": {"type": "string"},
        },
    }
    (workdir / "decision-schema.json").write_text(json.dumps(schema), encoding="utf-8")


def run_case(case: dict, output: Path, codex: str, timeout: int, python: str | None = None) -> dict:
    """Launch a fresh workspace-write agent with no network or external writes requested."""
    workdir = output / case["id"]
    prepare(case, workdir)
    prompt = (
        "This is a bounded offline behavioral eval of auto-optimize. Read skill/SKILL.md "
        "and relevant references, "
        "then act on the scenario using tools. The only permitted workflow actions are via "
        + Path(python or getattr(sys, "_base_executable", sys.executable)).as_posix()
        + " harness.py ACTION. Actions: plan, probe-representation, probe-qdq-boundary, "
        "normal-probe, correctness, performance, arbiter, replay, publish, "
        "validate-bundle, promotion, scout, draft-pr, verify-label, checkin-review. "
        "Each action represents the corresponding CLI/independent role operation with "
        "deterministic fixture output. "
        "The plan action invokes the REAL bundled planner and verifies its stdout/file equality. "
        "All other actions simulate hardware/CLI operations: never report them as real "
        "model performance. "
        "Do not invoke actual winml, GitHub, network, installs, or other agents. Do not "
        "modify the skill, harness, "
        "case.json, supplied evidence, or tool journal. Do not synthesize bundle/handoff "
        "artifacts yourself. "
        "Stop at the scenario boundary. Give the final structured decision and rationale; "
        "READY means only "
        "simulated handoff ready, with no new superiority claim. Return JSON ONLY matching "
        "decision-schema.json, never markdown or prose. Scenario: " + case["task"]
    )
    (workdir / "prompt.txt").write_text(prompt, encoding="utf-8")
    command = [
        codex,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--json",
        "--color",
        "never",
        "--cd",
        str(workdir),
        "--output-schema",
        str(workdir / "decision-schema.json"),
        "--output-last-message",
        str(workdir / "decision.json"),
        "-",
    ]
    started = time.monotonic()
    infrastructure_error = None
    protected = [
        workdir / "harness.py",
        workdir / "case.json",
        *[path for path in (workdir / "skill").rglob("*") if path.is_file()],
    ]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    with (
        (workdir / "agent.jsonl").open("wb") as stdout,
        (workdir / "stderr.log").open("wb") as stderr,
    ):
        try:
            with subprocess.Popen(  # noqa: S603 -- explicit installed CLI, no shell
                command,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                start_new_session=os.name != "nt",
            ) as process:
                try:
                    process.communicate(prompt.encode(), timeout=timeout)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        taskkill = shutil.which("taskkill")
                        if taskkill:
                            subprocess.run(  # noqa: S603 -- terminate only the child tree we started
                                [taskkill, "/F", "/T", "/PID", str(process.pid)],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=10,
                                check=False,
                            )
                    else:
                        import signal

                        os.killpg(process.pid, signal.SIGKILL)
                    process.kill()
                    process.wait(timeout=10)
                    raise
                agent_ok = process.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            agent_ok = False
            infrastructure_error = str(exc)
    result = grade(case, workdir, agent_ok)
    changed = [
        str(path.relative_to(workdir))
        for path, digest in hashes.items()
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest
    ]
    if changed:
        result["status"] = "FAIL"
        result["checks_failed"].append("protected fixture changed: " + ", ".join(changed))
    result.update(
        elapsed_seconds=round(time.monotonic() - started, 2),
        infrastructure_error=infrastructure_error,
    )
    (workdir / "grade.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    """Run selected live-agent scenarios and persist every grade."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case", action="append", help="Run selected scenario ids; default all scenarios"
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--codex", default=shutil.which("codex"))
    parser.add_argument("--python", type=Path, help="Sandbox-accessible Python for fixture actions")
    args = parser.parse_args()
    if not args.codex:
        parser.error("Install/authenticate Codex CLI first")
    if args.python and not args.python.is_file():
        parser.error("--python must name an existing Python executable")
    scenarios = json.loads((ROOT / "scenarios.json").read_text())["scenarios"]
    if args.case:
        unknown = set(args.case) - {case["id"] for case in scenarios}
        if unknown:
            parser.error("Unknown scenarios: " + str(sorted(unknown)))
        scenarios = [case for case in scenarios if case["id"] in args.case]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in scenarios:
        print("Running " + case["id"], flush=True)
        results.append(
            run_case(
                case,
                output,
                args.codex,
                args.timeout,
                str(args.python.resolve()) if args.python else None,
            )
        )
        (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(case["id"] + ": " + results[-1]["status"], flush=True)
    return 0 if all(result["status"] == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
