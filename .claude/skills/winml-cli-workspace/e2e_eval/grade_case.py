"""Grade a single E2E trial: read trial directory + final agent message, run outcome assertions.

A "trial" is one independent agent run for a case. Pass@K is computed across trials by
aggregating multiple trial-level grading.json files (see archive_run.py).

Usage:
    python grade_case.py <case_id> <trial_dir> <tool_uses> <duration_ms>

    <trial_dir> contains:
        agent_summary.md       (required) the agent's final user-facing message
        <agent-produced files> (optional) artifacts the agent wrote during the run

Writes:
    <trial_dir>/grading.json   the outcome-assertion report for this single trial
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

E2E = Path(__file__).parent
CASES = E2E / "cases.json"


def grade(case_id: str, trial_dir: Path, summary_text: str, tool_uses: int, duration_ms: int) -> dict:
    cases = json.loads(CASES.read_text())["cases"]
    case = next((c for c in cases if c["id"] == case_id), None)
    if not case:
        raise ValueError(f"unknown case: {case_id}")
    exp = case["expected_outcomes"]
    expectations: list[dict] = []

    # Artifact existence — search within the trial directory
    if exp.get("artifact_glob"):
        files = list(trial_dir.rglob(exp["artifact_glob"])) if trial_dir.exists() else []
        # Exclude grading.json itself from artifact matches if pattern is broad
        files = [f for f in files if f.name != "grading.json"]
        passed = len(files) > 0
        expectations.append({
            "id": "artifact-exists",
            "text": f"At least one file matching `{exp['artifact_glob']}` exists in trial directory.",
            "passed": passed,
            "evidence": f"matched {len(files)} file(s)" + (f": {[f.name for f in files[:3]]}" if files else ""),
        })

        if passed and exp.get("artifact_keys"):
            try:
                data = json.loads(files[0].read_text(encoding="utf-8"))
                keys_ok = all(k in str(data) for k in exp["artifact_keys"])
                expectations.append({
                    "id": "artifact-has-keys",
                    "text": f"Artifact contains expected keys: {exp['artifact_keys']}",
                    "passed": keys_ok,
                    "evidence": f"keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}",
                })
            except (json.JSONDecodeError, OSError):
                pass

    # Refusal case: no build artifacts should exist
    if exp.get("should_refuse"):
        build_artifacts = []
        if trial_dir.exists():
            for pat in ["*.onnx", "*build*", "*compiled*"]:
                build_artifacts.extend(trial_dir.rglob(pat))
        no_build = len(build_artifacts) == 0
        expectations.append({
            "id": "no-build-artifacts",
            "text": "Agent refused; no build artifacts produced.",
            "passed": no_build,
            "evidence": f"build-like files: {len(build_artifacts)}" + (f" ({[f.name for f in build_artifacts[:3]]})" if build_artifacts else ""),
        })

    if exp.get("final_message_must_match"):
        m = re.search(exp["final_message_must_match"], summary_text, re.IGNORECASE)
        expectations.append({
            "id": "summary-matches",
            "text": f"Final message matches `{exp['final_message_must_match']}`",
            "passed": bool(m),
            "evidence": f"match: {m.group(0)[:60] if m else 'none'}",
        })

    if exp.get("final_message_must_NOT_match"):
        m = re.search(exp["final_message_must_NOT_match"], summary_text, re.IGNORECASE)
        expectations.append({
            "id": "summary-doesnt-match-forbidden",
            "text": f"Final message does NOT contain forbidden pattern `{exp['final_message_must_NOT_match']}`",
            "passed": not bool(m),
            "evidence": f"forbidden match: {m.group(0)[:60] if m else 'none'}",
        })

    expectations.append({
        "id": "efficient-tool-use",
        "text": f"Tool calls <= {exp['max_tool_uses']}",
        "passed": tool_uses <= exp["max_tool_uses"],
        "evidence": f"tool_uses = {tool_uses}",
    })
    expectations.append({
        "id": "duration-budget",
        "text": f"Wall time <= {exp['max_duration_s']}s",
        "passed": (duration_ms / 1000) <= exp["max_duration_s"],
        "evidence": f"duration = {duration_ms/1000:.1f}s",
    })

    passed = sum(1 for e in expectations if e["passed"])
    total = len(expectations)
    return {
        "case_id": case_id,
        "trial_dir": str(trial_dir).replace("\\", "/"),
        "telemetry": {"tool_uses": tool_uses, "duration_ms": duration_ms},
        "expectations": expectations,
        "summary": {"passed": passed, "total": total, "pass_rate": round(passed / total, 4)},
    }


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    case_id, trial_dir_str, tool_uses, duration_ms = argv
    trial_dir = Path(trial_dir_str)
    summary_file = trial_dir / "agent_summary.md"
    if not summary_file.exists():
        print(f"Missing agent_summary.md in trial dir: {trial_dir}")
        return 2
    summary_text = summary_file.read_text(encoding="utf-8")
    report = grade(case_id, trial_dir, summary_text, int(tool_uses), int(duration_ms))
    (trial_dir / "grading.json").write_text(json.dumps(report, indent=2))
    s = report["summary"]
    print(f"[{case_id} @ {trial_dir.name}] {s['passed']}/{s['total']} ({int(s['pass_rate']*100)}%)")
    for e in report["expectations"]:
        tag = "PASS" if e["passed"] else "FAIL"
        print(f"  [{tag}] {e['id']}: {e['evidence']}")
    return 0 if s["passed"] == s["total"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
