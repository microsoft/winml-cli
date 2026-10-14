# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Offline CLI simulator for live-agent skill evaluations; never runs hardware or GitHub."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ACTIONS = (
    "scout",
    "draft-pr",
    "verify-label",
    "checkin-review",
    "plan",
    "probe-representation",
    "probe-qdq-boundary",
    "normal-probe",
    "correctness",
    "performance",
    "arbiter",
    "replay",
    "publish",
    "validate-bundle",
    "promotion",
)


def digest(path: Path) -> str:
    """Hash a fixture without interpreting it as evidence from real hardware."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(workdir: Path, action: str) -> tuple[int, dict]:
    """Execute one simulated action and keep an append-only attempt journal."""
    case = json.loads((workdir / "case.json").read_text(encoding="utf-8"))
    journal = workdir / "actions.jsonl"
    previous = (
        [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    )
    successful = {row["action"] for row in previous if row["exit_code"] == 0}
    result = {"simulation": True, "status": "pass"}
    code = 0
    if action in {"publish", "promotion", "draft-pr"} and action in successful:
        code, result = 2, {"status": "blocked", "reason": "duplicate action"}
    elif action == "plan":
        planner = workdir / "skill" / "scripts" / "plan_hotspot.py"
        proc = subprocess.run(  # noqa: S603 -- fixed bundled planner, no shell
            [
                sys.executable,
                str(planner),
                str(workdir / "hotspot_evidence.json"),
                "--output",
                str(workdir / "hotspot_plan.json"),
            ],
            capture_output=True,
            check=False,
        )
        code = proc.returncode
        if code:
            result = {"status": "error", "stderr": proc.stderr.decode(errors="replace")}
        else:
            result = json.loads(proc.stdout)
            result["stdout_matches_output"] = (
                proc.stdout == (workdir / "hotspot_plan.json").read_bytes()
            )
    elif action.startswith("probe-"):
        if "plan" not in successful:
            code, result = 2, {"status": "blocked", "reason": "plan missing"}
        else:
            result["outcome"] = "DISCARD"
    elif action == "correctness":
        if case["id"] == "correctness-failure":
            code, result = 1, {"status": "fail", "max_abs": 0.25, "frozen_limit": 0.001}
    elif action == "performance":
        if "correctness" not in successful:
            code, result = 2, {"status": "blocked", "reason": "correctness gate missing"}
        else:
            result.update({"gain_pct": 1.0, "ci_low_pct": -1.2, "ci_high_pct": 2.4})
    elif action == "arbiter":
        result.update({"verdict": "STATISTICAL_TIE", "superiority_proven": False})
    elif action == "replay":
        if case["id"] == "replay-failure":
            code, result = (
                7,
                {"status": "fail", "stage": "build", "reason": "native compiler failure"},
            )
        else:
            replay = workdir / "clean-replay"
            replay.mkdir(exist_ok=True)
            (replay / "champion.fixture").write_bytes(b"simulated-model-v1")
            result["clean_directory"] = str(replay)
    elif action == "publish":
        if "replay" not in successful:
            code, result = 2, {"status": "blocked", "reason": "replay missing or failed"}
        else:
            bundle = workdir / "bundle"
            bundle.mkdir(exist_ok=True)
            (bundle / "champion.fixture").write_bytes(
                (workdir / "clean-replay/champion.fixture").read_bytes()
            )
            manifest = {"simulation": True, "champion_sha256": digest(bundle / "champion.fixture")}
            (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result["manifest_sha256"] = digest(bundle / "manifest.json")
    elif action == "validate-bundle":
        if "publish" not in successful:
            code, result = 2, {"status": "blocked", "reason": "bundle missing"}
        else:
            bundle = workdir / "bundle"
            manifest = json.loads((bundle / "manifest.json").read_text())
            if manifest["champion_sha256"] != digest(bundle / "champion.fixture"):
                code, result = 2, {"status": "fail", "reason": "champion hash mismatch"}
            else:
                result["manifest_sha256"] = digest(bundle / "manifest.json")
    elif action == "promotion":
        if "validate-bundle" not in successful:
            code, result = 2, {"status": "blocked", "reason": "validated bundle required"}
        else:
            result["manifest_sha256"] = digest(workdir / "bundle/manifest.json")
            (workdir / "promotion_handoff.json").write_text(json.dumps(result), encoding="utf-8")
    elif action in {"draft-pr", "verify-label", "checkin-review"}:
        required = {
            "draft-pr": "promotion",
            "verify-label": "draft-pr",
            "checkin-review": "verify-label",
        }[action]
        if required not in successful:
            code, result = 2, {"status": "blocked", "reason": required + " required"}
        elif action == "verify-label" and case["id"] == "label-failure":
            code, result = 2, {"status": "blocked", "reason": "label missing"}
    elif action == "scout":
        result["verdict"] = "NO_MATERIAL_OMISSION"
    elif action not in ACTIONS:
        code, result = 2, {"status": "blocked", "reason": "unknown action"}
    entry = {"action": action, "exit_code": code, "result": result}
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + chr(10))
    return code, result


def main() -> int:
    """Dispatch one offline action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    args = parser.parse_args()
    code, result = invoke(Path.cwd(), args.action)
    print(json.dumps(result))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
