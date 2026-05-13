"""Archive a single /run-e2e invocation into runs/<run-id>/, aggregating Pass@K across trials.

Each case has K trials (independent agent runs). For each case:
  - Each trial sits at scratch/<case_id>/trial-<N>/ with agent_summary.md, grading.json,
    telemetry.json, and whatever artifacts the agent produced.
  - This script copies each trial's light files into runs/<run-id>/cases/<case_id>/trial-<N>/,
    builds an artifacts_manifest per trial (heavy binaries registered but not duplicated),
    then writes runs/<run-id>/cases/<case_id>/aggregate.json with Pass@K stats.

Outputs:
  runs/<run-id>/report.md
  runs/<run-id>/meta.json
  runs/<run-id>/cases/<case_id>/aggregate.json
  runs/<run-id>/cases/<case_id>/trial-<N>/{agent_summary.md, grading.json, telemetry.json,
                                             artifacts_manifest.json, outputs/...}

Usage:
    python archive_run.py <run_id> --meta meta.json --cases <case_id> [<case_id> ...]
        [--skipped '<json>']
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE / "scratch"
RUNS = HERE / "runs"

ALWAYS_ARCHIVE = {"agent_summary.md", "grading.json", "telemetry.json"}
TEXT_EXTS = {".json", ".md", ".txt", ".log", ".yaml", ".yml", ".csv", ".html"}
BINARY_SIZE_THRESHOLD = 1 * 1024 * 1024  # 1 MB


def build_manifest_and_archive(trial_scratch: Path, trial_archive: Path) -> dict:
    artifacts = []
    outputs = trial_archive / "outputs"
    for f in sorted(trial_scratch.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(trial_scratch)
        if str(rel) in ALWAYS_ARCHIVE:
            continue
        size = f.stat().st_size
        is_text = f.suffix.lower() in TEXT_EXTS
        archived = is_text or size < BINARY_SIZE_THRESHOLD
        if archived:
            dst = outputs / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        artifacts.append({
            "path": str(rel).replace("\\", "/"),
            "size_bytes": size,
            "archived": archived,
            "kind": "text" if is_text else "binary",
        })
    return {"scratch_dir": str(trial_scratch).replace("\\", "/"), "artifacts": artifacts}


def archive_trial(trial_scratch: Path, trial_archive: Path) -> dict:
    """Copy lightweight files + build manifest. Return {trial: N, passed, total, telemetry}."""
    trial_archive.mkdir(parents=True, exist_ok=True)
    for fname in ALWAYS_ARCHIVE:
        src = trial_scratch / fname
        if src.exists():
            shutil.copy2(src, trial_archive / fname)
    manifest = build_manifest_and_archive(trial_scratch, trial_archive)
    (trial_archive / "artifacts_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    trial_summary = {"trial": trial_scratch.name}
    grading_path = trial_archive / "grading.json"
    if grading_path.exists():
        try:
            g = json.loads(grading_path.read_text(encoding="utf-8"))
            s = g.get("summary", {})
            trial_summary["passed"] = s.get("passed")
            trial_summary["total"] = s.get("total")
            trial_summary["pass_rate"] = s.get("pass_rate")
            trial_summary["all_pass"] = (s.get("passed") == s.get("total") and (s.get("total") or 0) > 0)
        except (json.JSONDecodeError, OSError):
            pass
    telemetry_path = trial_archive / "telemetry.json"
    if telemetry_path.exists():
        try:
            t = json.loads(telemetry_path.read_text(encoding="utf-8"))
            trial_summary["tool_uses"] = t.get("tool_uses")
            trial_summary["duration_ms"] = t.get("duration_ms")
        except (json.JSONDecodeError, OSError):
            pass
    return trial_summary


def archive_case(case_id: str, run_dir: Path) -> dict:
    """Archive all trials for a case, compute Pass@K aggregate."""
    case_scratch = SCRATCH / case_id
    case_archive = run_dir / "cases" / case_id
    case_archive.mkdir(parents=True, exist_ok=True)

    summary = {"case_id": case_id, "trials": []}

    if not case_scratch.exists():
        summary["error"] = "scratch directory missing"
        return summary

    trial_scratches = sorted(case_scratch.glob("trial-*"))
    if not trial_scratches:
        summary["error"] = "no trial-* subdirs found in scratch"
        return summary

    for ts in trial_scratches:
        trial_archive = case_archive / ts.name
        summary["trials"].append(archive_trial(ts, trial_archive))

    # Pass@K aggregate
    k = len(summary["trials"])
    passes = sum(1 for t in summary["trials"] if t.get("all_pass"))
    summary["k"] = k
    summary["pass_at_k"] = passes
    summary["pass_at_k_rate"] = round(passes / k, 4) if k else 0.0

    # Per-trial assertion pass rate average (different from Pass@K)
    valid_trials = [t for t in summary["trials"] if t.get("total")]
    if valid_trials:
        avg_pass_rate = sum(t["pass_rate"] for t in valid_trials) / len(valid_trials)
        summary["avg_assertion_pass_rate"] = round(avg_pass_rate, 4)

    (case_archive / "aggregate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def render_report(meta: dict, case_summaries: list[dict], skipped: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"# E2E run — {meta['run_id']}\n")
    lines.append(f"**Started:** {meta.get('started_at_utc', '?')}  ")
    lines.append(f"**Host:** {meta.get('host', '?')}  ")
    lines.append(f"**Skill commit:** `{meta.get('skill_commit', '?')}`  ")
    lines.append(f"**winml:** {meta.get('winml_version', '?')}  ")
    lines.append(f"**Registered EPs:** {', '.join(meta.get('registered_eps', [])) or '(none)'}\n")

    attempted = len(case_summaries)
    full_pass_n = sum(1 for c in case_summaries
                      if c.get("k") and c.get("pass_at_k") == c.get("k"))
    lines.append("## Summary\n")
    lines.append(f"- Cases attempted: {attempted}")
    lines.append(f"- Fully-passing cases (Pass@K = K/K): **{full_pass_n}/{attempted}**")
    if skipped:
        lines.append(f"- Skipped: {len(skipped)}")
    lines.append("")

    if case_summaries:
        lines.append("| Case | Pass@K | Avg assertion pass rate | Trials |")
        lines.append("|------|--------|-------------------------|--------|")
        for c in case_summaries:
            if c.get("error"):
                lines.append(f"| `{c['case_id']}` | ERROR | — | — |")
                continue
            pk = c.get("pass_at_k", 0)
            k = c.get("k", 0)
            avg = c.get("avg_assertion_pass_rate", 0.0)
            trials = c.get("trials", [])
            trial_str = ", ".join(
                f"{t['trial']}={t.get('passed','?')}/{t.get('total','?')}"
                for t in trials
            )
            lines.append(f"| `{c['case_id']}` | **{pk}/{k}** | {avg*100:.0f}% | {trial_str} |")
        lines.append("")

    if skipped:
        lines.append("## Skipped cases\n")
        for s in skipped:
            lines.append(f"- `{s['case_id']}` — {s.get('reason', 'no reason given')}")
        lines.append("")

    lines.append("## Per-case detail\n")
    for c in case_summaries:
        lines.append(f"### `{c['case_id']}`")
        if c.get("error"):
            lines.append(f"- Error: {c['error']}\n")
            continue
        lines.append(f"- Pass@{c['k']}: **{c['pass_at_k']}/{c['k']}**")
        for t in c.get("trials", []):
            tag = "PASS" if t.get("all_pass") else "FAIL"
            tu = t.get("tool_uses", "?")
            dur = t.get("duration_ms", 0) / 1000 if t.get("duration_ms") else "?"
            dur_str = f"{dur:.1f}s" if isinstance(dur, (int, float)) and dur != "?" else dur
            lines.append(f"  - `{t['trial']}` — {tag} ({t.get('passed','?')}/{t.get('total','?')}, {tu} tool calls, {dur_str})")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    run_id = argv[0]
    case_ids: list[str] = []
    skipped: list[dict] = []
    meta_path: Path | None = None

    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok == "--meta":
            meta_path = Path(argv[i + 1])
            i += 2
        elif tok == "--cases":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                case_ids.append(argv[i])
                i += 1
        elif tok == "--skipped":
            skipped = json.loads(argv[i + 1])
            i += 2
        else:
            print(f"Unknown arg: {tok}")
            return 2

    meta: dict = {}
    if meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["run_id"] = run_id
    meta.setdefault("started_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    run_dir = RUNS / run_id
    (run_dir / "cases").mkdir(parents=True, exist_ok=True)

    summaries = [archive_case(cid, run_dir) for cid in case_ids]
    meta["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["cases_eligible"] = case_ids
    meta["cases_skipped"] = skipped
    meta["case_summaries"] = summaries

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(meta, summaries, skipped), encoding="utf-8")

    full_pass = sum(1 for c in summaries
                    if c.get("k") and c.get("pass_at_k") == c.get("k"))
    print(f"Archived to: {run_dir}")
    print(f"Full-pass cases (Pass@K = K/K): {full_pass}/{len(summaries)}  Skipped: {len(skipped)}")
    return 0 if full_pass == len(summaries) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
