"""Aggregate one response-eval run into a benchmark + per-case comparisons.

Reads each `eval-<case>/{with_skill,without_skill}/run-1/grading.json` + the matching
response.md, plus the canonical case prompts from `response_eval/cases.json`. Produces:

  <run>/benchmark.md            run-level summary table (human-readable)
  <run>/benchmark.json          machine-readable shadow of the same
  <run>/eval-<case>/comparison.md   per-case side-by-side: prompt + both responses +
                                    inline assertion table with evidence

No external tool dependencies. Run after `grade.py` for the run. Idempotent —
re-running overwrites generated files.

Usage:
    python aggregate.py                    # latest run
    python aggregate.py <UTC-datetime>     # specific run
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # response_eval/
RUNS_ROOT = HERE / "runs"
CASES_PATH = HERE / "cases.json"
DATETIME_RE = re.compile(r"^\d{8}-\d{6}$")


def latest_run() -> Path | None:
    if not RUNS_ROOT.exists():
        return None
    candidates = sorted(
        c for c in RUNS_ROOT.iterdir()
        if c.is_dir() and DATETIME_RE.match(c.name)
    )
    return candidates[-1] if candidates else None


def _cell(s: str, max_len: int = 220) -> str:
    """Markdown table-cell-safe: collapse newlines, escape pipes, truncate."""
    s = s.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def render_comparison(case_dir: Path, case_def: dict) -> str:
    lines: list[str] = []
    lines.append(f"# {case_dir.name}\n")
    lines.append("## Prompt\n")
    prompt = case_def.get("prompt", "(prompt not found in cases.json)")
    for ln in prompt.splitlines() or [prompt]:
        lines.append(f"> {ln}")
    lines.append("")

    for config, label in [("with_skill", "with_skill"), ("without_skill", "without_skill (baseline)")]:
        cfg_dir = case_dir / config / "run-1"
        grading_path = cfg_dir / "grading.json"
        response_path = cfg_dir / "outputs" / "response.md"

        if not grading_path.exists():
            lines.append(f"## {label}\n")
            lines.append("_(no grading.json — skipped)_\n")
            continue

        grading = json.loads(grading_path.read_text(encoding="utf-8"))
        s = grading.get("summary", {})
        passed_n, total_n = s.get("passed"), s.get("total")
        flag = "" if passed_n == total_n else " ⚠"
        lines.append(f"## {label} — {passed_n}/{total_n}{flag}\n")

        if response_path.exists():
            lines.append("### Response\n")
            lines.append(response_path.read_text(encoding="utf-8").rstrip())
            lines.append("")

        lines.append("### Grading\n")
        lines.append("| | Assertion | Result | Evidence |")
        lines.append("|---|---|---|---|")
        for exp in grading.get("expectations", []):
            mark = "✓" if exp["passed"] else "✗"
            result = "PASS" if exp["passed"] else "**FAIL**"
            text = _cell(exp.get("text", ""), 180)
            evidence = _cell(exp.get("evidence", ""), 200)
            lines.append(f"| {mark} | {text} | {result} | {evidence} |")
        lines.append("\n---\n")

    return "\n".join(lines)


def render_benchmark(run_dir: Path, summary: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Response eval — {run_dir.name}\n")

    ws = summary["with_skill"]
    bs = summary["without_skill"]
    delta_pp = (ws["pass_rate"] - bs["pass_rate"]) * 100

    lines.append("## Overall\n")
    lines.append("| Metric | with_skill | baseline | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Pass rate | **{ws['pass_rate']*100:.1f}%** "
        f"({ws['passed']}/{ws['total']}) | "
        f"{bs['pass_rate']*100:.1f}% ({bs['passed']}/{bs['total']}) | "
        f"{delta_pp:+.1f}pp |"
    )
    lines.append("")

    lines.append("## Per case\n")
    lines.append("| Case | with_skill | baseline | Details |")
    lines.append("|---|---|---|---|")
    for c in summary["cases"]:
        ws_c = c.get("with_skill", {})
        bs_c = c.get("without_skill", {})
        ws_str = f"{ws_c.get('passed', '?')}/{ws_c.get('total', '?')}"
        bs_str = f"{bs_c.get('passed', '?')}/{bs_c.get('total', '?')}"
        ws_flag = "" if ws_c.get("passed") == ws_c.get("total") else " ⚠"
        link = f"[comparison]({c['id']}/comparison.md)"
        lines.append(f"| `{c['id']}` | {ws_str}{ws_flag} | {bs_str} | {link} |")
    lines.append("")

    if any(c.get("with_skill", {}).get("passed") != c.get("with_skill", {}).get("total")
           for c in summary["cases"]):
        lines.append("Cases marked ⚠ have failing assertions — open the linked `comparison.md` for response text and per-assertion evidence.\n")

    return "\n".join(lines)


def aggregate(run_dir: Path) -> dict:
    cases_doc = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    case_defs = {f"eval-{c['name']}": c for c in cases_doc.get("evals", [])}

    case_results: list[dict] = []
    totals = {
        "with_skill": {"passed": 0, "total": 0},
        "without_skill": {"passed": 0, "total": 0},
    }

    for case_dir in sorted(run_dir.glob("eval-*")):
        case_id = case_dir.name
        case_def = case_defs.get(case_id, {})

        comparison_md = render_comparison(case_dir, case_def)
        (case_dir / "comparison.md").write_text(comparison_md, encoding="utf-8")

        case_summary: dict = {"id": case_id}
        for config in ("with_skill", "without_skill"):
            grading_path = case_dir / config / "run-1" / "grading.json"
            if not grading_path.exists():
                continue
            g = json.loads(grading_path.read_text(encoding="utf-8"))
            s = g.get("summary", {})
            case_summary[config] = {
                "passed": s.get("passed", 0),
                "total": s.get("total", 0),
                "pass_rate": s.get("pass_rate", 0.0),
            }
            totals[config]["passed"] += s.get("passed", 0)
            totals[config]["total"] += s.get("total", 0)
        case_results.append(case_summary)

    for config in ("with_skill", "without_skill"):
        t = totals[config]["total"]
        totals[config]["pass_rate"] = round(totals[config]["passed"] / t, 4) if t else 0.0

    summary = {
        "run": run_dir.name,
        "with_skill": totals["with_skill"],
        "without_skill": totals["without_skill"],
        "cases": case_results,
    }
    (run_dir / "benchmark.md").write_text(render_benchmark(run_dir, summary), encoding="utf-8")
    (run_dir / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str]) -> int:
    if argv:
        run_dir = RUNS_ROOT / argv[0]
        if not run_dir.exists():
            raise SystemExit(f"Run not found: {run_dir}")
    else:
        run_dir = latest_run()
        if not run_dir:
            raise SystemExit("No run directories found.")

    summary = aggregate(run_dir)
    ws = summary["with_skill"]
    bs = summary["without_skill"]
    delta = (ws["pass_rate"] - bs["pass_rate"]) * 100
    print(f"Aggregated {run_dir.name}:")
    print(f"  with_skill: {ws['passed']}/{ws['total']} ({ws['pass_rate']*100:.1f}%)")
    print(f"  baseline:   {bs['passed']}/{bs['total']} ({bs['pass_rate']*100:.1f}%)")
    print(f"  delta:      {delta:+.1f}pp")
    print(f"  Output:     {run_dir}/benchmark.md + per-case comparison.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
