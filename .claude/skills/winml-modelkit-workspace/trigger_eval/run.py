"""Trigger eval runner — Pillar 1.

Each "round" is a snapshot of (description, queries, judge responses, results) at a
moment in time, stored as `rounds/<UTC-datetime>/`. Top-level files (`queries.json`,
`run.py`, `RUNBOOK.md`) are the canonical source; rounds/ is the history.

Workflow:
  1. `python run.py --new-round`
        creates a new `rounds/<now>/` directory, snapshots the current description
        from SKILL.md + the current queries.json, and renders judge_prompt.txt into it.
  2. The parent agent spawns a judge subagent with the rendered prompt and saves the
     JSON output to `rounds/<that-datetime>/judge_responses.json`.
  3. `python run.py --grade`
        finds the latest round, reads judge_responses.json, compares to queries.json
        labels, writes results.json into the round directory.

Round directory contents:
  description.md            snapshot of SKILL.md description at this round
  queries.json              snapshot of queries.json at this round
  judge_prompt.txt          rendered prompt (gitignored — derived)
  judge_responses.json      judge subagent output
  results.json              grading output
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
SKILL = Path(r"C:/repo/WinML-ModelKit/.claude/skills/winml-modelkit/SKILL.md")
QUERIES = HERE / "queries.json"
ROUNDS_DIR = HERE / "rounds"
DATETIME_RE = re.compile(r"^\d{8}-\d{6}$")


def extract_description(skill_path: Path) -> str:
    text = skill_path.read_text(encoding="utf-8")
    m = re.search(r"^---\s*\n(.+?)\n---", text, re.DOTALL)
    if not m:
        raise ValueError("No YAML frontmatter found in SKILL.md")
    fm = m.group(1)
    dm = re.search(r"^description:\s*(.+?)(?=^\w+:|\Z)", fm, re.MULTILINE | re.DOTALL)
    if not dm:
        raise ValueError("No `description:` field found in frontmatter")
    return dm.group(1).strip()


def latest_round() -> Path | None:
    if not ROUNDS_DIR.exists():
        return None
    candidates = sorted(
        c for c in ROUNDS_DIR.iterdir()
        if c.is_dir() and DATETIME_RE.match(c.name)
    )
    return candidates[-1] if candidates else None


def render_judge_prompt(description: str, queries: list[dict]) -> str:
    lines: list[str] = []
    lines.append("You are simulating how an AI coding-assistant agent decides whether to load a skill.")
    lines.append("")
    lines.append("A skill named `winml-modelkit` is available. Here is its description (everything the agent sees before deciding to load):")
    lines.append("")
    lines.append("---")
    lines.append(description)
    lines.append("---")
    lines.append("")
    lines.append("For each user query below, decide whether loading this skill would help. Answer with **YES** or **NO** — based ONLY on the description above, not on any other knowledge of winml or model deployment.")
    lines.append("")
    lines.append("Return your answers as a JSON array of objects with this exact shape:")
    lines.append('  [{"id": 0, "decision": "YES", "reason": "..."}, {"id": 1, "decision": "NO", "reason": "..."}, ...]')
    lines.append("")
    lines.append("Each `reason` should be one short sentence on what tipped the decision.")
    lines.append("")
    lines.append("Queries:")
    lines.append("")
    for i, q in enumerate(queries):
        lines.append(f"{i}. {q['query']}")
    return "\n".join(lines)


def new_round() -> Path:
    """Create a new round directory with snapshots of description + queries + prompt."""
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    round_dir = ROUNDS_DIR / now
    round_dir.mkdir(parents=True, exist_ok=True)

    description = extract_description(SKILL)
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))

    (round_dir / "description.md").write_text(description + "\n", encoding="utf-8")
    shutil.copy2(QUERIES, round_dir / "queries.json")
    (round_dir / "judge_prompt.txt").write_text(
        render_judge_prompt(description, queries), encoding="utf-8"
    )
    return round_dir


def grade(round_dir: Path) -> dict:
    queries = json.loads((round_dir / "queries.json").read_text(encoding="utf-8"))
    responses_path = round_dir / "judge_responses.json"
    if not responses_path.exists():
        raise FileNotFoundError(f"{responses_path} not found — spawn the judge subagent first")
    responses = json.loads(responses_path.read_text(encoding="utf-8"))

    by_id = {r["id"]: r for r in responses}
    rows = []
    correct = 0
    false_pos = 0
    false_neg = 0
    for i, q in enumerate(queries):
        r = by_id.get(i)
        if not r:
            rows.append({"id": i, "query": q["query"], "expected": q["should_trigger"],
                         "predicted": None, "correct": False, "reason": "no judge response"})
            continue
        predicted = (r["decision"].upper().strip() == "YES")
        is_correct = predicted == q["should_trigger"]
        if is_correct:
            correct += 1
        else:
            if predicted and not q["should_trigger"]:
                false_pos += 1
            elif not predicted and q["should_trigger"]:
                false_neg += 1
        rows.append({
            "id": i, "query": q["query"], "expected": q["should_trigger"],
            "predicted": predicted, "correct": is_correct,
            "reason": r.get("reason", ""), "rationale": q.get("rationale", ""),
        })

    n = len(queries)
    accuracy = correct / n if n else 0.0
    n_pos = sum(1 for q in queries if q["should_trigger"])
    n_neg = n - n_pos
    recall = (n_pos - false_neg) / n_pos if n_pos else 0.0
    specificity = (n_neg - false_pos) / n_neg if n_neg else 0.0

    return {
        "round": round_dir.name,
        "total": n, "correct": correct, "accuracy": round(accuracy, 4),
        "false_positives_over_trigger": false_pos,
        "false_negatives_under_trigger": false_neg,
        "recall_on_positives": round(recall, 4),
        "specificity_on_negatives": round(specificity, 4),
        "rows": rows,
    }


def main(argv: list[str]) -> int:
    if "--new-round" in argv:
        round_dir = new_round()
        print(f"New round: {round_dir.name}")
        print(f"  description.md, queries.json, judge_prompt.txt written to {round_dir}")
        print(f"  Next: spawn judge subagent with judge_prompt.txt, save JSON to {round_dir / 'judge_responses.json'}")
        return 0
    if "--grade" in argv:
        # Allow override: --grade <round-name>
        round_name = None
        for i, a in enumerate(argv):
            if a == "--grade" and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                round_name = argv[i + 1]
        if round_name:
            round_dir = ROUNDS_DIR / round_name
        else:
            round_dir = latest_round()
            if not round_dir:
                raise SystemExit("No rounds found. Run `--new-round` first.")
        report = grade(round_dir)
        (round_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Trigger eval results [{round_dir.name}]:")
        print(f"  Accuracy: {report['correct']}/{report['total']} = {report['accuracy']:.0%}")
        print(f"  Over-trigger (false positives):  {report['false_positives_over_trigger']}")
        print(f"  Under-trigger (false negatives): {report['false_negatives_under_trigger']}")
        print(f"  Recall on should-trigger:        {report['recall_on_positives']:.0%}")
        print(f"  Specificity on should-not-trigger: {report['specificity_on_negatives']:.0%}")
        for r in report["rows"]:
            if not r["correct"]:
                tag = "OVER" if r["predicted"] and not r["expected"] else "UNDER"
                print(f"  [{tag}] q{r['id']}: expected={r['expected']} got={r['predicted']}")
                print(f"         query:  {r['query'][:120]}")
                print(f"         reason: {r['reason'][:200]}")
        return 0 if report["correct"] == report["total"] else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
