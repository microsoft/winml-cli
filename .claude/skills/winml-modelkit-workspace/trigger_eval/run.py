"""Trigger eval runner — Pillar 1.

For each query in queries.json, ask a judge LLM (here, a Claude subagent
through the parent's Agent tool — invoked manually for this MVP) whether
the skill's description would load for the user prompt. Compare against
should_trigger label.

Out scope: this script doesn't drive the subagent itself (parent agent does
that). It just provides the prompt template and the grader.

Usage:
    1. Parent calls render_judge_prompt() to build a single prompt with the
       skill description + all 20 queries
    2. Parent spawns a subagent with that prompt; subagent answers Y/N for each
    3. Parent saves subagent's JSON output as judge_responses.json
    4. python run.py --grade  -- to score against ground truth
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SKILL = Path(r"C:/repo/WinML-ModelKit/.claude/skills/winml-modelkit/SKILL.md")
QUERIES = HERE / "queries.json"
RESPONSES = HERE / "judge_responses.json"
RESULTS = HERE / "results.json"


def extract_description(skill_path: Path) -> str:
    """Pull the `description:` value from the SKILL.md YAML frontmatter."""
    text = skill_path.read_text(encoding="utf-8")
    m = re.search(r"^---\s*\n(.+?)\n---", text, re.DOTALL)
    if not m:
        raise ValueError("No YAML frontmatter found in SKILL.md")
    fm = m.group(1)
    # description may span multiple lines until next top-level key; grab everything after "description:"
    dm = re.search(r"^description:\s*(.+?)(?=^\w+:|\Z)", fm, re.MULTILINE | re.DOTALL)
    if not dm:
        raise ValueError("No `description:` field found in frontmatter")
    return dm.group(1).strip()


def render_judge_prompt() -> str:
    """Render the prompt for a judge subagent to score all queries at once."""
    description = extract_description(SKILL)
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
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


def grade() -> dict:
    """Compare judge_responses.json against queries.json labels."""
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    if not RESPONSES.exists():
        raise FileNotFoundError(f"{RESPONSES} not found — run the judge subagent first")
    responses = json.loads(RESPONSES.read_text(encoding="utf-8"))

    by_id = {r["id"]: r for r in responses}
    rows = []
    correct = 0
    false_pos = 0  # judge said YES, label was NO (over-trigger)
    false_neg = 0  # judge said NO, label was YES (under-trigger)

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
            "id": i,
            "query": q["query"],
            "expected": q["should_trigger"],
            "predicted": predicted,
            "correct": is_correct,
            "reason": r.get("reason", ""),
            "rationale": q.get("rationale", ""),
        })

    n = len(queries)
    accuracy = correct / n if n else 0.0
    n_pos = sum(1 for q in queries if q["should_trigger"])
    n_neg = n - n_pos
    recall = (n_pos - false_neg) / n_pos if n_pos else 0.0
    specificity = (n_neg - false_pos) / n_neg if n_neg else 0.0

    return {
        "total": n,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "false_positives_over_trigger": false_pos,
        "false_negatives_under_trigger": false_neg,
        "recall_on_positives": round(recall, 4),
        "specificity_on_negatives": round(specificity, 4),
        "rows": rows,
    }


def main(argv: list[str]) -> int:
    if "--render-prompt" in argv:
        print(render_judge_prompt())
        return 0
    if "--grade" in argv:
        report = grade()
        RESULTS.write_text(json.dumps(report, indent=2))
        # Human-readable summary
        print(f"Trigger eval results:")
        print(f"  Accuracy: {report['correct']}/{report['total']} = {report['accuracy']:.0%}")
        print(f"  Over-trigger (false positives):  {report['false_positives_over_trigger']}")
        print(f"  Under-trigger (false negatives): {report['false_negatives_under_trigger']}")
        print(f"  Recall on should-trigger:        {report['recall_on_positives']:.0%}")
        print(f"  Specificity on should-not-trigger: {report['specificity_on_negatives']:.0%}")
        print()
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
