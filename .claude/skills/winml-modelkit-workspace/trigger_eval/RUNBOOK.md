# Trigger eval — Runbook

Pillar 1: does the agent decide to load this skill given a user prompt? Iterate description vs. a curated eval set of 20 queries.

This RUNBOOK is **agent-facing** — when a user says "let's iterate trigger" or "add a trigger case", the agent driving the conversation follows these steps.

## What this measures

A skill has a one-line `description` in its YAML frontmatter. That description is the **only** thing the host agent sees when deciding whether to load the skill body. Two failure modes:

- **Under-trigger** — the description is too narrow; relevant user prompts don't load the skill.
- **Over-trigger** — the description is too broad; adjacent unrelated prompts load the skill anyway and the agent gives off-topic advice.

`queries.json` holds a balanced set of should-trigger / should-not-trigger prompts. Grading runs them through a judge LLM with only the description visible and compares to the labels.

## How to iterate

### Step 1 — Read the current state

```
trigger_eval/queries.json        — current 20 queries
trigger_eval/results.json        — last grading run (if any)
```

If `results.json` shows < 100%, look at which queries failed. Each fail is either over-trigger or under-trigger and points at how the description needs to change.

### Step 2 — Decide what to change

Three actions to consider, in priority order:

| Action | When |
|---|---|
| **Add new queries** | The current set feels too easy, or you want to lock in coverage of an edge case the description might not handle. |
| **Edit description** in `.claude/skills/winml-modelkit/SKILL.md` (YAML frontmatter) | The eval shows specific over/under-triggers and the description is the lever. |
| **Both** | Often together — add the case that exposes a gap, then fix description to pass it. |

### Step 3a — Add a query

Append to `queries.json`:

```json
{
  "query": "<real-user-voice prompt, with project context / file paths / casual tone>",
  "should_trigger": true|false,
  "rationale": "<one short sentence on why this is the right label>"
}
```

Design rules for good queries:
- **Real voice**: include backstory, file paths, mixed casing, possible typos. Avoid generic "Format X for Y".
- **Hard negatives** are worth more than easy negatives. A negative should share keywords / domain with positives but have different intent (e.g., "Phi-3" or "winml" appearing in an out-of-scope ask).
- **Mix languages** if the real user base does — the existing set has some Chinese.

### Step 3b — Edit description

Open `.claude/skills/winml-modelkit/SKILL.md`. Edit the `description:` line in the YAML frontmatter at the top. Keep it pushy ("Use this skill whenever the user mentions...") because agents tend to under-trigger by default.

### Step 4 — Render the judge prompt

```bash
python trigger_eval/run.py --render-prompt > trigger_eval/judge_prompt.txt
```

This bundles the current description + all queries from `queries.json` into a single prompt for the judge.

### Step 5 — Spawn the judge subagent

The driving agent uses the Agent tool. Prompt: pipe in `judge_prompt.txt` and instruct the subagent to save its JSON array of decisions to `trigger_eval/judge_responses.json`. Required output schema:

```json
[
  {"id": 0, "decision": "YES"|"NO", "reason": "<one short sentence>"},
  ...
]
```

### Step 6 — Grade

```bash
python trigger_eval/run.py --grade
```

Prints accuracy + per-query failures. Writes `results.json`.

### Step 7 — Iterate

If accuracy < target, look at the failures:
- **Over-trigger**: tighten the description (be more specific about what's in scope).
- **Under-trigger**: broaden the description or add explicit "use this skill when..." cues.

Re-render, re-spawn judge, re-grade.

## Cost per iteration

- One judge subagent call: ~30 seconds
- Single trial (K=1) — see design doc for why this pillar doesn't need K=3

## When to stop iterating

- Accuracy ≥ 95% (19+/20) with no easy misses
- Remaining failures are genuinely borderline (a human would also disagree)
- Diminishing returns: 3 rounds without improvement → stop

## K=1 caveat

This eval is single-trial. If you suspect a query is flaky (the judge gives different answers on different runs), spawn the judge twice and compare. If the labels disagree, that query is borderline — consider rewording it or removing.
