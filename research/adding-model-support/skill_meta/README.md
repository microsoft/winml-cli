# Skill Meta-Findings

Findings about **this skill itself** — not about any particular model family.

Lives separately from [`../model_knowledge/`](../model_knowledge/) so that the dialectical
record of "the skill said X and was wrong" doesn't pollute per-model lookups when a
contributor opens `model_knowledge/<family>.json`.

## When to write here

- The skill's own documentation drifted from the codebase (path moved, registry renamed,
  helper deleted) and a contributor was misled.
- A cross-family pattern emerged that is **not** a property of any one model but of the
  framework's coverage — e.g. "no encoder-decoder recipe ships in the repo, every
  seq2seq contributor pays the template cost."
- A skill-axis (Effort / Goal / Outcome) tier turned out to be missing or wrong, and a
  new tier was added.
- A task family in `TASK_REGISTRY` has zero registered models on the export side, so
  the **first** contributor for that task is implicitly doing task-family infrastructure
  work — record the asymmetry so SKILL.md can warn about it.

## When NOT to write here

- A property of one specific model or one HF `model_type` → that goes in
  `model_knowledge/<family>.json`.
- A property of one execution provider → that goes in
  [`research/autoconfig/ep_knowledge/`](../../autoconfig/ep_knowledge/) instead.

## Schema

Same as per-family findings (see [`../model_knowledge/_template.json`](../model_knowledge/_template.json))
with `_meta.family = "_meta"` and a `purpose` field describing what kind of meta-finding
this file collects. `effort_tier_required` and `goal_tier_reached` are `"n/a"` for
methodology findings.

## Files

- [`findings.json`](./findings.json) — current meta-findings about `adding-model-support`
  (path drift, encoder-decoder recipe gap, first-of-task-family asymmetry, etc.)
