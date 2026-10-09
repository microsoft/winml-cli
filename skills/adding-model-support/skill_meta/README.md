# Skill Meta-Findings

Findings about **this skill itself** — not about any particular model family.

The **learner** agent ([`../agents/learner.md`](../agents/learner.md)) writes here as part of
Step 4b (methodology learnings); the **reviewer** ([`../agents/reviewer.md`](../agents/reviewer.md))
audits that the entry ships with its paired skill edit in the same pushed Lane A run.

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
  work — record the asymmetry so the planner ([`../agents/planner.md`](../agents/planner.md))
  can warn about it.

## When NOT to write here

- A property of one specific model or one HF `model_type` → that goes in
  `model_knowledge/<family>.json`.
- A reusable property of one execution provider → that goes in the separately
  installed `auto-optimize` skill's knowledge base instead.

## Schema

Model-like empirical findings use the per-family skeleton (see
[`../model_knowledge/_template.json`](../model_knowledge/_template.json)), with
`_meta.family = "_meta"`, a `purpose`, and `"n/a"` for model-only tier fields where
appropriate. Methodology-change records may instead use the event schema established
by `_meta-040` onward: `trigger`, `what_was_done`, `why_it_matters`, `rule_change`, and
`patch_site`, plus `scope`, mechanism fields, relationships, resolution, and date.
Do not add empty model-only fields to event records merely for shape uniformity.

## Files

- [`findings.json`](./findings.json) — current meta-findings about `adding-model-support`
  (path drift, encoder-decoder recipe gap, first-of-task-family asymmetry, etc.)
