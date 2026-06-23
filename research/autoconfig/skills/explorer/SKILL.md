---
name: explorer
description: >
  Use this sub-skill (driven by orchestrator) to decide WHAT to try next
  in a winml-cli config search. It builds the hypothesis pool, applies confirmed-KB
  hard-blocks and the Phase 1 Insight Engine skip_set to prune dead-end passes, then
  ranks the survivors by Insight priority boost into a priority_queue and yields the
  next hypothesis. It never builds or benchmarks — it only chooses the next experiment.
---

# explorer

The Explorer is the **"what to try next"** sub-skill of the autoconfig loop
(Phase 2). It owns search *order* only. Mirrors the `Explorer` class in
`skills/orchestrator/autoconfig.py` and the Explorer box in
`research/autoconfig/docs/autoconfig_diagram.html`.

**Implementation in this folder:** `analyze_insight.py` (the Phase 1 Insight Engine
that produces the `skip_set` + `priority_boosts` this skill ranks by) and
`analyze_graph.py` (ONNX graph-pattern helper).

## When to use

Invoked by `orchestrator` at the top of each Phase 2 iteration to get
the next candidate config delta. Not used standalone.

## Inputs

- `hypothesis_pool` — list of `(label, patch_fn, dimension)` candidates (opset bumps, EP toggles, graph-optimization passes).
- `kb` — confirmed `ep_knowledge/<ep>.json` rules, especially `skip_passes` hard-blocks.
- `insight` — Phase 1 output: `skip_set` (passes to prune for this model) + `priority_boosts` (per-label ranking weight).

## Procedure

1. **Build the priority_queue** — stable-sort the hypothesis pool by descending
   Insight `priority_boosts` (model-aware ranking; ties keep pool order).
2. **Pop the next hypothesis** from the queue.
3. **Skip-check before yielding** (`skip_reason`):
   - KB hard-block: if the candidate's flags match a confirmed `skip_passes` rule, skip with that rule as the reason (e.g. npu-006 conv-fusion block when Conv% > 20%).
   - Insight skip_set: if the label is in `insight.skip_set`, skip with "Insight Engine: <label>".
   - Otherwise, yield the hypothesis to the Optimizer.

## Outputs

- The next `(label, config-delta, dimension)` to run, **or**
- A skip decision with a human-readable reason (logged, not benchmarked).

## Constraints

- Pruning is architecture-driven via Insight/KB, never via hardcoded model names.
- Explorer must be cheap and deterministic — no winml build/perf calls here.
- A confirmed KB hard-block always wins over a priority boost (safety before speed).
