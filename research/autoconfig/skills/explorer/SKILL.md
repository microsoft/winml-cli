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

- `hypothesis_pool` — the full OFAT search grid from the orchestrator: from a FP32
  baseline, one factor varied at a time — opset (17–21), quant precision
  (fp32/fp16/int8/int16/w8a16), or one single graph pass — as
  `(label, patch_fn, dimension)` triples (~74 combinations). The Explorer
  prunes/reorders it; it does not generate it.
- `kb` — confirmed `ep_device_knowledge/<ep>_<device>.json` rules, especially `skip_passes` hard-blocks.
- `insight` — Phase 1 output: `skip_set` (passes to prune for this model) + `priority_boosts` (per-label ranking weight).

## Procedure

1. **Build the priority_queue** — stable-sort the hypothesis pool by descending
   Insight `priority_boosts` (model-aware ranking; ties keep pool order).
2. **Pop the next hypothesis** from the queue.
3. **Skip-check before yielding** (`skip_reason`):
   - KB hard-block: if the candidate's flags match a confirmed `skip_passes` rule, skip with that rule as the reason (e.g. npu-006 conv-fusion block when Conv% > 20%).
   - Insight skip_set: if the label is in `insight.skip_set`, skip with "Insight Engine: <label>".
   - Otherwise, yield the hypothesis to the Optimizer.

### Graph-presence pruning (`skip_set` source)

The Insight Engine pre-estimates, from the **baseline graph analysis**, which graph
passes can actually fire. For each `graph_pass` hypothesis it checks whether the
pass's required pattern is present (`_pass_can_fire` over the detected
`fusion_candidates`):

- **present** → the pass is kept and gets a priority boost proportional to the
  candidate count (try the promising passes first).
- **confidently absent** (e.g. no Conv→BN subgraph for `conv_bn_fusion`, no
  Softmax for `attention_fusion`) → the label is added to `skip_set` and **cut** —
  there is nothing to fuse, so benchmarking it would be wasted.
- **not statically estimable** → left in the queue for the empirical search
  (no false cuts).

This is complemented at build time by the orchestrator's runtime no-op check
(`graph_is_noop`): if a pass that survived pruning still produces a graph
identical to the baseline, that iteration is discarded before screen/bench.

## Outputs

- The next `(label, config-delta, dimension)` to run, **or**
- A skip decision with a human-readable reason (logged, not benchmarked).

## Constraints

- Pruning is architecture-driven via Insight/KB, never via hardcoded model names.
- Explorer must be cheap and deterministic — no winml build/perf calls here.
- A confirmed KB hard-block always wins over a priority boost (safety before speed).
