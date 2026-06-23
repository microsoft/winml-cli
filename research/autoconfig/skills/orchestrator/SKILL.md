---
name: orchestrator
description: >
  Use this skill as the top-level brain for an automated winml-cli build-config
  search. It runs the full autoconfig lifecycle (Phase 0 Intake, Phase 1 Insight,
  Phase 2 Opt Loop, Phase 3 Outcome) and coordinates three sub-skills —
  explorer (what to try), optimizer (run it), and
  reviewer (judge it) — to find the best EP + opset + graph-optimization
  config for a given model on the current Windows hardware. Owns session state,
  crash-resume, champion tracking, and stop conditions; sub-skills own one phase each.
---

# orchestrator

The Orchestrator is the **main brain** of the autoconfig loop. It does not build,
benchmark, or judge experiments itself — it sequences the phases and delegates the
Phase 2 work to three sub-skills, then aggregates their results into a champion
config plus auditable artifacts.

Reference implementation: `skills/orchestrator/autoconfig.py` (the `main()`
orchestrator wiring the `Explorer`, `Optimizer`, and `Reviewer` classes).
Design spec: `research/autoconfig/docs/autoconfig_diagram.html`.

**Implementation in this folder:** `autoconfig.py` (the `main()` orchestrator plus
the `Explorer` / `Optimizer` / `Reviewer` classes — the runnable reference loop that
the explorer / optimizer / reviewer sub-skills formalize).

## When to use

- "Find the fastest config for this model on my NPU/GPU/CPU"
- "Sweep opset 17–21 and graph optimizations and tell me what actually helps"
- "Run an automated, statistically-honest config search and give me an auditable report"
- Driving a catalog sweep across many models (see `catalog_qnn_sweep.py`)

## Sub-skills it coordinates

| Phase | Sub-skill | Responsibility |
| --- | --- | --- |
| 2 — pick | `explorer` | Build the hypothesis pool, prune with KB hard-blocks + Insight skip_set, rank into a priority_queue, yield the next hypothesis |
| 2 — run | `optimizer` | `winml build` -> Phase A screen (CV gate) -> Phase B full bench -> `winml eval` accuracy; returns raw measurements only |
| 2 — judge | `reviewer` | Apply the ThroughputOnly verdict (`threshold = max(1%, 2x CV)`) -> KEEP / MARGINAL / DISCARD, draft KB entries for real wins |

The orchestrator is the only component that holds global state. Sub-skills are
stateless with respect to each other: Explorer never benchmarks, Optimizer never
decides, Reviewer never builds.

## Lifecycle (the procedure)

**Phase 0 — Intake**
- `winml inspect` the model; resolve `model_type` (architecture family — never hardcode arch names).
- `winml analyze --ep <ep>` for EP compatibility; establish the correctness contract via `winml eval --mode compare` (cosine ~= 1.000 baseline).
- Build the baseline config and record its p50 as the reference.

**Phase 1 — Insight**
- Run the static/graph analyzer to produce a hypothesis pool tailored to the model
  (e.g. Conv% drives the npu-006 conv-fusion hard-block).
- Hand the pool + `skip_set` + `priority_boosts` to the Explorer.

**Phase 2 — Opt Loop** (repeat until a stop condition)
1. Ask **explorer** for the next hypothesis (it pops from the priority_queue and skips KB/Insight-blocked passes).
2. Ask **optimizer** to build + benchmark it (screen early-exits if delta < 1%; full bench is 3x1000 with 60 s cool-down).
3. Ask **reviewer** for the verdict; on KEEP, update the champion.
4. Persist `session.json` atomically (crash-resume) and append the TSV row + experiment.md.

**Phase 3 — Outcome**
- Emit the champion config, an HTML/Markdown report, the per-experiment artifacts, and KB draft entries (`status="draft"`).
- Summarize confirmed findings and any feature requirements surfaced during the run.

## Stop conditions

Stop the Phase 2 loop when **any** holds:
- Objective met (target improvement reached), or
- 30 consecutive DISCARDs (architectural levers exhausted), or
- Priority queue empty, or
- User stops.

## Constraints

- No hardcoded model/architecture logic — all arch reasoning comes from winml `model_type`.
- Accuracy gate (`winml eval`) is mandatory before any KEEP.
- All perf claims use session-level averaging; never report a point estimate as a win.
- KB writes are drafts only; promotion to `confirmed` is a human gate (>=2 models + mechanism understood).
