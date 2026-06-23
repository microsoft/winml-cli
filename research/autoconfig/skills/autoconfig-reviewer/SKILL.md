---
name: autoconfig-reviewer
description: >
  Use this sub-skill (driven by autoconfig-orchestrator) to JUDGE the measurements an
  autoconfig-optimizer produced for one hypothesis. It applies the ThroughputOnly
  verdict policy with a noise-aware threshold (max(1% floor, 2x screen-CV)), enforces
  the accuracy floor, and returns KEEP / MARGINAL_KEEP / DISCARD / ACC_FAIL with a
  human-readable rationale. On a real, non-marginal win it drafts a KB entry
  (status="draft") for later human promotion. It never builds or benchmarks.
---

# autoconfig-reviewer

The Reviewer is the **"judge it"** sub-skill of the autoconfig loop (Phase 2). It
turns Optimizer measurements into a verdict. Mirrors the `Reviewer` class in
`research/autoconfig/autoconfig.py` (wrapping `ThroughputOnly` from
`bench_utils.py`) and the Reviewer box in
`research/autoconfig/docs/autoconfig_diagram.html`.

## When to use

Invoked by `autoconfig-orchestrator` after `autoconfig-optimizer` returns
benchmark + accuracy data. Not used standalone.

## Inputs

- `full_p50s` (per-session p50s) and `accuracy` from the Optimizer.
- `screen_cv` (drives the statistical threshold) and the current `baseline_p50`.

## Procedure

1. **Baseline promotion** — if no baseline yet, the first successful full bench median becomes the baseline.
2. **Compute improvement** — `improvement_pct = (baseline - median_p50) / baseline x 100`.
3. **Accuracy gate** — pass requires `accuracy is None or accuracy >= ACCURACY_FLOOR (0.70)`; else `ACC_FAIL`.
4. **Verdict (ThroughputOnly)** — statistically honest threshold `max(MIN_IMPROVEMENT 1%, STAT_BAR 2.0 x screen_CV)`:
   - `KEEP` — improvement > 1.5x threshold.
   - `MARGINAL_KEEP` — improvement between 1x and 1.5x threshold.
   - `DISCARD` — improvement below threshold (noise-level), or
   - `ACC_FAIL` — accuracy below floor.
5. **KB draft** — on a non-marginal KEEP with improvement > 10%, append a `status="draft"` finding to `ep_knowledge/<ep>.json` (de-duplicated per label+model).

## Outputs

- A status string (`keep` / `keep (marginal)` / `discard (...)`) plus the verdict reasoning written into the experiment record.
- Optionally, a KB draft entry for human review.

The Reviewer reports the verdict; the orchestrator owns champion tracking and
applies the verdict to the loop's stop-condition counters.

## Constraints

- Threshold is noise-aware — a delta inside `2x CV` is never reported as a win.
- KB writes are drafts only; promotion to `confirmed` is a human gate (Gate 2: >=2 independent models + mechanism understood).
- No hardcoded architecture logic in the verdict; the policy is model-agnostic.
