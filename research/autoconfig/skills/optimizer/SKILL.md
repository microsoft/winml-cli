---
name: optimizer
description: >
  Use this sub-skill (driven by orchestrator) to RUN one winml-cli config
  hypothesis and produce raw measurements. It does winml build, a Phase A 200-iter
  screen with a CV stability gate and early-exit, a Phase B full bench (3 sessions x
  1000 iters with cool-down), and a winml eval accuracy check. It makes no keep/discard
  decision — it only returns benchmark + accuracy data for the reviewer to judge.
---

# optimizer

The Optimizer is the **"run it"** sub-skill of the autoconfig loop (Phase 2). It
turns one hypothesis into measurements. Mirrors the `Optimizer` class in
`skills/orchestrator/autoconfig.py` and the Optimizer box in
`research/autoconfig/docs/autoconfig_diagram.html`.

**Implementation in this folder:** `bench_utils.py` (the shared bench primitives —
`bench_screen`, `bench_full`, `SessionManager`, and the `ThroughputOnly` verdict
policy the Reviewer consumes).

## When to use

Invoked by `orchestrator` after `explorer` yields a
hypothesis. Not used standalone.

## Inputs

- The hypothesis config delta from the Explorer (applied to the base config).
- Build target: model id, EP, device (held on the Optimizer; thresholds are module constants).

## Procedure

1. **Build** — write `config.json`, run `winml build -c ... --ep <ep> --device <device> --no-quant --no-compile`. Abort the hypothesis on non-zero exit.
2. **Phase A — screen** (`SCREEN_ITERS = 200`):
   - Run `bench_screen`; reject as unstable if `CV > SCREEN_CV_MAX (0.10)` (thermal/scheduling noise — cool device and retry later).
   - The orchestrator early-exits (skips Phase B) if screen improvement vs baseline < 1% (`SCREEN_PASS_MIN_IMPROVEMENT_PCT`), saving 25–90 min per dead hypothesis.
3. **Phase B — full bench** (`FULL_SESSIONS = 3` x `FULL_ITERS = 1000`, `COOL_DOWN_S = 60`):
   - Returns one p50 per session; the loop uses the median across sessions (DVFS-aware averaging, npu-007).
4. **Accuracy** — run `winml eval --samples 50`; parse top-1 / cosine accuracy. Latency comes from the bench, never from eval.

## Outputs

- `screen_p50`, `screen_cv` (Phase A).
- `full_p50s` list + median p50 (Phase B).
- `accuracy` (or None when the model/eval is unavailable).

All handed to `reviewer` — the Optimizer never decides KEEP/DISCARD.

## Constraints

- No hardcoded architecture logic; EP/device come from the orchestrator/Insight.
- Phase B only runs when Phase A is stable and shows promise (cost control).
- Measurements are session-level; the Optimizer never collapses them to a single point estimate before review.
