# Per-EP Empirical Knowledge Base

Each JSON file stores empirical findings for one EP/device combination.

## ⚠️ CRITICAL EPISTEMICS

These findings are **observational hypotheses, not ground truth**. They were derived
from a small number of experiments on a single model (ConvNext-tiny) on a single device
(Snapdragon X Elite CRD). Every finding carries a `confidence` field and a `falsified_by`
field. Before using a finding to prune a search space, check:

1. **Is the model architecture similar?** (ConvNext ≠ BERT ≠ ResNet)
2. **Is the hardware the same?** (X Elite CRD ≠ X Plus ≠ X1E-80-100)
3. **Is the ORT/QNN SDK version the same?**
4. **Is the mechanism confirmed?** (see `mechanism_confirmed` field)

**Dialectical rule**: A finding that prunes a search dimension must be re-enabled
if a new experiment on a new model/hardware contradicts it. Findings degrade over time
as ORT and QNN SDK versions change.

## ✅ Promotion checklist (before a finding becomes a pruning rule)

These rules exist because of the **npu-001 / MobileViT failure**: a `+26.5%` opset-21
"win" was recorded from a single sweep whose baseline (~12 ms) was silently inflated by
DVFS/thermal throttling. A clean from-scratch rerun (2026-06-22) measured the baseline at
~5.5 ms and the same config at +2.8% — fully within noise. The fake gain came from a
**polluted baseline and a cross-run comparison**, the two least reliable things on a
DVFS NPU. To avoid recording artifacts as findings, a result must clear ALL of these
before its `confidence` is raised above `draft` / before it is used to prune search space:

1. **Paired / same-thermal-window measurement.** Compare a config against its baseline
   measured in the *same* thermal window (interleave A/B/A/B), and compare the
   within-window **delta** — never an absolute baseline carried over from another run.
2. **Clean baseline gate.** Reject the whole comparison if the baseline session-to-session
   CV is high or contains a >2σ spike. A noisy baseline poisons every ratio derived from it.
3. **Effect size > noise floor.** Require `gain% >= 2 × (session-to-session CV)` AND
   non-overlapping session p50 ranges. A sub-5% median win on QNN NPU is noise by default.
   (`catalog_qnn_sweep.py` now emits `best_gain_verdict`: `RELIABLE` /
   `NEUTRAL_WITHIN_NOISE` / `UNRELIABLE_RANGES_OVERLAP` for exactly this.)
4. **Independent reruns, then tiered confidence.** A single sweep is **L1 (draft)** only.
   Promote to **L3** only after ≥N independent reruns (fresh build) agree in direction;
   reach **L5** only after cross-time / cross-device stability. Only ≥L3 findings may be
   used to prune the search space (see `docs/self-evolution-design.html`, L1–L5).
5. **Track absolute-baseline drift.** Record each model's absolute baseline over time. If
   the baseline shifts beyond threshold between runs, **invalidate dependent findings** and
   re-measure — a baseline that moves 2× is itself a regression signal, not a constant.

> One-line rule: on DVFS hardware, trust only **same-window paired deltas that exceed the
> noise floor and reproduce across independent reruns** — never single-run absolute
> baselines or cross-run ratios.

## Files
- `qnn_npu.json` — QNN HTP (NPU) EP findings
- `qnn_gpu.json` — QNN GPU EP findings
- `dml.json`     — DirectML EP findings
- `cpu.json`     — CPU EP findings
