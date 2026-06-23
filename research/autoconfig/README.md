# autoconfig — Automated Config Search POC

**Status: Research POC — not production code.**

This directory contains an experimental automated search system that finds the optimal
`winml-cli` build configuration (execution provider, opset version, graph optimizations)
for a given model on Windows hardware — without requiring the user to understand the
underlying ORT/EP optimizer mechanics.

---

## What This Is

`autoconfig.py` implements an Explorer/Optimizer/Reviewer loop as three explicit
classes wired by a thin orchestrator (`main()`):

1. **`Explorer`** — proposes the next hypothesis (opset, EP flags, graph passes): builds
   the `priority_queue` and prunes already-refuted configs via KB hard-blocks + the
   Insight Engine `skip_set`. Owns search *order* only.
2. **`Optimizer`** — runs `winml build` + `winml perf` (two-phase: 200-iter CV screen → 3×500-iter full bench)
   + `winml eval` accuracy. Produces raw measurements only.
3. **`Reviewer`** — applies the `ThroughputOnly` verdict (`threshold = max(1%, 2×CV)`),
   decides keep/discard, and drafts KB entries.

The loop terminates after 30 consecutive discards (plateau detection) or a time budget.

The same four-role architecture is also captured as composable **skill definitions**
under `skills/` — an `autoconfig-orchestrator` (the brain) that delegates to three
sub-skills `autoconfig-explorer`, `autoconfig-optimizer`, and `autoconfig-reviewer`.
Each `SKILL.md` mirrors the corresponding class and the diagram phase.

`catalog_sweep.py` is a single, JSON-driven multi-model sweep. It reads the hypothesis
matrix, model catalog, and per-EP bench protocol from `ep_device_knowledge/<ep>_<device>.json`
and runs them for any `--ep/--device` combination (qnn/npu, qnn/gpu, dml/gpu, cpu/cpu),
collecting structured results in `catalog-<device-or-ep>-sweep/<model-slug>/results.json`.

`analyze_graph.py` is an ONNX graph analysis helper that identifies architectural
patterns relevant to EP optimization (Transpose sandwiches, residual branches, GELU
variants, depthwise Conv) and surfaces gaps in `winml analyze` output.

`gen_report_v3.py` generates an HTML sweep report from `results.json` files.

`autoconfig_diagram.html` is an interactive architecture diagram of the Explorer/Optimizer/
Reviewer loop.

---

## Key Findings — 8-Model QNN NPU Catalog Sweep (2026-06-13)

### npu-001: opset 21 NHWC bypass is real — but architecture-specific

Opset ≥ 21 bypasses ORT's NHWC layout transformer for QNN EP, giving a large speedup
on **Conv + residual** models but no benefit (or slight regression) on pure transformers:

| Architecture | Models | opset 21 vs opset 17 |
|---|---|---|
| Conv + residual | MobileViT-small, DINOv2-small | **+26–31% speedup** |
| Pure transformer | ViT-base, YOLOS-small | neutral / slight regression |
| BERT-family NLP | DistilBERT, MiniLM, RoBERTa | neutral (within DVFS noise) |
| Plain Conv (ResNet) | ResNet-18 | ~+20% (h1→h3), but DVFS-dominated |

Root cause: ORT's `IsSupportedOpset()` gate in `layout_transformation.cc` causes the
NHWC layout transform to insert Transpose nodes around Conv ops. For Conv+residual
models these Transposes cannot be cancelled, so bypassing the transform (opset 21) gives
a cleaner HTP graph. Pure attention models have no Conv→NHWC transposes, so the bypass
has no effect.

### npu-006: Conv fusions cause ~4900% regression on QNN NPU for Conv-dominant models

`conv_bn_fusion`, `conv_add_fusion`, `conv_activation_fusion` produce fused op nodes
that QNN EP cannot execute natively — falling back to CPU for every fused Conv:

| Model | h4 (conv fusions) vs h1 (baseline) |
|---|---|
| ResNet-18 | **132.3 ms vs 2.72 ms (+4764% regression)** |
| MobileViT-small | 11.36 ms vs 11.72 ms (neutral) |
| DistilBERT | 19.59 ms vs 19.5 ms (neutral — no Conv to fuse) |

This is a critical correctness/performance hazard. `winml` should detect when the target
EP would CPU-fallback fused Conv ops and suppress incompatible fusions automatically
(see [Feature Gaps](#feature-gaps)).

### npu-007: DVFS thermal noise requires session-level averaging for reliable results

QNN NPU exhibits extreme DVFS thermal throttling. CV is consistently 0.10–2.0+ across
all models. Practical implications:

- The CV < 15% Phase-A gate must be **disabled** for QNN NPU (blocks all models)
- Differences < 10% between configs are **unreliable** without ≥ 1500 total iterations
- Recommended protocol: **3 × 500-iter sessions** with 30 s cool-down; report median of
  session p50 values
- 30 s cool-down reduces but does not eliminate DVFS spikes

---

## How to Run

### Prerequisites

- `winml` CLI installed and on PATH
- Python 3.11+ with `onnx` package (`pip install onnx`)
- For QNN experiments: Snapdragon X Elite device with QNN SDK (Hexagon HTP driver)

### autoconfig.py — single-model adaptive search

Configured at the top of the file (edit `MODEL_ID`, `TASK`, `EP`, `DEVICE`, `WORK_DIR`):

```bash
# Default: facebook/convnext-tiny-224 on CPU
python skills/orchestrator/autoconfig.py
```

Results are written to `WORK_DIR/results.tsv` and per-hypothesis subdirectories.
The script reads `ep_device_knowledge/<ep>_<device>.json` to prune already-refuted configurations.

### catalog_sweep.py — JSON-driven multi-model sweep

One driver covers every EP/device. The hypothesis matrix, model catalog, and bench
protocol (screen/full iterations, thermal handling, effect-size gate, paired A/B,
accuracy eval) all come from `ep_device_knowledge/<ep>_<device>.json`:

```bash
# Full QNN NPU catalog sweep (all models, ~6-8 hours on X Elite)
python tools/catalog_sweep.py --ep qnn --device npu

# CPU EP sweep, single model
python tools/catalog_sweep.py --ep cpu --device cpu --model microsoft/resnet-18

# QNN GPU sweep
python tools/catalog_sweep.py --ep qnn --device gpu

# Show the models/hypotheses configured for an EP/device
python tools/catalog_sweep.py --ep qnn --device npu --list
```

Results land in `catalog-<device-or-ep>-sweep/<model-slug>/results.json` and a `SUMMARY.md`
is regenerated at the end of each sweep.

### analyze_graph.py — ONNX graph analysis

```bash
# Edit the onnx path at the top of the file, then:
python skills/explorer/analyze_graph.py
```

Prints Transpose patterns, residual branch structure, GELU variants, and op domain
breakdown to stdout.

---

## ep_device_knowledge/ — Empirical Knowledge Base

Each JSON file stores empirical findings **and** the sweep configuration for one
EP/device combination, named `<ep>_<device>.json`:

| File | EP/device |
|---|---|
| `cpu_cpu.json` | CPU EP (Snapdragon X Elite Oryon) |
| `dml_gpu.json` | DirectML EP (GPU) |
| `qnn_gpu.json` | QNN Adreno GPU |
| `qnn_npu.json` | QNN HTP (Hexagon NPU) — most findings here |

### Schema overview

Each file has a `findings` array. Each finding has:

```json
{
  "id": "npu-001",
  "title": "...",
  "mechanism_confirmed": true,
  "architecture_requirement": ["has_conv_ops", "has_residual_connections"],
  "status": "confirmed",
  "confidence": "high"
}
```

It also carries the data-driven sweep contract consumed by `catalog_sweep.py`:
`sweep_config` (bench protocol), `hypotheses` (the h0–hN matrix with opset/optim/guards),
`models` (the catalog), and `cross_checks` (npu-001 opset-bypass, npu-006 catastrophic
regression, cpu-001 regression probe).

And a `search_space_rules` object that `autoconfig.py` reads to prune configurations
(only findings with `"mechanism_confirmed": true` are applied as pruning rules).

### Adding a new finding

1. Run the experiment and collect bench data
2. Add an entry to the appropriate `ep_device_knowledge/<ep>_<device>.json` under `findings`
3. Set `"mechanism_confirmed": false` and `"confidence": "draft"` until the mechanism
   is understood from ORT/EP source code
4. If the finding prunes a search dimension, add a rule under `search_space_rules`
5. Set `"mechanism_confirmed": true` only after source code investigation confirms
   the root cause — do NOT promote to confirmed based on benchmark numbers alone
6. See `ep_device_knowledge/README.md` for the epistemics guidelines

---

## Self-Evolution Tooling

Implements the loop from [`docs/self-evolution-design.html`](docs/self-evolution-design.html) —
how sweeps stabilize their own conclusions and promote findings without a human in the loop.

### skills/optimizer/bench_utils.py — paired A/B + adaptive sampling

Shared bench primitives used across sweeps:

- **`paired_ab_bench(run_session, baseline, hyp, n_pairs)`** (Fix #1) — interleaves the
  baseline and hypothesis perf sessions in one thermal window so DVFS/thermal drift appears
  in both legs and **cancels** in the within-pair ratio. Returns mean gain, 95% CI, and a
  verdict (`KEEP_CONFIRMED` / `MARGINAL` / `DISCARD`). This is the unbiased fix for the
  npu-001/MobileViT failure, where a cold baseline vs warm hypothesis manufactured a fake win.
- **`adaptive_paired_ab_bench(...)`** (Fix #2) — keeps adding pairs until the 95% CI is
  decisive (clears the KEEP or DISCARD band) or `MAX_PAIRS` is reached. Stable models finish
  in `MIN_PAIRS=3`; noisy ones automatically get more samples.
- **`thermal_classify(ref_p50, cold_ref_p50)`** (Fix #5) — classifies device thermal state
  (`COOL`/`WARM`/`HOT_RUN`) from a reference-model latency, for excluding throttled runs.
- **`session_cv(p50s)`** — between-session coefficient of variation (the effect-size noise floor).

The QNN sweep opts into paired A/B with `--paired-ab` (default off; the validated default is
the sequential Phase B):

```bash
python tools/catalog_sweep.py --ep qnn --device npu --model apple/mobilevit-small --task image-classification --paired-ab
```

### skills/reviewer/promote_findings.py — confidence-gated KB promotion (L1 → L4)

Post-processing script (Fix #4) that reads every `catalog-*-sweep/*/results.json` and applies
the confidence ladder, writing a **draft** to `ep_device_knowledge/_auto_promoted.json` (it never
clobbers the curated `<ep>_<device>.json` files):

| Level | Gate |
|---|---|
| **L1** Observed | median gain ≥ 5% on one model, one run |
| **L2** Confirmed | hypothesis p50 range strictly below baseline range **and** gain ≥ 2×(session CV) — the same effect-size gate the sweep uses |
| **L3** Generalized | same `(ep, flags)` reaches L2 on ≥2 distinct models of one architecture class (`model_type`) |
| **L4** Cross-cutting | same `(ep, flags)` reaches L2 across ≥3 architecture classes |

```bash
python skills/reviewer/promote_findings.py   # writes ep_device_knowledge/_auto_promoted.json
```

A human applies the promotion checklist in [`ep_device_knowledge/README.md`](ep_device_knowledge/README.md)
(paired A/B, clean baseline, effect-size > noise floor, independent reruns, baseline-drift
check) before merging any auto-promoted candidate into the curated KB.

### skills/explorer/analyze_insight.py — architecture-based pruning (Fix #3)

`build_insight()` fuses graph fingerprint + `winml analyze` + KB rules into a `skip_set`
(hypotheses to prune) and `priority_boosts` (reordering), cutting the 14-hypothesis matrix
to the few that matter per architecture.

---

## Feature Gaps Identified

Three actionable gaps in `winml-cli` surfaced by this research:

1. **FusedConv detection in `winml analyze`** — `analyze` should detect Conv ops that
   would CPU-fallback on QNN NPU after fusion (npu-006), and either warn or suppress
   incompatible fusions in the generated build config.

2. **DVFS-aware perf** — `winml perf` should support `--thermal-stabilization` mode
   that waits for device temperature to stabilize before measurements, and should report
   confidence intervals rather than a single p50.

3. **Budget-aware sweep** — `tools/catalog_sweep.py` exhausts the 20-min budget on models
   > 50 ms baseline after just 2 hypotheses (YOLOS: 78 ms × 3×500 iters = 207 s/hypothesis).
   A `--quick` flag that reduces to 1×200-iter for large models is needed.

---

## Directory Layout

```
research/autoconfig/
├── README.md                    ← this file
│
├── skills/                      ← the agent loop, one folder per role (each has SKILL.md + its scripts)
│   ├── orchestrator/            ← the brain: Phase 0–3 lifecycle
│   │   ├── SKILL.md
│   │   └── autoconfig.py        ← adaptive single-model search loop (Explorer/Optimizer/Reviewer classes)
│   ├── explorer/                ← "what to try next": priority_queue + skip_set
│   │   ├── SKILL.md
│   │   ├── analyze_insight.py   ← graph + analyze + KB → skip_set / priority_boosts
│   │   └── analyze_graph.py     ← ONNX graph pattern analysis helper
│   ├── optimizer/               ← "run it": build → screen → full bench → eval
│   │   ├── SKILL.md
│   │   └── bench_utils.py       ← shared bench primitives (paired A/B, adaptive, thermal, verdict)
│   └── reviewer/                ← "judge it": ThroughputOnly verdict + KB draft
│       ├── SKILL.md
│       └── promote_findings.py  ← L1→L4 confidence-gated KB promotion (draft sink)
│
├── lib/                         ← shared, role-agnostic helpers
│   ├── report_gen.py            ← HTML/markdown report rendering
│   └── gen_model_report.py      ← per-model report builder used by the sweeps
│
├── tools/                       ← batch drivers and one-off utilities
│   ├── catalog_sweep.py         ← JSON-driven multi-model sweep (--ep/--device, --paired-ab)
│   ├── validation_sweep.py      ← re-runs to validate KB findings
│   └── gen_report_v3.py         ← legacy HTML report generator
│
├── docs/                        ← design docs (self-evolution, agent, skills, cross-device)
│   └── autoconfig_diagram.html  ← Explorer/Optimizer/Reviewer architecture diagram
│
├── ep_device_knowledge/
│   ├── README.md                ← epistemics guidelines + promotion checklist
│   ├── _auto_promoted.json      ← promote_findings.py output (auto-generated draft)
│   ├── cpu_cpu.json             ← CPU EP findings + sweep config (ConvNext, 6 findings)
│   ├── dml_gpu.json             ← DirectML EP findings + sweep config
│   ├── qnn_gpu.json             ← QNN Adreno GPU findings + sweep config
│   └── qnn_npu.json             ← QNN HTP NPU findings + sweep config (npu-001 … npu-007)
│
├── catalog-qnn-sweep/           ← QNN NPU sweep results (also catalog-cpu-sweep/, catalog-gpu-sweep/)
│   ├── SUMMARY.md               ← 8-model sweep results and cross-model analysis
│   ├── apple--mobilevit-small/results.json
│   ├── facebook--dinov2-small/results.json
│   ├── microsoft--resnet-18/results.json
│   ├── google--vit-base-patch16-224/results.json
│   ├── deepset--roberta-base-squad2/results.json
│   ├── distilbert--distilbert-base-uncased-finetuned-sst-2-english/results.json
│   ├── sentence-transformers--all-MiniLM-L6-v2/results.json
│   └── hustvl--yolos-small/results.json
│
└── catalog-cpu-sweep/, catalog-gpu-sweep/  ← analogous per-model results for CPU / QNN GPU
```
