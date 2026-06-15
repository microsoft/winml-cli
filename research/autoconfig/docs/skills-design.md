# WinML CLI Skills Design Doc

## Overview

This document defines the design for 11 skills to be added to `skills/` in winml-cli.
Skills are split into three audiences:

- **Consumer skills (7)** — for WinApp developers deploying models
- **Contributor skills (3)** — for engineers extending winml-cli itself
- **Internal research skills (1)** — for winml-cli team to find optimization gaps and backlog items

Each skill follows the SKILL.md frontmatter convention (`name:`, `description:`) established
by Mobius, NVIDIA Model-Optimizer, and Google LiteRT-CLI as the de facto standard.

### Consumer skill dependency graph

```
ep-compatibility-check ──┐
                          ├──► optimize-for-device ──┐
use-winml-cli ────────────┤                           ├──► validate-before-ship
                          └──► debug-accuracy-drop ───┤
                                                       │
prepare-for-winapp ────────────────────────────────────┘

autoconfig ────────────────────────────────────────────► validate-before-ship
  (autoresearch loop: finds optimal config for user-defined EP/accuracy/latency targets)
```

### Internal research skill

```
optimization-research ──► [GitHub issues / winml backlog]
  (deep search: ORT source + Olive + ONNX ecosystem + native stack models
   → find better solutions → diagnose winml gaps → produce work items)
```

### Contributor skill dependency graph

```
adding-model-support ──► contributing-a-skill
adding-ep-support    ──► contributing-a-skill
```

---

## Design principle: Skills as agentic workflows

### The shift: documentation → automation

Current state (most skills in the ecosystem):
> Skill tells the user what commands to run → user runs them → user interprets output

Target state for winml-cli:
> Skill tells the **agent** what commands to run → **agent runs them** → agent interprets output → agent gives a specific answer

The difference:

| | Documentation skill | Agentic skill |
|---|---|---|
| Agent sees low cosine | "Run `winml eval --mode compare`" | Runs it, reads cosine=0.87, says "drop at quantize stage, Attention layers" |
| EP compatibility | "Run `winml sys` then `winml analyze`" | Runs both, parses JSON, says "QNN available but LayerNorm is partial" |
| Optimize precision | "Use the decision framework" | Runs fp16/w8a16/w8a8 sweep, builds actual tradeoff table, recommends W8A16 |
| Validate before ship | "Check these 6 gates" | Runs all 6 gates, generates a pass/fail report with actual numbers |

This is only possible if skills describe a **GATHER → ANALYZE → DECIDE → ACT** workflow,
and winml-cli commands emit **machine-readable structured output** that the agent can parse.

### Structured output: current state and gaps

Copilot agents have shell tool access and can run `winml` commands directly.
The key requirement is `--format json` on stdout so the agent can parse results
without screen-scraping Rich/ANSI terminal output.

| Command | Structured output today | Gap |
|---|---|---|
| `winml inspect` | ✓ `--format json` (stdout) | None |
| `winml sys` | ✓ `--format json` (stdout) | None |
| `winml run` | ✓ `--format json` (stdout) | None |
| `winml analyze` | ⚠ `--output file.json` (file only) | Add `--format json` stdout |
| `winml perf` | ⚠ `--output file.json` (file only) | Add `--format json` stdout |
| `winml eval` | ✗ No structured output | Add `--format json` stdout |

**Required code changes** (enables agentic skill execution):
1. `winml eval --format json` — outputs `{cosine, sqnr, psnr, task_metric}` to stdout
2. `winml analyze --format json` — outputs `{supported: [...], partial: [...], unsupported: [...]}` to stdout
3. `winml perf --format json` — outputs `{p50_ms, p90_ms, p99_ms, mean_ms}` to stdout

### The GATHER → ANALYZE → DECIDE → ACT skill structure

Each skill section should be written with agent execution in mind:

```
## GATHER: what to run
Commands the agent runs first (with --format json) to collect facts.

## ANALYZE: what to look for
How to interpret the JSON output. What values matter. What thresholds to apply.

## DECIDE: what to recommend
Decision logic. If X → recommend Y. If A and B → recommend C.

## ACT: what to tell the user
What to surface to the user: specific diagnosis + specific next step.
```

In practice this maps onto the existing "Sections" structure — the key is ensuring
each section has **concrete commands to run** and **concrete interpretation rules**,
not just prose description.

### Example: `debug-accuracy-drop` as an agentic workflow

```
User: "My W8A8 model has low accuracy"

GATHER:
  agent runs: winml eval --mode compare -m quantized.onnx --model-id <id> --format json
  agent gets: {"cosine_similarity": 0.87, "sqnr_db": 28.3, "stage": "quantize"}

ANALYZE:
  cosine=0.87 < 0.90 threshold → problem is real
  sqnr=28.3 < 30 dB → significant degradation
  stage=quantize → problem introduced at quantize, not optimize or compile

DECIDE:
  quantize-stage drop on W8A8 → check if Attention layers are the culprit
  agent runs: winml analyze -m quantized.onnx --ep qnn --format json
  agent gets: {"partial": ["MultiHeadAttention", "LayerNorm"], "unsupported": []}

ACT:
  Agent: "The accuracy drop (cosine=0.87) is at the quantize stage.
          MultiHeadAttention is partial on QNN — activations may be falling back to FP32.
          Try W8A16 to keep activations at FP16: winml build -c config.json --precision w8a16"
```

Without structured output (`--format json`), the agent would have to tell the user to run
each step manually and paste the results back. With structured output, the agent runs the
full diagnostic in one turn.

---

## Validation confidence levels (L1–L5)

Inspired by Mobius `writing-tests`. Applied in `validate-before-ship` as the Definition-of-Done backbone.
Each level is checked **independently** — a model can pass L3 without passing L2.

| Level | Name | What it verifies | Key command |
|---|---|---|---|
| **L1** | Loadable | Artifact is valid ONNX, loads without error | `winml inspect -m <artifact>` |
| **L2** | Shape correct | Output shape matches expected spec | `winml eval -m <artifact> --model-id <model>` (check shape in output) |
| **L3** | Numerical parity | Output matches FP32 baseline (cosine ≥ 0.99 FP16, ≥ 0.95 W8A16, ≥ 0.90 W8A8) | `winml eval --mode compare -m <artifact> --model-id <model>` |
| **L4** | Task accuracy | Task metric (Top-1/F1/mAP) within acceptable drop from FP32 reference | `winml eval -m <artifact> --model-id <model>` (task metric) |
| **L5** | Production ready | Perf SLA met on target device + cross-EP consistency verified | `winml perf --iterations 100 --monitor` |

**Quick pass criteria:**

| Precision | L3 threshold |
|---|---|
| FP16 | cosine_similarity ≥ 0.99 |
| W8A16 | cosine_similarity ≥ 0.95 |
| W8A8 | cosine_similarity ≥ 0.90 (or task-specific) |

Waivers: any level that cannot be verified must be documented with a reason and tracking issue.
The `validate-before-ship` skill maps each of its 6 gates to an L-level.

---

---

## Competitive Analysis

### Summary

winml-cli has a solid optimization pipeline (export→quantize→compile→benchmark) but lacks the **debugging/diagnostic loop**, **accuracy recovery tooling**, and **developer observability** that distinguish great toolchains from adequate ones.

---

### Competitor Feature Matrix

| Feature | Apple | ExecuTorch | AI Hub | NVIDIA | OpenVINO | Optimum | Olive | winml-cli |
|---|---|---|---|---|---|---|---|---|
| Per-layer accuracy debugging | ❌ | ✅ SVG graph | ✅ cloud | ❌ | ❌ | ❌ | ❌ | ❌ |
| Compute unit utilization report | ❌ | ✅ | ✅ | ❌ | Partial | ❌ | ❌ | ❌ |
| Accuracy-Aware PTQ (auto layer rollback) | ❌ | ❌ | ❌ | ❌ | ✅ NNCF | ❌ | ❌ | ❌ |
| Standard NLP benchmark (MMLU/PPL) | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Cross-EP side-by-side compare | ❌ | ❌ | Partial | ❌ | ❌ | ❌ | ❌ | ❌ |
| Zero-deploy validation (model.predict) | ✅ macOS | ✅ | ✅ cloud | ❌ | ✅ | ✅ | ❌ | Partial |
| Pre-quantized model zoo | ❌ | ❌ | ✅ 500+ | ✅ HF org | ✅ | ❌ | ❌ | ❌ |
| One-line optimize command | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Multi-EP artifact packaging | ✅ .mlpackage | ✅ .pte | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| QAT / accuracy recovery fine-tuning | ✅ | ❌ | ✅ AIMET | ✅ | ✅ | ❌ | ❌ | ❌ |
| Advanced quant (AWQ/SmoothQuant) | ❌ | ❌ | ✅ | ✅ | ✅ NNCF | ❌ | ❌ | ❌ |
| Thermal/sustained-load profiling | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

### Competitor Deep Dives

#### Apple coremltools
**Most relevant**: zero-deploy validation + compute_units API + palettization

- `model.predict({'input': np_array})` — validates converted model in one Python call without any device deploy. Can force `ComputeUnit.CPU_ONLY` for numerical comparison vs `CPU_AND_NE`.
- `compute_units` is switchable **at prediction time** (not just compile time) — enables A/B testing EP performance without re-converting.
- **Palettization**: LUT-based weight compression at 1–8 bits (k-means clustering, not linear quant). Matches Neural Engine hardware kernels better than INT4 linear quantization for many models.
- Three compression workflows: data-free / calibration-based / fine-tuning-based (QAT).
- `.mlpackage` separates architecture from weights → streaming-friendly, supports on-device compilation after download.

#### ExecuTorch (Meta)
**Most relevant**: per-layer QNN accuracy debugging (best-in-class of all competitors)

- `QNNIntermediateDebugger`: dumps intermediate tensor outputs at every QNN op, computes cosine similarity per layer vs CPU reference, generates **color-coded SVG computation graph** (green ≥ 0.9, red < 0.9).
- `get_delegation_info()`: table of ops showing delegated-to-NPU count vs CPU-fallback count per op type.
- `ETDump` + `Inspector` API: per-op timing table with avg (ms), op type, is_delegated. Returns pandas DataFrame.
- QAIRT Visualizer: `pip install qairt-visualizer` — interactive GUI overlaying op trace + QHAS (QNN HTP Analysis Summary) on model graph.
- **Missing**: no cloud device testing, no automated accuracy-latency sweep, build process is complex.

#### Qualcomm AI Hub
**Most relevant**: cloud profiling with physical hardware, per-step memory breakdown

- Compile + Profile + Inference on real physical devices (Snapdragon X Elite laptops, Galaxy S24) in the cloud — no local hardware needed.
- Per-step memory profiling: compilation time/memory, first-load time/memory (NE optimization), subsequent-load (cached), inference latency.
- 500+ pre-optimized models in model zoo.
- `--clone j1glw6y8p` — clone any previous job with modified params.
- Cloud AIMET quantization: sophisticated PTQ as a service (`submit_quantize_job()`).

#### NVIDIA ModelOpt
**Most relevant**: 16 compression techniques + MMLU benchmark scripts + pre-quantized HF checkpoints

- Compression techniques beyond PTQ: AWQ, SmoothQuant, QAT, pruning (Minitron 33% smaller, 50% faster), distillation, speculative decoding, sparsity, NAS (Puzzletron).
- Windows accuracy benchmark: `mmlu_benchmark.py` (57 subjects, DirectML/ORT/TensorRT-LLM/CPU), perplexity on WikiText-2, KL-divergence metrics.
- Pre-quantized HF checkpoints: `nvidia/DeepSeek-R1-FP4`, `nvidia/Llama-3.3-70B-FP4` etc. — pull validated optimized models without running pipeline.

#### Intel OpenVINO + NNCF
**Most relevant**: Accuracy-Aware PTQ (auto layer rollback)

- NNCF `AccuracyAwareQuantization`: automatically identifies sensitivity of each layer to quantization, rolls back sensitive layers to float when accuracy drop exceeds threshold. Fully automated accuracy-performance tradeoff solver.
- `benchmark_app -hint latency` vs `-hint throughput`: auto-configures streams, batch, inference requests for each mode. `-d AUTO`: automatic device selection with fallback.
- 100+ Jupyter notebooks on Binder/Colab — zero setup barrier.
- `OpenVINO GenAI`: high-level `LLMPipeline`, `WhisperPipeline` — deploy-ready LLM inference in 5 lines.

#### HuggingFace Optimum
**Most relevant**: drop-in Transformers replacement + multi-backend hub

- Replace `AutoModelForSequenceClassification.from_pretrained()` with `ORTModelForSequenceClassification.from_pretrained()` → ONNX Runtime inference with zero code change.
- 8 hardware backends: ONNX Runtime, OpenVINO, NVIDIA TensorRT-LLM, AMD Ryzen AI, AWS Inferentia, ExecuTorch, Intel Gaudi, FuriosaAI.
- Task-aware export: `--task text-generation` auto-configures dynamic axes and model wrapping.

#### Microsoft Olive (direct competitor)
**Most relevant**: one-line optimize command + VS Code AI Toolkit

- `olive optimize --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --precision int4 --output_path models/qwen` — one command, no per-step config.
- JSON-based pipeline config for full declarative multi-step control.
- VS Code AI Toolkit extension: GUI for model optimization, fine-tuning, and inference testing — no CLI knowledge needed.
- MultiLoRA serving support.

---

### Top 5 High-Impact Gaps for winml-cli

#### 🔴 Gap 1: Per-Layer Accuracy Debugging

**Pain**: Accuracy degrades after QNN compilation/quantization, user has no idea which layer caused it. Currently requires QNN SDK expert knowledge.

**Solution**: `winml debug --model model.onnx --ep qnn --inputs calibration_data/`
1. Runs model on CPU and QNN, captures intermediate tensor outputs at each op
2. Computes cosine similarity per layer
3. Outputs HTML/SVG graph with color-coded accuracy (green/red per layer)

**Reference**: ExecuTorch `QNNIntermediateDebugger` → `OutputFormat.SVG_GRAPH` + `QcomCosineSimilarityComparator`

**Impact**: Turns multi-day debugging into a 30-minute diagnosis. Currently no Windows-on-NPU tool does this.

---

#### 🔴 Gap 2: Compute Unit Utilization Report

**Pain**: `winml perf` shows slower-than-expected latency with no explanation. User doesn't know what % of ops ran on NPU vs fell back to CPU.

**Solution**: Extend `winml analyze` to output delegation table:
```
Op Type         | NPU Delegated | CPU Fallback | Reason
----------------|---------------|--------------|------------------
MatMul (INT8)   | 47 / 47       | 0            | -
LayerNorm       |  0 / 12       | 12           | Unsupported dtype
Softmax (FP32)  |  0 /  6       |  6           | Requires INT8 input
```

**Reference**: ExecuTorch `get_delegation_info().get_operator_delegation_dataframe()` / AI Hub per-layer compute unit mapping

**Impact**: Directly actionable — if user sees "60% of ops on CPU due to unsupported dtype," they know to switch to W8A8.

---

#### 🟠 Gap 3: Quantization Sensitivity Analysis

**Pain**: `winml quantize --algo w8a8` produces a model with unacceptable accuracy. User doesn't know if it's a specific layer, the algorithm, or the calibration data.

**Solution**: `winml analyze-quant --model model.onnx --calibration data/ --eval-dataset eval/`
1. Run full W8A8 quantization
2. For each block/layer, measure accuracy impact of reverting to FP16
3. Rank layers by sensitivity
4. Report: "reverting 3 attention layers to FP16 recovers X% accuracy at Y% latency cost"

**Reference**: Intel NNCF `AccuracyAwareQuantization` (automatic per-layer rollback)

**Impact**: Replaces multi-day trial-and-error with a 10-minute automated report.

---

#### 🟠 Gap 4: Standard Benchmark Integration (MMLU / Perplexity)

**Pain**: `winml eval` supports custom scripts but no out-of-box standard benchmarks. Users have no reference point for whether their quantized model's accuracy is "expected."

**Solution**: `winml eval --model model.onnx --benchmark mmlu --ep qnn`
- Built-in MMLU (57 subjects), WikiText-2 perplexity, KL-divergence scripts
- Reference numbers from FP32 baseline shown alongside quantized result
- `FP16 baseline: 78.2% → W8A8 QNN: 77.9% (−0.3%, expected range: −0.1% to −0.5%)`

**Reference**: NVIDIA ModelOpt `examples/windows/accuracy_benchmark/mmlu_benchmark.py` supports DirectML/ORT/CPU

**Impact**: Removes ambiguity and creates trust. Critical for LLM users.

---

#### 🟡 Gap 5: Cross-EP Side-by-Side Comparison

**Pain**: Choosing between QNN/DirectML/CPU/OpenVINO requires running each EP manually and aggregating results. No tool does this automatically.

**Solution**: `winml sweep --model model.onnx --precision w8a16,fp16 --ep qnn,dml,cpu`
- Runs build+eval+perf for each (precision × EP) combination
- Outputs a single comparison table: accuracy / latency / op coverage %
- Agent-driven: skill reads JSON output and recommends the optimal combination

**Reference**: Truly unique — no competitor does this for Windows multi-EP. Closest is AI Hub's multi-device fleet testing (Android only).

**Impact**: The single most-requested decision for Windows AI developers. Unique to winml-cli.

---

### Patterns in Great Toolchain DX

**Pattern 1: The "Why" Feedback Loop**
Great toolchains explain *why* results are the way they are. ExecuTorch's delegation table, AI Hub's compute unit mapping, NNCF's layer sensitivity analysis all answer "why?" winml-cli currently stops at "here's the result."

**Pattern 2: Progressive Disclosure of Complexity**
- Olive: `olive optimize --precision int4` (one line) → full JSON config pipeline
- coremltools: `ct.convert(model)` → MIL IR manipulation
- AI Hub: web dashboard → Python SDK → CLI → AIMET configs

winml-cli is currently too close to the expert path: each step requires understanding EP-specific options.

**Pattern 3: Zero-Deploy Validation**
Every strong toolchain lets you test model output before deploying to hardware: coremltools `model.predict()`, ExecuTorch Python pybind, AI Hub `submit_inference_job()`. winml-cli is strong for CPU but lacks the quick "compare CPU vs QNN output" path.

**Pattern 4: Pre-Validated Model Artifacts**
ModelOpt (HF nvidia/ org), AI Hub (500+ models), NNCF (Model Zoo with accuracy tables) all reduce the cold-start problem. Users don't need the full pipeline for popular models.

---

### Whitespace Opportunities (No Competitor Covers)

| Opportunity | Why it's winml-cli territory |
|---|---|
| **Cross-EP regression table** (one command, all EPs) | Multi-EP is the unique Windows AI challenge; no Android/iOS tool does this |
| **Quantization config recommender** (`winml recommend --target qnn --constraint latency=20ms`) | Rule-based recommendation from hardware+model arch analysis |
| **EP-aware ONNX graph visualizer** (Netron + green/yellow/red per EP) | Netron exists but has no EP coverage overlay |
| **Thermal/sustained-load profiling** (latency curve over 100 runs, detect throttling) | AI Hub hides variance; no tool surfaces thermal behavior |
| **Windows AI Model Package** (.mlpackage equivalent with multi-EP manifest) | Apple has .mlpackage; Windows has nothing equivalent |

---

## Skill 1: `use-winml-cli` (existing — extend)

**Status:** Exists at `skills/use-winml-cli/SKILL.md`. Needs two additions:
- Add `winml run` and `winml serve` usage (currently missing)
- Add "first-time onboarding" path for users who don't know where to start

No structural changes needed; the existing skill is the general entry point.

---

## Skill 2: `optimize-for-device`

### Frontmatter
```yaml
name: optimize-for-device
description: >
  Use this skill when a user wants the best performance for their model on a
  specific Windows device, or wants to compare latency/accuracy tradeoffs across
  quantization levels (FP16, W8A16, W8A8) and execution providers (QNN NPU,
  DirectML GPU, CPU). Covers the precision sweep workflow, hardware-specific
  recommendations, and how to read tradeoff results to make a deployment decision.
  Use when the user says "make it faster", "which precision should I use", "is NPU
  worth it", or asks to compare hardware.
```

### When to use
- "I want to run this on NPU, how much faster will it be?"
- "Which quantization should I pick?"
- "Compare QNN vs DirectML vs CPU for my model"
- "Is W8A8 accurate enough for my use case?"

### Sections

**1. The decision framework**
Two inputs: latency budget OR accuracy budget. Decision tree:
- Have a latency SLA (e.g. <50ms)? → Find highest accuracy within that budget
- Have an accuracy floor (e.g. <2% drop)? → Find fastest within that floor

**2. The precision ladder**
Table: FP32 → FP16 → W8A16 → W8A8, with typical speedup and accuracy-drop ranges
per model family (Encoder/BERT-like, Vision/ConvNet, Transformer/ViT).

**3. The sweep workflow**
Step-by-step: run `winml build` + `winml eval` + `winml perf` for each precision,
collect into a tradeoff table, apply decision framework.

Key commands:
```bash
winml config -m <model> --device <device> --precision fp16 -o config_fp16.json
winml build -c config_fp16.json -m <model> -o out_fp16/
winml eval -m out_fp16/<artifact>.onnx --model-id <model>
winml perf -m out_fp16/<artifact>.onnx --device <device> --iterations 50
# repeat for w8a16, w8a8
```

**4. Hardware-specific guidance table**
| Device | Best EP | Sweet-spot precision | Notes |
|---|---|---|---|
| Snapdragon X Elite NPU | QNN | W8A16 | HTP native for W8A16; W8A8 risky for Attention |
| Intel Core Ultra NPU | OpenVINO | W8A8 | OpenVINO PTQ handles INT8 well |
| AMD Ryzen AI NPU | VitisAI | W8A8 | Phoenix/Hawk Point prefer INT8 |
| Any GPU | DirectML | FP16 | FP16 sufficient; quantization rarely helps on GPU |
| CPU fallback | CPU | W8A8 | Size + latency both benefit |

**5. Reading the output**
How to interpret `winml eval` cosine_similarity, SQNR, and `winml perf` p50/p90/p99.
What values indicate "acceptable" vs "needs investigation".

**Cross-references:**
- If accuracy dropped unexpectedly → `debug-accuracy-drop`
- If EP not available → `ep-compatibility-check`
- After choosing a precision → `validate-before-ship`

---

## Skill 3: `debug-accuracy-drop`

### Frontmatter
```yaml
name: debug-accuracy-drop
description: >
  Use this skill when a quantized or optimized model produces worse accuracy than
  the FP32 baseline and the cause is unknown. Guides a structured diagnosis: first
  isolate which pipeline stage introduced the drop (optimize vs quantize vs compile),
  then use winml eval --mode compare to measure output similarity, then use winml
  analyze to check for partial/unsupported ops that may cause EP fallback. Covers
  calibration dataset issues, precision selection mistakes, and QNN-specific fallback
  patterns. Use when the user says "accuracy dropped after quantization", "results
  look wrong on NPU", or "cosine similarity is low".
```

### When to use
- "My model gives wrong results after quantization"
- "W8A8 accuracy is too low, how do I find out why"
- "Results differ between NPU and CPU"
- cosine_similarity < 0.95 from `winml eval --mode compare`

### Sections

**1. Isolation strategy: binary search on the pipeline**
Diagnose by bisecting the pipeline stages:
```
FP32 baseline
    → after optimize?   winml eval --mode compare (fp32 vs optimized)
    → after quantize?   winml eval --mode compare (fp32 vs quantized)
    → after compile?    winml eval --mode compare (fp32 vs compiled)
```
First stage where cosine drops → that's where the problem is.

Key commands:
```bash
# Export FP32 baseline
winml export -m <model> -o baseline/model.onnx

# Compare optimized vs baseline
winml eval --mode compare -m optimized/model.onnx --model-id <model>

# Compare quantized vs baseline
winml eval --mode compare -m quantized/model.onnx --model-id <model>

# Compare EP-compiled vs baseline (run on target EP)
winml eval --mode compare -m compiled/model.onnx --model-id <model> --ep qnn
```

**2. Interpreting similarity metrics**
Table of thresholds:
| Metric | Healthy | Investigate | Problem |
|---|---|---|---|
| cosine_similarity | > 0.99 | 0.95–0.99 | < 0.95 |
| SQNR (dB) | > 40 | 30–40 | < 30 |
| max_abs_diff | model-dependent | — | unbounded |

**3. Root cause patterns**

| Symptom | Likely cause | Fix |
|---|---|---|
| Drop appears at quantize stage | Calibration dataset not representative | Use task-relevant calibration data via `--calibration-dataset` |
| Drop appears at quantize stage for Attention layers | W8A8 quantizing activations in attention | Switch to W8A16 (keeps activations at FP16) |
| Drop appears at compile stage on QNN | Op pattern unsupported → CPU fallback | Run `winml analyze` to find partial ops |
| Inconsistent results across runs | Non-deterministic EP dispatch | Add `--iterations 20` to average out |
| Drop only in certain inputs | Input shape sensitivity | Test with calibration data matching real distribution |

**4. Checking for op fallback with `winml analyze`**
When compile-stage drop is suspected:
```bash
winml analyze -m quantized/model.onnx --ep qnn
```
Look for `partial` and `unsupported` ops — these fall back to CPU, introducing
numerical differences vs native NPU execution. Partial ops are the most common
source of unexpected accuracy variance on QNN.

**5. Precision escalation path**
If W8A8 is the problem and the model is accuracy-sensitive:
W8A8 → W8A16 → FP16 → FP32
Stop at the first precision that meets accuracy requirements.

**Cross-references:**
- To compare precision options systematically → `optimize-for-device`
- If op is listed as unsupported → `ep-compatibility-check`

---

## Skill 4: `prepare-for-winapp`

### Frontmatter
```yaml
name: prepare-for-winapp
description: >
  Use this skill when a WinApp developer needs to take winml-cli build artifacts
  and integrate them into a Windows application. Covers how to organize multi-EP
  artifacts (QNN/NPU, DirectML/GPU, CPU fallback), the recommended directory
  layout and manifest structure for runtime EP selection, how to load models
  using the Windows ML WinRT API or ONNX Runtime C++ API, and runtime EP
  detection and fallback patterns. Use when the user asks "how do I use this
  in my app", "how do I package the model", or "what file do I load at runtime".
```

### When to use
- "I built the model, how do I ship it in my app?"
- "How do I load different models for different hardware?"
- "What happens when the user doesn't have an NPU?"
- "How do I package QNN + DML + CPU variants together?"

### Sections

**1. The multi-EP artifact problem**
Explain why `winml compile` produces EP-locked files (not portable),
so a WinApp needs a strategy to select the right file per device.

**2. Recommended artifact layout**
```
my_model/
  manifest.json          ← EP → file mapping + version
  model_qnn.onnx         ← QNN NPU (compiled, Snapdragon X)
  model_openvino.onnx    ← OpenVINO NPU/GPU (Intel Core Ultra)
  model_vitisai.onnx     ← VitisAI NPU (AMD Ryzen AI)
  model_dml.onnx         ← DirectML GPU (any GPU, non-NPU machines)
  model_cpu.onnx         ← CPU fallback (universal)
```

**3. manifest.json schema**
```json
{
  "model_id": "facebook/convnext-tiny-224",
  "task": "image-classification",
  "version": "1.0.0",
  "variants": [
    { "ep": "qnn",       "device": "npu",  "file": "model_qnn.onnx",       "precision": "w8a16" },
    { "ep": "openvino",  "device": "npu",  "file": "model_openvino.onnx",  "precision": "w8a8"  },
    { "ep": "vitisai",   "device": "npu",  "file": "model_vitisai.onnx",   "precision": "w8a8"  },
    { "ep": "dml",       "device": "gpu",  "file": "model_dml.onnx",       "precision": "fp16"  },
    { "ep": "cpu",       "device": "cpu",  "file": "model_cpu.onnx",       "precision": "w8a8"  }
  ],
  "selection_order": ["qnn", "openvino", "vitisai", "dml", "cpu"]
}
```

**4. Building all variants with winml-cli**
```bash
# Generate configs per EP
winml config -m <model> --device npu --ep qnn -o config_qnn.json
winml config -m <model> --device npu --ep openvino -o config_ov.json
winml config -m <model> --device gpu --ep dml -o config_dml.json
winml config -m <model> --device cpu -o config_cpu.json

# Build all
winml build -c config_qnn.json -m <model> -o out_qnn/
winml build -c config_ov.json  -m <model> -o out_ov/
winml build -c config_dml.json -m <model> -o out_dml/
winml build -c config_cpu.json -m <model> -o out_cpu/
```

**5. Runtime EP selection pattern (C++ / ORT)**
Pseudocode for app-side logic:
- Read manifest.json
- Query available EPs on device (`GetAvailableProviders()` or `winml sys` equivalent)
- Walk `selection_order`, pick first EP available on this device
- Load the corresponding file
- If all fail → CPU is always available

**6. What NOT to do**
- Don't load a QNN-compiled model with CPU EP → will fail or produce wrong results
- Don't hardcode EP names → check availability at runtime
- Don't ship only the compiled artifact without a CPU fallback

**Cross-references:**
- To build the artifacts → `use-winml-cli`
- To verify each artifact → `validate-before-ship`

---

## Skill 5: `ep-compatibility-check`

### Frontmatter
```yaml
name: ep-compatibility-check
description: >
  Use this skill to determine whether a specific model will work on specific
  Windows hardware before starting a full build. Covers winml inspect for model
  support verification, winml sys for EP availability on the current machine,
  winml analyze for operator-level EP compatibility, and the EP-to-hardware
  mapping for Windows AI PCs. Use when the user asks "will this work on my
  device", "is QNN supported here", "what hardware do I need for NPU", or
  when they get an unsupported operator error.
```

### When to use
- "Will this model work on my Snapdragon X Elite laptop?"
- "I don't know if my machine has a QNN EP"
- "The compile step failed with unsupported op"
- Starting a new project: verify feasibility before investing build time

### Sections

**1. Three-layer compatibility check**
Layer 1 — Model support: does winml-cli know this model type?
Layer 2 — EP availability: is the target EP registered on this machine?
Layer 3 — Operator coverage: does the target EP support all ops in this model?

Each layer has a command; run in order, stop at first failure.

**2. Layer 1: Model support**
```bash
winml inspect -m <model-id>
```
What to look for: `loader`, `exporter`, `winml_inference_class` fields populated.
If inspect fails or shows "unsupported" → model is out of scope for winml-cli.

**3. Layer 2: EP availability**
```bash
winml sys --list-ep --list-device
```
EP-to-hardware reference table:
| EP | Hardware requirement | Check for |
|---|---|---|
| QNN | Qualcomm Snapdragon X Elite / X Plus | QNNExecutionProvider in list |
| OpenVINO | Intel Core Ultra (Meteor Lake / Lunar Lake+) | OpenVINOExecutionProvider |
| VitisAI | AMD Ryzen AI (Phoenix / Hawk Point / Strix) | VitisAIExecutionProvider |
| NvTensorRTRTX | NVIDIA discrete GPU (RTX series) | NvTensorRTRTXExecutionProvider |
| DML | Any DirectX 12 GPU | DmlExecutionProvider |
| CPU | Any | Always available |

If the desired EP is not listed → recommend next best EP from fallback chain.

**4. Layer 3: Operator coverage**
```bash
winml analyze -m <exported_model>.onnx --ep <ep>
# or for all EPs at once:
winml analyze -m <exported_model>.onnx --device all
```
Output interpretation:
- `supported` (green): op runs natively on EP
- `partial` (yellow): op may fall back to CPU for some configurations
- `unsupported` (red): op cannot run on this EP

Decision rule: any `unsupported` → either change EP or accept CPU fallback
for those ops (which may impact accuracy and latency).

**5. Fallback chain recommendation**
If target EP not available or has unsupported ops:
```
QNN not available → OpenVINO (if Intel) or VitisAI (if AMD) → DML → CPU
```

**6. Fast-fail before compile**
`winml compile` is expensive (minutes). Always run analyze first.
If analyze shows >20% unsupported ops → likely not worth compiling for that EP.

**Cross-references:**
- After confirming compatibility → `use-winml-cli` (build)
- If all EPs show unsupported ops → model may be out of scope for winml-cli

---

## Skill 6: `validate-before-ship`

### Frontmatter
```yaml
name: validate-before-ship
description: >
  Use this skill when preparing to release a Windows application with an
  on-device AI model. Provides a Definition-of-Done checklist covering artifact
  completeness, accuracy validation against FP32 baseline, performance SLA
  verification, output correctness on real inputs, cross-EP consistency, and
  fallback chain verification. Every item must be checked or explicitly waived
  before shipping. Use when the user says "I'm ready to ship", "what should I
  test before release", or "how do I know the model is good enough".
```

### When to use
- About to ship a WinApp with on-device inference
- Final QA gate before a model artifact goes to production
- After any build config change (new quantization, new EP, new model version)

### Sections

**1. The checklist**

**Gate 1 — Artifact completeness**
- [ ] All target EP artifacts exist and are loadable
- [ ] CPU fallback artifact exists
- [ ] manifest.json (if using multi-EP layout) is valid and references existing files
- [ ] Artifact was built with `winml build` (not opaque cache artifact)

Command:
```bash
winml inspect -m <artifact>.onnx  # verify each artifact loads
```

**Gate 2 — Accuracy vs FP32 baseline**
- [ ] cosine_similarity ≥ 0.99 for FP16 artifacts
- [ ] cosine_similarity ≥ 0.95 for W8A16 artifacts
- [ ] cosine_similarity ≥ 0.90 for W8A8 artifacts (or task-specific threshold)
- [ ] Task accuracy metric (Top-1, F1, mAP) within acceptable drop from FP32

Commands:
```bash
winml eval --mode compare -m <artifact>.onnx --model-id <model>
winml eval -m <artifact>.onnx --model-id <model>  # task accuracy
```

**Gate 3 — Performance SLA**
- [ ] p50 latency meets application target on target device
- [ ] p99 latency within 2x p50 (no outlier spikes)
- [ ] Benchmark run on actual target hardware (not developer machine)

Command:
```bash
winml perf -m <artifact>.onnx --device <target> --iterations 100 --monitor
```

**Gate 4 — Output correctness on real inputs**
- [ ] Model produces correct output on ≥3 representative real-world inputs
- [ ] No NaN or Inf in outputs
- [ ] Output shape matches expected shape

Command:
```bash
winml run -m <artifact>.onnx --file <real_input>  # visual/manual check
```

**Gate 5 — Cross-EP consistency (if shipping multiple EP variants)**
- [ ] QNN and DML outputs agree within tolerance on same input
- [ ] CPU fallback output agrees with primary EP within tolerance

Command (manual comparison across runs):
```bash
winml run -m model_qnn.onnx     --file sample.jpg --format json -o qnn_out.json
winml run -m model_dml.onnx     --file sample.jpg --format json -o dml_out.json
winml run -m model_cpu.onnx     --file sample.jpg --format json -o cpu_out.json
# compare qnn_out.json vs dml_out.json vs cpu_out.json manually
```

**Gate 6 — Fallback chain**
- [ ] CPU fallback artifact verified independently (not just assumed to work)
- [ ] App runtime selects correct artifact when target EP is absent (simulate by removing EP)

**2. Waiver policy**
Any item that cannot be completed must be waived explicitly:
```
Waivers:
- Cross-EP consistency: VitisAI not available on developer machine.
  Verified on target hardware by QA team. Issue #NNN.
- Performance SLA: Target hardware (Snapdragon X Elite) in procurement.
  Benchmark deferred to post-merge, tracked in issue #NNN.
```
Unchecked items without waiver → do not ship.

**3. L-level mapping**

The 6 gates map directly to the L1–L5 confidence system (see Overview):

| Gate | L-level |
|---|---|
| Gate 1 — Artifact completeness | L1 |
| Gate 2 — Accuracy vs FP32 baseline | L3 + L4 |
| Gate 3 — Performance SLA | L5 |
| Gate 4 — Output correctness on real inputs | L4 |
| Gate 5 — Cross-EP consistency | L5 |
| Gate 6 — Fallback chain | L1 (CPU artifact) |

Minimum to ship: L1 + L3 all passing. L4 + L5 required for production release.

**3. Quick command reference**
```bash
# Gate 1: inspect all artifacts
for f in model_qnn.onnx model_dml.onnx model_cpu.onnx; do winml inspect -m $f; done

# Gate 2: accuracy
winml eval --mode compare -m <artifact>.onnx --model-id <model>
winml eval -m <artifact>.onnx --model-id <model>

# Gate 3: perf
winml perf -m <artifact>.onnx --device auto --iterations 100 --monitor

# Gate 4: real input
winml run -m <artifact>.onnx --file <sample>

# Gate 5: cross-EP (run individually, compare outputs)
winml run -m model_qnn.onnx --file <sample> --format json
winml run -m model_dml.onnx --file <sample> --format json
```

**Cross-references:**
- If accuracy gate fails → `debug-accuracy-drop`
- If performance gate fails → `optimize-for-device`
- If EP not available for testing → `ep-compatibility-check`
- For multi-EP artifact packaging → `prepare-for-winapp`

---

## Skill 7: `adding-model-support` (contributor)

### Frontmatter
```yaml
name: adding-model-support
description: >
  Use this skill when contributing support for a new Hugging Face model to
  winml-cli. Covers finding the correct exporter, writing a recipe config,
  verifying at each pipeline stage (export → optimize → quantize → compile),
  and passing the L1–L5 validation gates before submitting a PR. Use when
  a contributor says "I want to add support for model X", "this model type
  is not supported", or "how do I write a recipe for a new architecture".
```

### When to use
- "I want to add support for Qwen3 / Phi-4 / [new model]"
- "winml-cli says this model is unsupported"
- "How do I write a recipe config for a new model family?"

### Sections

**1. Find the right exporter**
```bash
winml inspect -m <hf_model_id>  # check if auto-detected
```
If inspect fails → the model needs a new exporter or recipe.
Look in `src/winml/modelkit/export/` for existing exporters as reference.

**2. Find a reference model of the same family**
- Same architecture class (e.g., LlamaForCausalLM, BertModel)?
- Check `recipes/` for an existing `.json` config for that class
- Prefer copying the closest recipe and adjusting rather than writing from scratch

**3. Write the recipe config**
Minimal recipe template:
```json
{
  "model_id": "org/model-name",
  "task": "text-generation",
  "export": { "opset": 17 },
  "optimize": { "passes": ["MatMulAddFusion", "LayerNormFusion"] },
  "quantize": { "mode": "w8a16", "calibration_dataset": "wikitext2" }
}
```

**4. Validate at each stage (L1 → L5)**

| Stage | Command | Pass criterion |
|---|---|---|
| L1: Export loads | `winml inspect -m <exported>.onnx` | No error |
| L2: Shape correct | `winml eval -m <exported>.onnx --model-id <id>` | Output shape matches |
| L3: Numerical parity | `winml eval --mode compare -m <quantized>.onnx --model-id <id>` | cosine ≥ threshold |
| L4: Task accuracy | `winml eval -m <quantized>.onnx --model-id <id>` | Task metric in spec |
| L5: Perf on target EP | `winml perf -m <compiled>.onnx --device <target>` | Meets latency target |

**5. Common pitfalls for new models**
- New op types not in operator coverage → run `winml analyze` early
- Attention variant (GQA, MQA, MLA) → check quantization mode compatibility
- Dynamic shapes → add explicit shape hints in export config
- Non-standard tokenizer → verify `winml run` input preprocessing

**Cross-references:**
- If EP shows unsupported ops → `ep-compatibility-check`
- After L1–L5 all pass → `validate-before-ship` for PR gate

---

## Skill 8: `adding-ep-support` (contributor)

### Frontmatter
```yaml
name: adding-ep-support
description: >
  Use this skill when adding a new execution provider (EP) backend to
  winml-cli. Covers implementing the compile backend interface, adding
  EP-specific optimize passes, wiring the new EP into winml sys and
  winml analyze, and verifying coverage with the L1–L5 test gates.
  Use when a contributor says "I want to add support for a new EP",
  "how does the QNN compile backend work", or "can we support EP X".
```

### When to use
- Adding a new EP compile backend (e.g., a new NPU vendor)
- Extending an existing EP with new optimization passes
- Understanding how the existing QNN / OpenVINO / VitisAI backends are structured

### Sections

**1. EP backend interface**
Reference implementation: `src/winml/modelkit/compile/qnn_backend.py`
Three methods to implement:
```python
class MyEPBackend(CompileBackend):
    def is_available(self) -> bool: ...      # detect EP on current machine
    def optimize(self, model, config): ...   # EP-specific graph transforms
    def compile(self, model, config): ...    # produce EP-locked artifact
```

**2. Wire into EP registry**
Register in `src/winml/modelkit/ep_registry.py`:
```python
EP_REGISTRY["myep"] = MyEPBackend
```
This makes `--ep myep` work in `winml config`, `winml compile`, `winml analyze`.

**3. Add operator coverage data**
Add a coverage JSON to `src/winml/modelkit/analyze/coverage/myep_ops.json`:
```json
{ "Add": "supported", "LayerNorm": "partial", "CustomOp": "unsupported" }
```
This is what `winml analyze --ep myep` reads.

**4. Add to `winml sys` output**
Add EP availability check to `src/winml/commands/sys.py` so it appears
in `winml sys --list-ep`.

**5. L1–L5 validation for the new EP**
Minimum before merging:
- L1: A known-good model compiles without crash
- L3: Compiled artifact passes `winml eval --mode compare` (cosine threshold)
- L5: `winml perf` produces valid latency output on target hardware

**Cross-references:**
- Operator coverage analysis → `ep-compatibility-check`
- After adding: document the EP in `ep-compatibility-check` hardware table

---

## Skill 9: `contributing-a-skill` (contributor)

### Frontmatter
```yaml
name: contributing-a-skill
description: >
  Use this skill when writing a new SKILL.md for winml-cli or improving
  an existing one. Covers frontmatter requirements, description writing
  (the description is the agent trigger, not a human summary), section
  structure conventions, cross-reference format, command accuracy
  requirements, and the review checklist before submitting. Use when a
  contributor says "I want to add a new skill", "how should I write
  SKILL.md", or "what are the skill authoring rules".
```

### When to use
- Writing a new skill for a gap not covered by existing skills
- Improving an existing skill with new commands or sections
- Reviewing a skill PR

### Sections

**1. Frontmatter rules**
```yaml
name: kebab-case-skill-name   # matches directory name under skills/
description: >
  Use this skill when <trigger phrase describing user's problem>.
  Covers <what the skill teaches>.
  Use when the user says "<example trigger phrase 1>", "<example 2>", or <condition>.
```

**Critical:** The `description` field is what the Copilot agent reads to decide
whether to activate this skill. Write it as a trigger specification, not a
documentation summary. Include representative user phrases in quotes.

**2. Required sections (in order)**
1. `## When to use` — 3–5 bullet points with user-facing symptoms/questions
2. Diagnostic or decision section — symptom → cause → fix structure
3. Command examples — runnable `winml` commands with real flags
4. Reference tables — hardware, thresholds, EP names as concrete data
5. `## Cross-references` — links to related skills using relative paths

**3. Cross-reference format**
```markdown
- If accuracy dropped → see `.agents/skills/debug-accuracy-drop/SKILL.md`
- After validating → see `.agents/skills/validate-before-ship/SKILL.md`
```

**4. Content rules**
- All commands must be runnable exactly as written (no pseudocode flags)
- Include concrete numbers: thresholds (cosine ≥ 0.99), speedup (3–5×), latency (<50ms)
- Target ~200 lines prose + tables; move deep content to `references/` subdirectory
- Do not duplicate content from another skill — cross-reference instead

**5. Review checklist before PR**
- [ ] `description` contains ≥3 quoted user trigger phrases
- [ ] All commands are tested and produce the described output
- [ ] Cross-references use relative paths and the linked skill exists
- [ ] No commands reference flags that don't exist in current `winml --help`
- [ ] Hardware names and EP names match the canonical list in `ep-compatibility-check`
- [ ] `evals/eval.yaml` exists with ≥2 test cases (including at least one negative assertion)

---

## Skill 10: `autoconfig` (consumer — autoresearch loop)

### Frontmatter
```yaml
name: autoconfig
description: >
  Use this skill when a **WinApp developer** wants to automatically find the best
  winml-cli configuration for their model on one or more target EP/device combinations.
  The agent runs an autonomous experiment loop: it proposes config.json hypotheses,
  runs winml build + eval + perf, evaluates against user-defined objectives
  (accuracy floor, latency budget, or Pareto frontier), and iterates — keeping
  improvements, discarding regressions. Covers single-EP optimization, multi-EP
  parallel search, mixed-precision (nodes_to_exclude) exploration, calibration
  parameter tuning, and manifest.json output for multi-EP deployment.
  Use when the user says "find the best config for my model on QNN",
  "automate the config search", "generate configs for all EPs",
  or "I want to leave this running overnight".

audience: external (WinApp developers)
```

### When to use
- "Find the best W8A8 config that keeps accuracy > 0.95 on QNN"
- "Generate optimized configs for QNN + DirectML + CPU and build a manifest"
- "I don't know which quantization settings to use, figure it out for me"
- "Run overnight and give me the best accuracy-latency tradeoff you can find"
- User has a latency SLA or accuracy floor but doesn't know how to achieve it

### What this skill does NOT do
- It only searches within what `winml build` currently supports (existing capabilities)
- It does not look for optimization techniques outside winml's current feature set
- It does not suggest that winml needs new features or file bugs
- For finding what winml is *missing*, use `optimization-research` instead

---

### Epistemic standard for autoconfig findings

**Any conclusion this skill writes into a report or recommends to a user must meet this bar:**

| Requirement | What it means |
|---|---|
| **Observation vs explanation** | State what was measured separately from why it happened. "latency increased 270ms" is fact. "because NHWC causes cache thrashing" is a hypothesis — label it as such unless confirmed by profiling. |
| **Statistical validity** | A latency claim requires ≥ 3 independent runs with warmup. A single `winml eval` run (no warmup, includes preprocessing) is insufficient to quote as a latency number. It can guide search decisions but not final reports. |
| **Mechanism confirmation** | Do not explain a regression unless the mechanism is confirmed (e.g., by profiler, by op-level timing, or by **source code inspection of ORT/QNN SDK**). If unknown, write "cause unconfirmed; further profiling needed." |
| **Scope boundary** | Results measured on one model/EP are never generalized to other models/EPs without explicit qualification. "On ConvNext-tiny CPU" is allowed. "CPU dislikes fusion" is not — it's an overgeneralization. |
| **Unresolved uncertainty** | If an observation contradicts the expected behavior (e.g., a "disabled" fusion still appears in the output), the report must flag this as an open question, not silently adopt an explanation. |
| **EP isolation** | A finding on one EP (positive or negative) MUST NOT be applied to prune the search space of a different EP without independent validation. CPU opset regression ≠ QNN NPU opset regression. Always validate per EP independently. |

The skill MUST NOT write confident root-cause explanations in the HTML report or chat summary for regressions where only the measurement is available. Use hedged language: "this likely relates to…", "one hypothesis is…", or simply omit the explanation and recommend profiling.

#### Perf gain validation protocol

Before **any** perf gain is written into a report, config recommendation, or knowledge base as a confirmed finding, it must pass ALL three gates:

**Gate 1 — Statistical: two-phase bench protocol (from GPU Optimizer V2)**

```
Phase A — Quick screen (fast, ~2 min):
  winml perf -m <model> --ep <ep> --device <device> --warmup 20 --iterations 200 -o screen.json
  CV = screen.json.std / screen.json.p50
  IF CV > 0.10 (10%): REJECT — high DVFS variance, measurement unreliable
                       → cool down 120s, retry once
                       → if still CV > 0.10: flag as [UNSTABLE], skip candidate

Phase B — Full bench (only if Phase A passes, ~15 min):
  # 3 independent sessions with 60s cool-down between each
  winml perf ... --warmup 50 --iterations 1000 -o run1.json
  sleep 60
  winml perf ... --warmup 50 --iterations 1000 -o run2.json
  sleep 60
  winml perf ... --warmup 50 --iterations 1000 -o run3.json

  # KEEP if ALL of:
  #   1. p50(run1,2,3) are all faster than baseline p50 × (1 - min_improvement)
  #   2. CV of each run < 0.10
  #   3. cosine_similarity ≥ accuracy_floor
  KEEP_threshold = baseline_p50 × 0.99   # ≥1% improvement required
```
Rationale: DVFS on mobile NPUs causes 2-10x run-to-run variance. CV check catches this before wasting 15 min on full bench.

**Gate 2 — Mechanism: read ORT/QNN source code before explaining why**

**Gate 2 — Mechanism: read ORT/QNN source code before explaining why**
- For QNN EP gains: check `onnxruntime/core/providers/qnn/builder/` for opset-conditional dispatch
- For CPU EP gains: check `onnxruntime/core/optimizer/` for pass applicability conditions
- For DML EP gains: check DML operator mapping tables
- **Do not publish "opset 21 = 2.3x faster on QNN NPU" without confirming the mechanism in source code.** It may be DVFS bias, not a real architectural difference.

**Gate 3 — Reproducibility: baseline and candidate measured in same thermal state**
- Run baseline and candidate back-to-back in the same session OR
- Use a device-level tool to lock NPU clock frequency
- If you cannot control thermal state, report min_ms (peak-performance ceiling) alongside p50 (typical performance), and flag the variance explicitly.

**Lesson from ConvNext opset sweep (2026-06-10):**
Initial opset 21 measurement (8.45ms, 50 iters) vs opset 17 (19.4ms) appeared to show 2.3x gain. Full 17-22 sweep with 50 iters each showed:
- All opsets min ~9-10ms (same peak capability)
- opset 17 p50=54ms, opset 19-22 p50=12ms — but opset 18 p50=43ms (bimodal)
- opset 21 std varied from 10ms (cool device) to 37ms (warm device)
**Conclusion: data is inconclusive. Gain may be real OR may be thermal artifact. Gates 1+2 not yet passed.**

---

### Design Comparison: GPU Optimizer V2 vs WinML Autoconfig

**Reference**: "Agentic GPU Model Optimization" doc (cheye@, 2026-03-20). GPU Optimizer V2 is a 6-role multi-agent system for cloud GPU inference optimization (ONER-1B KNN service, H100). Autoconfig is a local edge inference optimizer (winml-cli, Snapdragon X). Most of their infrastructure (machine pool, SSH fleet, Triton serving, custom CUDA kernels, SM occupancy tuning) does not apply here. But the agent loop design has several directly adoptable ideas.

#### Adoptable insights from GPU Optimizer V2

| V2 design decision | V2 rationale | Adopt into autoconfig? | Notes |
|---|---|---|---|
| **Two-phase bench: 200-iter quick screen → 3×1000-iter full bench** | "CV<2% gates full bench — avoid wasting time on high-variance results" | ✅ **YES — highest priority gap** | We've been doing single 50-iter runs and calling them facts. CV check would have caught the DVFS noise immediately. |
| **Verdict policy names (ThroughputOnly, ThroughputOrLatency…)** | "Named policies prevent Reviewer from ad-hoc criteria drift" | ✅ YES (simplified) | Autoconfig should have explicit KEEP criteria: `p50_ms < baseline × (1 - threshold)` AND `cosine ≥ floor` |
| **Append-only experiment_log.md + results.tsv written only by Reviewer** | "Single writer = no drift, full audit trail" | ✅ YES | Our results.tsv exists but no "single writer" discipline |
| **Explorer mandatory external-research triggers** | "After 15 consecutive DISCARDs → external research sweep" | ✅ YES — this is the exact gap that caused the opset 21 miss | If we had this rule, we would have searched ORT source after N DISCARDs and found kMaxSupportedOpset earlier |
| **Knowledge agent with review gate before KB save** | "Learnings reviewed before they prune future search" | ✅ YES | ep_knowledge/*.json entries should be marked draft until Gate 2 (mechanism) is confirmed |
| **Correctness contract locked after Phase 0, never modified** | "Prevents accuracy goal-post moving" | ✅ YES | We have accuracy gate but no locked contract file |
| **30-consecutive-DISCARD stop condition** | "Prevents endless search in exhausted space" | ✅ YES | autoconfig has no stop condition today |
| **Per-experiment structured output: Hypothesis → Implementation → Parity → Perf → Analysis → Decision** | "Enables post-analysis and knowledge extraction" | ✅ YES | autoconfig report is currently holistic, not per-experiment |
| **Role separation: Profiler / Explorer / Optimizer / Reviewer are separate agents** | "Prevents context drift; each agent stays focused" | ⚠️ Partial | Full 6-agent split is overkill for CLI tool; but Explorer / Reviewer distinction is valuable |
| **Resource lock: only one GPU job at a time** | "Prevents benchmark interference" | ✅ YES (trivially) | Already serial; but should be explicitly enforced if autoconfig ever parallelizes |
| **Machine pool + SSH fleet + Model Registry** | Cloud GPU fleet management | ❌ N/A | Local device only |
| **Custom CUDA kernel writing** | "Extreme asymmetry benefits from custom kernels" | ❌ N/A | CLI-only constraint; no kernel modification |
| **SM occupancy / GEMM tile count tuning** | "H100 has 132 SMs; 48 output tiles = 36% occupancy" | ❌ N/A | Edge NPU/GPU, not H100 multi-SM |
| **FlashAttention / fused QKV** | "Eliminate HBM traffic for attention score matrix" | ❌ N/A | Model is already trained; deployment-time optimization only |

#### Key gaps in current autoconfig design (from V2 comparison)

**Gap 1 (critical): No two-phase bench protocol**
Current design runs `--iterations 50` and accepts the result. V2 runs:
1. Quick screen: 200 iters, check CV < 2% (Coefficient of Variation = std/mean)
2. Only if CV < 2%: full bench 3×1000 iters with 60s cool-down between sessions
3. KEEP only if Δp50 > threshold AND CV(candidate) < 2%

This directly matches the "iter ≥ 1000" rule we just added. Formalize it as two phases.

**Gap 2 (critical): No mandatory external-research trigger in Explorer**
V2 Explorer triggers external research (web search, papers, source code) after:
- 15 consecutive DISCARDs
- Every KEEP that changes model/precision
- Before declaring backlog_empty

We discovered kMaxSupportedOpset only by accident (downloading QNN Hub models). A mandatory "read ORT source after 5 DISCARDs in opset dimension" rule would have found it in Phase 2.

**Gap 3 (important): ep_knowledge/*.json has no draft/confirmed state**
V2 Knowledge agent requires review gate before KB entries are used to prune search space. Our ep_knowledge findings should have:
- `status: "draft"` — observed, mechanism unconfirmed (Gate 2 not passed)
- `status: "confirmed"` — mechanism confirmed via source code (Gate 2 passed)  
- `status: "deprecated"` — finding invalidated by new experiment or ORT version change
Only `"confirmed"` entries should prune search space. `"draft"` entries inform hypothesis priority but don't prune.

**Gap 4 (nice-to-have): No per-experiment structured artifact**
V2 produces per-experiment: Hypothesis / Implementation / Parity / Perf / Analysis / Decision
autoconfig produces: one aggregate report.html. Should produce both.

### Design: The Autoresearch Loop

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
agent modifies a config file, runs a fixed-cost experiment, checks if the objective improved, keeps or discards, and repeats autonomously until manually stopped or convergence criteria met.

```
OBJECTIVE (user-defined, one of):
  A. Accuracy-primary:  maximize cosine_similarity  subject to  p50_ms ≤ <budget>
  B. Latency-primary:   minimize p50_ms             subject to  cosine ≥ <floor>
  C. Pareto search:     find the full accuracy-latency frontier

SEARCH SPACE — config.json has three sections the agent can modify:

  [export]
    opset_version          : int   — 17, 18, 19, 20  (higher = newer ops, EP may not support)
    do_constant_folding    : bool  — may affect graph structure visible to EP
    dynamic_axes           : dict  — static vs dynamic shapes (QNN prefers static batch=1)

  [optimize]  — full capability list (from winml optimize --list-capabilities)

    GraphPipe (run via ORT SessionOptions):
      GELU:
        gelu-fusion            : bool  — fuse tanh-GELU subgraph → Gelu op
        fast-gelu-fusion       : bool  — fuse fast-GELU (tanh-approx) → FastGelu
        bias-gelu-fusion       : bool  — fuse Bias+GELU (requires gelu-fusion)
        quick-gelu-fusion      : bool  — fuse x*sigmoid(1.702x) → FastGelu
        gelu-approximation     : bool  — convert exact Gelu → FastGelu (requires gelu-fusion)
      Activation:
        bias-softmax-fusion    : bool  — fuse Bias+Softmax
        bias-dropout-fusion    : bool  — fuse Bias+Dropout
      Convolution:
        conv-add-fusion        : bool  — fuse Conv+Add (bias)
        conv-bn-fusion         : bool  — fuse Conv+BatchNorm into weights
        conv-mul-fusion        : bool  — fuse Conv+Multiply
        conv-activation-fusion : bool  — fuse Conv+activation (ReLU, Sigmoid, etc.)
      Elimination:
        slice-elimination      : bool  — remove redundant Slice ops
        expand-elimination     : bool  — remove no-op Expand
        unsqueeze-elimination  : bool  — fold Unsqueeze into initializers
      GEMM:
        gemm-activation-fusion : bool  — fuse GEMM+activation
        gemm-sum-fusion        : bool  — fuse GEMM+Sum
        gemm-transpose-fusion  : bool  — fuse GEMM+Transpose
      Graph:
        concat-slice-elimination   : bool  — remove Concat+Slice that restore originals
        double-qdq-pairs-remover   : bool  — remove consecutive QDQ pairs
        constant-folding           : bool  — pre-compute constant exprs (default=True; disable to reduce size)
      LayerNorm:
        layer-norm-fusion          : bool  — fuse ReduceMean→Sub→Pow→Sqrt→Div→Mul→Add
        skip-layer-norm-fusion     : bool  — fuse Add(residual)+LayerNorm → SkipLayerNorm (requires layer-norm-fusion)
        simplified-layer-norm-fusion : bool — fuse simplified LayerNorm (no mean-centering)
      Layout:
        transpose-optimizer        : bool  — eliminate redundant transpose chains
        nhwc-transformer           : bool  — NCHW→NHWC (GPU memory layout)
        nchwc-transformer          : bool  — NCHW→NCHWc (CPU SIMD layout)
        conv-add-activation-fusion : bool  — fuse Conv+Add+Activation → FusedConv
      MatMul:
        matmul-add-fusion          : bool  — fuse MatMul+Add → single kernel
        matmul-activation-fusion   : bool  — fuse MatMul+activation (DML-only, requires matmul-transpose-fusion)
        matmul-transpose-fusion    : bool  — fuse MatMul+Transpose → FusedMatMul
        matmul-scale-fusion        : bool  — fuse MatMul+Scale
        matmul-bn-fusion           : bool  — fuse MatMul+BatchNorm
        dynamic-quantize-matmul-fusion : bool — dynamic quant for MatMul
      Misc:
        gather-slice-to-split-fusion : bool — fuse Gather+Slice → Split
        gather-to-slice-fusion       : bool — convert Gather to Slice (contiguous idx)
        pad-fusion                   : bool — fuse Pad with Conv/Pool
        not-where-fusion             : bool — fuse Not+Where

    FusionPipe (ORT transformer fusions, via FusionOptions):
      attention-fusion              : bool  — fuse MHA pattern → Attention/MultiHeadAttention
      layer-norm-fusion             : bool  — (FusionPipe variant, same flag)
      skip-layer-norm-fusion        : bool  — (FusionPipe variant)
      simplified-layer-norm-fusion  : bool  — (FusionPipe variant)
      embed-layer-norm-fusion       : bool  — fuse Embedding+Position+LayerNorm (requires layer-norm-fusion)
      bias-skip-layer-norm-fusion   : bool  — fuse Bias+SkipLayerNorm (requires skip-layer-norm-fusion)
      fuse-rmsnorm                  : bool  — fuse RMSNorm → LpNormalization(p=2) [custom, QNN-compatible]
      packed-qkv-fusion             : bool  — (SD only)
      packed-kv-fusion              : bool  — (SD only)
      skip-group-norm-fusion        : bool  — (SD only)
      bias-add-fusion               : bool  — fuse BiasAdd
      qordered-matmul               : bool  — (SD only)

    SurgeryPipe (pre-EP graph fixes):
      clamp-constant-values         : bool  — clamp -inf/+inf constants → [-1e3, 1e3] (prevents QNN quant issues)
      remove-isnan-in-attention-mask: bool  — remove Softmax→IsNaN→Where guards (use after clamp)

    RewritePipe (pattern-based subgraph rewriting):
      --enable-{source-slug}-{target-slug}  (run winml optimize --list-rewrites for full list)
      Examples: --enable-gelu-singlegelu, --enable-matmuladdpattern-reshapegemmreshapepattern

  [quant]
    precision              : fp16 | w8a16 | w8a8
    calibration_method     : minmax | entropy | percentile
    samples                : 64 | 128 | 256 | 512
    per_channel            : bool
    symmetric              : bool
    op_types_to_quantize   : list[str]  — restrict which op types get quantized
    nodes_to_exclude       : list[str]  — exclude specific named nodes

FIXED:  winml build + winml eval + winml perf  (the experiment harness)
METRIC: cosine_similarity  (from winml eval --format json)
        p50_ms             (from winml perf --format json)
RECORD: results.tsv
```

---

### Profiler-Enhanced Agent Architecture (redesigned)

**Insight from GPU Optimizer v2 analysis and ConvNext POC:**
Running the profiler *before* the search loop would have shown Gemm=57.7% on ConvNext —
immediately ruling out layout-pass experiments (Transpose only 2.6%, already fused Gelu already
canonical). Profile-first makes the Explorer smarter and the search shorter.

**New 4-phase structure:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 0 — INTAKE                                                    │
│   winml inspect → validate model is supported                       │
│   winml build (baseline config) → get model.onnx                   │
│   winml eval --mode compare → lock FP32 correctness baseline        │
│   winml perf (baseline) → establish latency floor                   │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1 — PROFILE  (runs ONCE, before any search)                   │
│   winml perf -m baseline/model.onnx --ep <ep> --profile             │
│   Parse bottleneck.json:                                            │
│     - top_bottleneck: op type with highest % of kernel time         │
│     - top3_concentration_pct: how concentrated the compute is       │
│     - headroom_hints: actionable pass recommendations               │
│   Classify each bottleneck op type:                                 │
│     - "compute" (Gemm, Conv, Attention) → quant/kernel matters      │
│     - "layout" (Transpose, Reshape) → graph pass matters            │
│     - "already_canonical" (op shows as fused type) → fusion N/A    │
│   Output: prioritized_hypothesis_queue (ordered by profile evidence)│
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2 — PROFILE-GUIDED OPTIMIZATION LOOP                          │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────┐  │
│  │   EXPLORER   │───►│  OPTIMIZER   │───►│      REVIEWER       │  │
│  │              │    │              │    │                     │  │
│  │ Pops next    │    │ Runs ONE     │    │ Cross-exp verdict:  │  │
│  │ hypothesis   │    │ experiment:  │    │ - CV gate Phase A   │  │
│  │ from queue,  │    │ build +      │    │ - full bench Gate 1 │  │
│  │ motivated by │    │ quick-screen │    │ - keep / discard    │  │
│  │ profile data │    │ → full bench │    │ - detect plateau    │  │
│  │              │    │ → eval       │    │ - stop condition    │  │
│  └──────────────┘    └──────────────┘    │ - write KB draft   │  │
│         ▲                               └─────────────────────┘  │
│  mandatory external-research triggers (adopted from V2):           │
│    • after 5 consecutive DISCARDs in same search dimension         │
│      → search ORT/QNN SDK source code for mechanism               │
│    • after every KEEP that changes precision or EP                 │
│      → re-read ep_knowledge for updated constraints                │
│    • before declaring search_space_exhausted                       │
│      → ORT source sweep: opset gates, EP-specific dispatch rules   │
│                                                                     │
│  Explorer prunes via bottleneck.json (only "confirmed" KB rules):  │
│    IF top_bottleneck == "Gemm" (>50%):                              │
│      → SKIP layout passes (transpose-optimizer, nchwc, nhwc)        │
│      → FOCUS on: quant precision, calibration, matmul fusions       │
│    IF top_bottleneck == "Transpose" (>10%):                         │
│      → CHECK kMaxSupportedOpset for current ORT version FIRST       │
│    IF top_bottleneck == "Conv" (>20%):                              │
│      → try nchwc-transformer, conv-activation-fusion               │
│    IF "Gelu"/"LayerNormalization" op_type (already canonical):      │
│      → SKIP corresponding fusion flags                              │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3 — REPORT                                                    │
│   config_<ep>_optimal.json  ← champion config with _autoconfig_meta│
│   report.html               ← full benchmark + profile section      │
│   experiments/<n>/          ← per-exp: hypothesis/impl/parity/     │
│                                perf/analysis/decision (V2 pattern)  │
│   kb_entry.json             ← status="draft"; promoted to          │
│     "confirmed" only after mechanism confirmed (Gate 2)             │
└─────────────────────────────────────────────────────────────────────┘
```

**ep_knowledge draft/confirmed lifecycle (Gap 3 fix):**

```
KB entry states:
  "draft"     — observed perf delta, mechanism unconfirmed (Gate 2 not passed)
                Can influence hypothesis PRIORITY but NOT prune search space
  "confirmed" — mechanism confirmed via ORT/QNN source code (Gate 2 passed)
                Can prune search space for future runs
  "deprecated"— finding invalidated by new experiment or stack version change
                Must NOT influence search space; kept for history only

Transition rules:
  draft → confirmed:   requires mechanism_confirmed=true + source_citation
  confirmed → deprecated: requires contradicting experiment OR stack version bump
  deprecated entries:  kept in JSON with status field, never deleted
```

**Profiler output → Explorer mapping table:**

| Profile finding | Explorer action | Hypothesis skipped |
|---|---|---|
| Gemm > 50% | Prioritize quant/calib experiments | All layout-transform passes |
| Transpose < 5% (opset=17) | Transpose Optimizer already working | transpose-optimizer trials |
| op_type "Gelu" present | Already fused | gelu-fusion, fast-gelu-fusion |
| op_type "LayerNormalization" present | Already fused | layer-norm-fusion trials |
| Reorder{Input,Output} present (>4%) | NCHWc already active | nchwc-transformer trials |
| op_type "Attention" present | MHA already fused | attention-fusion trials |
| QDQ ops > 15% | Quant overhead high | Focus on op_types_to_quantize exclusions |
| Transpose > 10% + opset ≥ 19 | kMaxSupportedOpset issue | Flag as [KNOWN_TRADEOFF], lower opset |

**Why profile-first matters (validated on ConvNext):**

The ablation experiment ran 22 experiments over multiple days. Had the profiler run first:
- Profile shows: Gemm=57.7%, Conv=12.6%, Transpose=2.6%, Gelu=8% (already "Gelu" op)
- Explorer would have immediately skipped: `gelu-fusion`, `layer-norm-fusion`, `transpose-optimizer`,
  `nchwc-transformer` (already active via ReorderInput/Output)
- Only candidates from profile: `matmul-add-fusion` (Gemm bottleneck), `conv-activation-fusion`
- This would have reduced 22 experiments to ~6, with the same conclusions

**POC profiler:** `C:\tmp\autoconfig-demo\winml_profile.py`
- Uses ORT `enable_profiling=True` + `end_profiling()` (same pattern as AI Studio's profile_file.py)
- CPU EP: parses `_kernel_time` events from ORT JSON trace
- Output: `bottleneck.json` (structured) + `bottleneck.txt` (human-readable) + raw ORT trace
- ConvNext result: Gemm 57.7%, Conv 12.6%, Transpose 2.6% → confirms baseline is optimal for CPU

---

### Sections

**1. Phase 0 — Intake + Baseline**

```bash
# Step 1: verify the model is supported
winml inspect -m <model-id> --format json

# Step 2: baseline build (default config, opset=17)
winml export -m <model-id> -o baseline/
winml build -c config_baseline.json -m <model-id> -o baseline_built/

# Step 3: correctness contract
winml eval --mode compare -m baseline_built/model.onnx --model-id <model-id> --format json
# Expected: cosine=1.0 (FP32 self-comparison)

# Step 4: baseline perf
winml perf -m baseline_built/model.onnx --ep <ep> --warmup 10 --iterations 50 --format json
# Record: baseline_p50_ms
```

Initialize `results.tsv` (TSV, not CSV — commas break in description field):
```
commit	precision	nodes_excluded	cosine	p50_ms	calibration_samples	status	notes
```

---

**2. Phase 1 — Profile (runs once, BEFORE any search experiments)**

```bash
# Run profiler on baseline model (--profile flag added to winml perf)
winml perf -m baseline_built/model.onnx --ep <ep> \
  --warmup 5 --iterations 20 --profile --out profile_out/ --format json
# Reads: profile_out/bottleneck.json
# POC (before --profile ships): python winml_profile.py --model ... --ep ...
```

Profiler output drives Explorer hypothesis initialization:

```
READ bottleneck.json:
  top_bottleneck: <op_type>
  op_summary: [{op_type, pct}, ...]  (sorted by descending pct)
  headroom_hints: [...]

BUILD skip_set (passes not worth trying):
  FOR each op_type in op_summary:
    IF op_type == "Gelu":          skip_set.add(gelu-fusion, fast-gelu-fusion)
    IF op_type == "LayerNormalization": skip_set.add(layer-norm-fusion)
    IF op_type == "Attention":     skip_set.add(attention-fusion)
    IF "ReorderInput" in op_summary AND pct > 2%:
                                   skip_set.add(nchwc-transformer)  # already active
  IF Transpose pct < 5% AND opset=17:
                                   skip_set.add(transpose-optimizer)  # already working, no gain
  IF Transpose pct > 10% AND opset >= 19:
                                   flag as [KNOWN_TRADEOFF]; add to report

BUILD priority_queue (hypotheses in evidence-based order):
  IF top_bottleneck == "Gemm" OR "MatMul":
    queue: [quant_precision, calib_method, calib_samples, matmul_fusions, per_channel]
  IF top_bottleneck == "Conv":
    queue: [nchwc (if not in skip_set), conv_fusions, quant_precision]
  IF top_bottleneck == "Attention":
    queue: [quant_precision, nodes_to_exclude (Attention), calib_method]
  DEFAULT:
    queue: [quant_precision, calib_method, calib_samples]
```

---

**3. Phase 2 — Profile-Guided Optimization Loop (single EP)**

```
LOOP FOREVER (until user stops or convergence):

1. EXPLORER: pop next hypothesis from priority_queue
   - Skip if in skip_set (pruned by profile)
   - If queue empty → enter Phase 4 (generalization) or stop

2. HYPOTHESIZE: build config.json delta based on hypothesis
   Hypothesis rules (profile-informed, in priority order):
   a. If first loop: start with full W8A8/W8A16, all ops quantized
   b. If cosine < floor: add worst partial_op to nodes_to_exclude (one at a time)
   c. If cosine ≥ floor but latency > budget: try W8A8 instead of W8A16,
      or reduce calibration_samples, or add per_channel=true
   d. If stuck (3 iterations no improvement): try calibration_method change
      (minmax → entropy → percentile)
   e. If still stuck: try precision escalation (W8A8 → W8A16 → FP16)

3. MODIFY: write updated config.json
   Key fields in quant section:
   {
     "precision": "w8a8",
     "samples": 128,
     "calibration_method": "minmax",
     "nodes_to_exclude": ["LayerNorm_0", "Softmax_3"],
     "per_channel": false
   }

4. OPTIMIZER: winml build -c config.json -m <model-id> -o out_<iteration>/
   If build crashes: log as "crash", revert config, try different hypothesis

5a. EVAL — quick sanity (cosine proxy, cheap):
    winml eval --mode compare -m out_<iteration>/artifact.onnx \
               --model-id <model-id> --format json
    → cosine_similarity, sqnr_db
    If cosine < hard_floor (e.g. 0.85): fail-fast, skip step 5b + 6, log as discard

5b. EVAL — task accuracy (real quality gate):
    winml eval -m out_<iteration>/artifact.onnx \
               --model-id <model-id> \
               --task <task>  --device <target> --ep <ep> \
               --samples 100 --format json
    → top1_accuracy (image-classification), f1 (text), mAP (detection), etc.
    This is the authoritative accuracy metric for Reviewer verdict.

    Why cosine alone is not sufficient:
    - High cosine (0.97) but top-1 drops 5%: logit magnitudes preserved but relative ranking shifted
    - Low cosine (0.92) but same top-1: relative ranking unchanged despite numeric difference
    → Only task accuracy tells you whether the model still does its job

6. PERF: winml perf -m out_<iteration>/artifact.onnx \
         --device <target> --ep <ep> --warmup 10 --iterations 50 --format json
   → p50_ms, p90_ms

7. REVIEWER: cross-experiment verdict
   keep    if task_accuracy ≥ accuracy_floor  AND  p50_ms ≤ latency_budget
   discard if task_accuracy < accuracy_floor  OR   p50_ms > latency_budget
   crash   if build/eval failed

   Reviewer also checks:
   - Plateau: 3+ keeps with Δlatency < 2% → likely at local optimum
   - Profile divergence: if new op_type appears after build, re-profile
   - Skip_set update: if experiment proves a pass is a no-op, add to skip_set
   - Accuracy cliff: if task_accuracy drops > 3% in one step → flag, do not cascade

8. LOG to results.tsv:
   <git-short-hash>  <precision>  <nodes_excluded>  <cosine>  <top1_acc>  <p50_ms>  <samples>  keep/discard/crash  <notes>

9. If keep: advance to next iteration from this config
   If discard: revert to last kept config, try different hypothesis
```

**Convergence criteria** (stop the loop):
- cosine ≥ target floor AND p50_ms ≤ latency budget: objective achieved
- 5 consecutive discards with no improvement: report best so far
- User manually stops the agent

---

**3. Hypothesis generation rules (the intelligence layer)**

The agent generates hypotheses by traversing the search space in priority order.
Each hypothesis is motivated by diagnostic data from the previous experiment, not random search.

**Priority ordering across the three config sections:**

```
Phase 1 — establish baseline (iteration 0)
  Start with: opset_version=17, all fusions enabled, precision=w8a16, minmax, 128 samples

Phase 2 — precision first (fastest to try, most impact)
  If cosine < floor:
    w8a16 → try w8a8 with selective exclusions, or w8a16 first
  If latency > budget:
    w8a16 → try w8a8 (smaller model, faster inference)
    fp16  → try w8a16 (if currently at fp16)

Phase 3 — calibration tuning (if precision is right but cosine still low)
  Try in order: minmax → entropy → percentile
  Try increasing samples: 128 → 256 → 512
  Try per_channel=true (better accuracy, slightly slower build)
  Try symmetric=false if currently true

Phase 4 — optimize pass tuning (independent of quant, affects graph structure)
  Hypothesis: some fusion patterns create op shapes QNN handles poorly
  Transformer models (try in order):
    attention-fusion → skip-layer-norm-fusion → layer-norm-fusion → fuse-rmsnorm
  Vision models (try in order):
    conv-bn-fusion → conv-add-fusion → conv-activation-fusion
  Shared (try if cosine drops or build crashes):
    constant-folding=false  (prevents size bloat; sometimes exposes EP-incompatible shape)
    clamp-constant-values=true  (fixes -inf attention mask → quantization issues)
    remove-isnan-in-attention-mask=true  (use after clamp; cleans dead IsNaN guards)
  Try opset_version: 17 → 18 → 19
    (Higher opsets expose newer op types that may have better EP support)

Phase 5 — selective node exclusion (when analyze shows partial ops)
  Read winml analyze --format json → partial_ops list
  Exclude one partial_op at a time (greedy: exclude highest-impact first)
  Also try excluding op_types_to_quantize selectively
    e.g., remove "LayerNorm" from op_types_to_quantize list

Phase 6 — combined search (if single-dimension changes are stuck)
  Try combinations of best Phase 3 + Phase 4 + Phase 5 changes together
```

**Diagnosis table — what to try given what you see:**

| Symptom | Likely cause | Phase to try next |
|---|---|---|
| cosine drops a lot at quant stage, all ops supported | Calibration data mismatch | Phase 3: entropy calib, more samples |
| cosine drops at quant, Attention ops partial | Attention activation quant on QNN | Phase 5: exclude Attention nodes |
| cosine OK but latency worse than CPU | Fusion pattern creating unoptimized subgraph | Phase 4: disable attention-fusion, try different opset |
| cosine OK but model larger than expected | Constant folding inlining large weights | Phase 4: constant-folding=false |
| Both cosine and latency good at w8a8 but build crashes | opset op not supported by quant pipeline | Phase 4: opset_version 17 → 16 |
| cosine highly variable across seeds | Calibration with too few samples | Phase 3: 128 → 256 samples |
| All ops supported, cosine still drops after fusions | Fusion creates non-quantizable shape | Phase 4: disable skip-layer-norm-fusion |
| QNN build fails with "invalid scale" | -inf in attention mask initializer | Phase 4: clamp-constant-values=true |
| Vision model: accuracy drops unexpectedly | Conv+BN fusion slightly changes weight values | Phase 4: disable conv-bn-fusion |
| MatMul-heavy model: latency not improving | MatMul not being fused | Phase 4: matmul-add-fusion, matmul-transpose-fusion |
| RMSNorm model (Llama etc.) poor QNN perf | ORT not recognizing RMSNorm pattern | Phase 4: fuse-rmsnorm=true |

This is the key difference from grid search: **each hypothesis is motivated by diagnostic data from `winml analyze` and the previous experiment result**.

---

**4. Multi-EP config generation**

Run parallel loops for each target EP, then aggregate into `manifest.json`:

```bash
# Agent runs loops for each EP (can be sequential or parallel):
# Loop 1: ep=qnn,   target_device=npu
# Loop 2: ep=dml,   target_device=gpu
# Loop 3: ep=cpu,   target_device=cpu

# After all loops complete, agent generates:
# - config_qnn_optimal.json   (best config found for QNN)
# - config_dml_optimal.json   (best config found for DirectML)
# - config_cpu_optimal.json   (best config found for CPU)

# Then builds final artifacts and assembles manifest.json
```

Generated `manifest.json` includes experiment provenance:
```json
{
  "model_id": "microsoft/resnet-50",
  "generated_by": "autoconfig",
  "experiments_run": 34,
  "variants": [
    {
      "ep": "qnn", "device": "npu",
      "file": "model_qnn.onnx",
      "precision": "w8a16",
      "nodes_excluded": ["MultiHeadAttention"],
      "cosine_similarity": 0.972,
      "p50_ms": 18.3,
      "config": "config_qnn_optimal.json"
    },
    {
      "ep": "dml", "device": "gpu",
      "file": "model_dml.onnx",
      "precision": "fp16",
      "nodes_excluded": [],
      "cosine_similarity": 0.999,
      "p50_ms": 22.1,
      "config": "config_dml_optimal.json"
    },
    {
      "ep": "cpu", "device": "cpu",
      "file": "model_cpu.onnx",
      "precision": "w8a8",
      "nodes_excluded": ["LayerNorm"],
      "cosine_similarity": 0.931,
      "p50_ms": 84.7,
      "config": "config_cpu_optimal.json"
    }
  ],
  "selection_order": ["qnn", "dml", "cpu"]
}
```

---

**5. results.tsv format**

Track all three config sections per experiment (TSV, not CSV):
```
commit	opset	fusions_disabled	precision	nodes_excluded	cosine	p50_ms	calib_samples	calib_method	status	notes
baseline	17	[]	fp32	[]	1.000	—	—	—	keep	FP32 reference
a1b2c3d	17	[]	w8a8	[]	0.871	16.2	128	minmax	discard	full W8A8 too aggressive
b2c3d4e	17	[]	w8a16	[]	0.967	19.8	128	minmax	keep	W8A16 baseline meets floor
c3d4e5f	17	[]	w8a16	[]	0.969	19.1	256	entropy	keep	entropy calib improvement
d4e5f6g	17	[attention-fusion]	w8a16	[]	0.971	18.4	256	entropy	keep	disabling attn-fusion helps latency
e5f6g7h	18	[attention-fusion]	w8a16	[]	0.973	17.9	256	entropy	keep	opset18 best so far
f6g7h8i	18	[attention-fusion]	w8a8	[MultiHeadAttention]	0.961	14.2	256	entropy	keep	mixed prec: meet latency budget
```

---

**6. Skill outputs**

autoconfig produces **two primary outputs** after convergence or user stop:

#### Output A: Best config file

`config_<ep>_optimal.json` — the winning config.json, ready to pass to `winml build`. Contains provenance metadata so it's reproducible:

```json
{
  "_autoconfig_meta": {
    "model_id": "facebook/convnext-tiny-224",
    "ep": "qnn",
    "objective": "latency-primary",
    "latency_budget_ms": 20,
    "accuracy_floor": 0.95,
    "experiments_run": 23,
    "best_iter": "iter_17",
    "timestamp": "2026-06-10T11:55:05+08:00"
  },
  "export": { "opset_version": 18 },
  "optimize": { "attention-fusion": false },
  "quantize": {
    "precision": "w8a16",
    "calibration_method": "entropy",
    "calibration_samples": 256,
    "nodes_to_exclude": ["MultiHeadAttention_0"]
  }
}
```

#### Output B: HTML benchmark report

`report.html` — self-contained single-file report (no external dependencies), viewable in any browser. Contains:

**Section 1 — Summary card**
```
Model:    facebook/convnext-tiny-224     EP: QNN (NPU)
Objective: latency-primary ≤ 20ms       Accuracy floor: 0.95
Result:   ✅ FOUND                       Experiments: 23  Time: 41 min

Best config:  W8A16, entropy calib, 256 samples
  Accuracy:   0.953  (floor 0.95 ✓)
  p50 latency: 15.8ms  (budget 20ms ✓)
```

**Section 2 — Search progress chart**
Scatter plot: all 23 experiments, x=p50_latency_ms, y=accuracy.
- Green dot = kept (improvement)
- Red dot = discarded (regression)
- Star = best found
- Hover tooltip: iter ID, config diff vs previous

**Section 3 — Iteration table**
Full results.tsv rendered as sortable HTML table with columns:
```
iter | opset | precision | nodes_excluded | calib | accuracy | p50_ms | Δacc | Δlatency | status | hypothesis
```
Color-coded rows: green = keep, red = discard, gold = best.

**Section 4 — Config diff timeline**
Visual diff showing what changed between each kept iteration (config deltas as `+`/`-` lines).

**Section 5 — Model graph analysis** (from pre-search `winml analyze`)
- Op distribution pie chart (ONNX vs com.microsoft)
- EP compatibility table: ops supported/unsupported on target EP
- Detected patterns (GELU variant, attention structure, Transpose-sandwich)

**Section 6 — Benchmark details**
For the best config, full `winml perf` output:
- p10/p50/p90/p99 latency histogram
- Throughput (samples/sec)
- Warmup vs steady-state comparison
- (If multi-EP: side-by-side EP comparison bar chart)

**Section 7 — Reproduction instructions**
```bash
# Reproduce the winning config:
winml build -c config_qnn_optimal.json -m facebook/convnext-tiny-224 -o out/
# For NPU: always compile after build (empirically +1.7× speedup)
winml compile -m out/model.onnx --device npu --ep qnn -o out_compiled/
winml perf -m out_compiled/model_npu_ctx.onnx --ep qnn --iterations 100 --warmup 10
```

**Report generation approach**: The agent generates report.html using inline Python with Jinja2-style string templating + embedded Chart.js (CDN or inlined). No external dependencies — single file, opens offline.

---

**7. What the agent says in chat**

After convergence or user stop (terminal summary, report is the real deliverable):

```
autoconfig completed. 23 experiments run (41 min).

Best config (QNN NPU):
  W8A16, entropy calib, 256 samples, MultiHeadAttention excluded
  accuracy 0.953 ✓ (floor 0.95)   p50 15.8ms ✓ (budget 20ms)

Outputs:
  config_qnn_optimal.json   ← drop into winml build -c
  report.html               ← open in browser for full benchmark breakdown

Next: winml validate-before-ship for production gate.
```

---

**8. Constraints and failure handling**

- **Build timeout**: If `winml build` exceeds 15 minutes, kill and log as crash
- **OOM**: If build fails with out-of-memory, reduce `calibration_samples` by half
- **All hypotheses exhausted**: Report best config found, note convergence limit
- **Latency not measurable** (target EP not on machine): run eval only, skip perf gate

**9. CLI-only constraint (critical)**

The agent MUST use only official `winml` CLI commands as its tool surface. No Python scripting, no direct ONNX manipulation, no third-party tools (onnxconverter-common, onnxsim, Olive, etc.) except where explicitly documented as a known workaround.

**Rationale**: autoconfig's output is a `config.json` + `report.html` that a user can reproduce with `winml build -c config.json`. If the agent used a Python hack to produce a model artifact, the config is not reproducible and the report is misleading.

**Known workarounds (allowed, must be flagged in report):**
| Workaround | Replaces | Tracking issue | Required flag in report |
|---|---|---|---|
| `python winml_profile.py` | `winml perf --profile` (not yet shipped) | pending | ⚠️ "Profile data via POC script, not official API" |

**Gap reporting rule**: If a hypothesis cannot be tested because the required `winml` CLI capability does not exist, the agent MUST:
1. Record the hypothesis as `SKIPPED — CLI gap` in the experiment table
2. Add an entry to **Section 6 "Gaps & Issues"** block in `report.html`:
   ```
   GAP: <hypothesis> requires <missing capability>
   Impact: <what speedup/accuracy improvement was not measurable>
   Filed: <issue URL or "not yet filed">
   ```
3. NOT silently substitute a Python workaround that produces unverifiable artifacts

**Example gaps encountered during ConvNext QNN GPU validation:**
- `winml build --precision fp16` flag not available (#867) → FP16 native export untested → `SKIPPED — CLI gap`
- `winml perf --ep-option` not available (#865) → runtime flag sweep untested → `SKIPPED — CLI gap`
- `winml perf --profile` for QNN EP not available → profiling via POC script (allowed workaround)
- W8A8 QDQ ONNX on QNN GPU EP hangs indefinitely — root cause is QNN SDK behavior; ``winml build`` already prevents this via ``_patch_device()``; fast-fail enhancement filed as #868 (low priority)

---

### Key commands used

```bash
# Phase 1: profiling (--profile flag on winml perf, before search)
winml perf -m baseline_built/model.onnx --ep <ep> --warmup 5 --iterations 20 \
  --profile --out profile_out/ --format json
# → profile_out/bottleneck.json  (machine-readable for Explorer)
# → profile_out/bottleneck.txt   (human-readable summary)
# POC: python winml_profile.py --model ... --ep ... (until --profile ships)

# Phase 2: analysis (informs nodes_to_exclude hypotheses)
winml analyze -m <exported>.onnx --ep <ep> --format json

# Phase 2: experiment
winml build -c config.json -m <model-id> -o out_<n>/

# Phase 2: metrics
winml eval --mode compare -m out_<n>/artifact.onnx --model-id <model-id> --format json
winml perf -m out_<n>/artifact.onnx --device <target> --ep <ep> --iterations 50 --format json

# Phase 3: compile best candidate to QNN EPContext (NPU only)
# Eliminates JIT overhead; empirically ~1.7× further speedup on ConvNext W8A16
winml compile -m best_candidate/model.onnx --device npu --ep qnn -o best_compiled/
# → best_compiled/model_npu_ctx.onnx  (loads context binary at runtime)
# → best_compiled/model_npu_ctx_qnn.bin  (QNN hardware-compiled graph)

# Phase 3: re-benchmark compiled model
winml perf -m best_compiled/model_npu_ctx.onnx --device npu --ep qnn --warmup 10 --iterations 50
```

**Empirical data: ConvNext QNN NPU compile impact**
| Version | p50 | vs FP32 NPU |
|---|---|---|
| FP32 baseline | 19.39ms | — |
| W8A16 quantized | 10.29ms | 1.9× |
| **W8A16 + compile** | **6.01ms** | **3.2×** |
→ `winml compile` alone adds ~1.7× on top of quantization. Always compile for NPU deployment.

**Empirical data: ConvNext QNN GPU optimization sweep (Adreno X1-85) — full search**
| Experiment | p50 | p90 | std | vs FP32 | Notes |
|---|---|---|---|---|---|
| FP32 baseline (autoconf) | **17.7ms** | 19.7ms | 0.97 | — | ✅ **OPTIMAL with current CLI** |
| NHWC transformer | 19.5ms | 23.8ms | 3.43 | ❌ −10% | Hurts Adreno+QNN EP |
| NHWC + all GPU fusions | 18.1ms | 23.9ms | 2.71 | ❌ −2% | Still worse |
| Conv/norm fusions (no NHWC) | 17.6ms | 22.6ms | 5.51 | ≈0% | Variance ↑, no gain |
| LayerNorm rewrite | 18.4ms | 21.4ms | 2.04 | ❌ −4% | Pattern mismatch anyway |
| Transpose optimizer | 0% node Δ | — | — | no-op | Already optimal positions |
| HiDimRTR→LowDimRTR | 0% node Δ | — | — | no-op | ConvNext RTR doesn't match pattern |
| MatMulAdd→Conv2D (2d/3d/4d) | 0% node Δ | — | — | no-op | ConvNext uses Reshape→MatMul, not bare MatMul+Add |
| FP32 + compile | 23.7ms | — | — | ❌ −34% | Compile hurts GPU (opposite of NPU) |
| W8A8 QDQ quantized | hangs | — | — | ❌ blocked | #868 enhancement (fast-fail) |
| FP16 (invalid CLI path) | 8.8ms | ~32ms | bimodal | ⚠️ 2× p50 | BLOCKED — need #867 |

**Root cause: why no pass matches ConvNext on QNN GPU**
- All 251 ops run natively on GPU (251/0/0/0) — no CPU fallback to eliminate
- ConvNext linear layers: `Reshape → MatMul → Reshape` pattern, not bare `MatMul+Add` → Conv2D rewrites don't match
- 72 Reshape + 42 Transpose are already at minimum / optimal topology from PyTorch export
- `winml build` autoconf (gelu_fusion + matmul_add_fusion) already applied all relevant transforms
- The bottleneck is compute throughput + memory bandwidth — only FP16 (smaller tensors) can improve this

**Key insight: gelu_fusion matters for variance, not p50**
| Version | p50 | p90 | std |
|---|---|---|---|
| Raw export (287 nodes, unfused Gelu) | 17.4ms | 29.2ms | 5.90 |
| Autoconf (251 nodes, fused Gelu+Gemm) | 17.7ms | 19.7ms | 0.97 |

Unfused Gelu = 5 separate GPU kernel launches (Mul→Div→Erf→Mul→Add) with scheduling jitter.
A single `Gelu` kernel eliminates dispatch overhead → p90 −48%, std −6×.
→ autoconf's role on GPU is **stability**, not speedup. Critical for real-time / latency-SLA deployments.

→ **QNN GPU search space exhausted.** FP16 is the only remaining lever, blocked by #867.

**Empirical data: ConvNext DML optimization sweep (Adreno X1-85, DirectML)**
| Experiment | p50 | p90 | std | vs FP32 |
|---|---|---|---|---|
| FP32 baseline (autoconf, 251 nodes) | **16.9ms** | 17.7ms | 0.52 | — ← OPTIMAL with current CLI |
| NHWC transformer | 16.5ms | 21.0ms | 1.89 | ❌ p90 worse |
| Raw unfused export (287 nodes) | 16.5ms | 18.4ms | 2.74 | ❌ p99=35ms, worse tail |
| FP16 (Python hack ⚠️) | **11.8ms** | 12.8ms | 0.66 | ✅ **1.4× faster, clean dist** — BLOCKED #867 |

**DML vs QNN GPU comparison (same Adreno X1-85):**
| | QNN GPU FP32 | DML FP32 | DML FP16 (invalid) |
|---|---|---|---|
| p50 | 17.7ms | **16.9ms** | **11.8ms** |
| p90 | 19.7ms | **17.7ms** | **12.8ms** |
| std | 0.97 | **0.52** | **0.66** |

→ DML is consistently faster and more stable than QNN GPU at FP32. Root cause: DML JIT-compiles HLSL shaders at model load time; QNN GPU EP does graph partitioning at each session creation.
→ DML FP16: no DVFS bimodal (unlike QNN GPU FP16) — DML's shader compilation locks in FP16 compute paths.
→ NHWC hurts DML too (same reason as QNN GPU: Adreno X1-85 + D3D12 doesn't benefit from explicit NHWC transforms).
→ Note: `winml analyze` returns 0/0/0/251 (all Unknown) for DML — no rule data. DML supports all standard ONNX ops by design.

**QNN Hub benchmark comparison (Snapdragon X Elite CRD) — WITH cross-stack test**

| Model | Stack | NPU p50 | GPU p50 | Notes |
|---|---|---|---|---|
| QNN Hub Float (opset 21, 222 nodes, MatMul) | qairt cloud | **2.687ms** | — | Reference |
| QNN Hub Float (same model) | winml ORT QNN EP | **8.78ms** | 23.9ms | Direct test on this device |
| Our Float (opset 17, 251 nodes, Gemm) | winml ORT QNN EP | 19.4ms | 17.7ms | winml build output |
| QNN Hub W8A16 (opset 21, 798 QDQ, uint16 input) | qairt cloud | **2.612ms** | — | Reference |
| QNN Hub W8A16 (same model) | winml ORT QNN EP | 14.82ms (std=8.8!) | — | ORT-QNN mismatch |
| Our W8A16 + compile (opset 17, ORT quant) | winml ORT QNN EP | **6.01ms** | — | Best we can do |

**Gap decomposition (three independent sources):**
```
QNN Hub cloud:   2.7ms
                  ↑ 3.3× Runtime gap  (qairt native vs ORT QNN EP adapter overhead)
QNN Hub on winml: 8.78ms
                  ↑ 2.2× Model graph gap (opset 21/MatMul/222 nodes vs opset 17/Gemm/251 nodes)
Our model on winml: 19.4ms (FP32)
```

**Actionable findings (updated 2026-06-10 — mechanism confirmed via ORT source):**
1. **opset 21 NPU speedup mechanism CONFIRMED — but ORT-version-dependent** (#869)
   - **Root cause**: `kMaxSupportedOpset` gate in `IsSupportedOpset()` (layout_transformation.cc). On older ORT where `kMaxSupportedOpset` < 21, opset 21 models bypass the NHWC layout transform entirely (`transform_layout_fn = nullptr`).
   - **Why bypass helps ConvNext**: NHWC transform inserts `Transpose(NCHW→NHWC/NHWC→NCHW)` around Conv. ConvNext residual connections **block** full transpose cancellation → extra Transpose ops on HTP → slower. Bypassing = cleaner graph = faster.
   - **Critical caveat**: Current ORT main has `kMaxSupportedOpset = 26` → BOTH opset 17 and 21 get NHWC transform. **Must verify ORT version** before assuming the speedup exists.
   - **Does NOT generalize** to: MobileNet/EfficientNet (no residual Transpose blocks), ViT (no Conv).
   - **Perf claim validation status**: Gate 1 (iter≥1000×3) and Gate 3 (thermal control) still FAILED. Perf numbers are DVFS-dominated.
2. **Runtime stack gap (3.3×) is structural**: qairt native will always be faster. Correct baseline = "QNN Hub ONNX on winml" (8.78ms).
3. **QNN Hub W8A16 is WORSE on our stack** (14.82ms, std=8.8ms): opset 21 QDQ + uint16 input incompatible with ORT QNN EP format.
4. **Opset is a search dimension** — but the correct action is a FULL SWEEP (17–22), not "try 21 first". The optimal opset depends on ORT version.

**EP-specific search space rules**

| EP | Quantization | Opset | Graph passes | Compile | Key insight |
|---|---|---|---|---|---|
| QNN NPU | ✅ W8A16 | Full sweep 17-22 (mechanism ORT-version-dependent) | autoconf (gelu+matmul_add) | ✅ Always | W8A8 catastrophic on LN+GELU; opset effect depends on ORT kMaxSupportedOpset |
| QNN GPU | ❌ Skip | 17 (opset 21 not validated) | autoconf only | ❌ Skip | Compile regresses; FP16 only lever (#867) |
| DML | ❌ Skip | 17 (opset 21 not validated) | autoconf only | N/A | FP16 primary lever (#867); faster+stabler than QNN GPU |
| CPU | ❌ Skip | 17 only (kMaxSupportedOpset causes 3-4× regression on 19+) | nchwc, matmul-add, gelu | N/A | kMaxSupportedOpset gate hurts CPU for same reason it helps QNN |

Rule: autoconfig must use EP-specific search space. Do NOT run quantization experiments for GPU/DML/CPU.
Rule: for QNN NPU opset sweep, verify ORT `kMaxSupportedOpset` first — if ≥ 22, all opsets get NHWC transform and the opset-based speedup may not apply.
Rule: for NPU, if W8A8 top-1 ≤ 15% on first attempt → skip all W8A8 variants, go directly to W8A16.
Rule: always run `winml compile` after finding best quantized config for QNN NPU. NEVER compile for GPU (regresses).
Rule: for GPU/DML, skip ALL graph optimization passes beyond what `winml build` autoconf applies (NHWC and additional fusions hurt).
Rule: W8A8 QDQ on GPU EP hangs — skip quantization immediately for GPU targets without testing.

**User scenario mapping**

| Scenario | How autoconfig addresses it |
|---|---|
| S1: LLM fast support (7-30d) | autoconfig replaces manual per-EP tuning; outputs `config_optimal.json + report.html` deployable in hours not days |
| S2: ISV non-LLM model support | Exact use case: ISV brings model → autoconfig finds config → report is deliverable with SOP turnaround |
| S3: Cross-EP parity | Multi-EP parallel run: same model, EP-specific search spaces in parallel → output config matrix per EP |
| S4: Customer ONNX can't run | Phase 0 intake diagnoses "can't run" (partial ops → block reason); Phase 1+2 finds "escape config" for "runs poorly" |
| S5: PyTorch HF Hub coverage | Phase 0 IS the "can WinML run it?" gate; failed Phase 0 → structured block reason feeds long-tail gap tracking |

**Dependencies on code changes**:
- `winml perf --profile` (new flag) — adds per-op bottleneck output alongside existing latency metrics; POC script `winml_profile.py` exists to unblock
- `--format json` on `winml eval` (#847), `winml analyze` (#848), `winml perf` (#849)

### Cross-references
- Run `ep-compatibility-check` before starting to verify EP is available
- After autoconfig completes → `validate-before-ship` for final production gate
- If autoconfig cannot meet objective → `debug-accuracy-drop` for deeper diagnosis
- Multi-EP output feeds directly into `prepare-for-winapp` manifest layout
- If the best config found is still not good enough → escalate to `optimization-research`

---

## Skill 11: `optimization-research` (internal — deep gap analysis)

### Frontmatter
```yaml
name: optimization-research
description: >
  Use this skill when a winml-cli engineer wants to find out whether a model can
  be optimized better than what winml-cli currently achieves, identify what is
  blocking that optimization, and produce concrete backlog work items.
  The agent performs a deep search across: ORT source code and its optimizer
  passes, Olive recipes and benchmarks, other ONNX ecosystem tools (onnxsim,
  onnxoptimizer, neural-compressor, etc.), and native stack reference models
  and datasets. It compares the best achievable result (using all available tools)
  against what winml produces today, diagnoses the gap, and files GitHub issues
  with reproduction steps. Use when an internal engineer says "why is this model
  slower than it should be", "what optimization techniques are we missing",
  or "what would it take to match Olive's results".

audience: internal (winml-cli team engineers)
```

### When to use
- "ConvNext on QNN is 3× slower than what Qualcomm's SDK achieves — why?"
- "Olive gets 15ms on this model; winml gets 28ms — what's the gap?"
- "We're seeing quantization accuracy drop on LLaMA; are there better calibration methods we're not supporting?"
- "What would it take to match ORT's best-known config for this architecture?"
- After `autoconfig` hits a ceiling: best config found is still not meeting the objective

### What this skill produces

**Primary outputs:**
1. **`gap_analysis.md`** — structured report of what the best achievable result is and what's missing
2. **`repro/`** — scripts to reproduce the better result using external tools
3. **GitHub issues** — one per identified gap, filed against winml-cli with: repro steps, expected vs actual, what ORT/Olive/ecosystem already does, proposed fix direction

---

### Design: Deep Search Process

```
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1 — BASELINE                                               │
│   winml autoconfig best result for this model/EP                 │
│   (or provided by user if already run)                           │
└─────────────────────────┬────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2 — EXTERNAL BENCHMARK                                     │
│   Run same model through:                                        │
│     A. ORT optimizer directly (onnxruntime.tools.transformers)   │
│     B. Olive (olive-ai) with ep-specific recipe                  │
│     C. onnxsim + onnxoptimizer (static graph simplification)     │
│     D. neural-compressor (Intel) for quantization comparison     │
│   Record: best latency, accuracy, config used                    │
└─────────────────────────┬────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 3 — GAP DIAGNOSIS                                          │
│   For each gap (external better than winml):                     │
│     a. Diff the ONNX graphs (what ops/patterns differ?)          │
│     b. Read ORT optimizer source to understand what it does      │
│     c. Check winml's capability registry — is this pass missing? │
│        disabled by default? wired incorrectly?                   │
│     d. Check Olive recipe — what flags/params does it use?       │
│   Classify gap as one of:                                        │
│     [MISSING_CAPABILITY]   — pass exists in ORT, not in winml   │
│     [WRONG_DEFAULT]        — pass exists but wrong default/order │
│     [BUG]                  — pass exists but produces wrong graph│
│     [CALIBRATION_DATA]     — accuracy gap from calibration set   │
│     [EP_LIMITATION]        — EP itself can't do this, not winml  │
│     [KNOWN_TRADEOFF]       — intentional: winml trades X for Y   │
└─────────────────────────┬────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 4 — NATIVE STACK VALIDATION                                │
│   Check existing reference models in winml-cli test suite:       │
│     - Are there models of this architecture in tests/models/?    │
│     - Do their expected results match what we see?               │
│   Check Windows AI Studio / WinML model zoo:                     │
│     - Is this architecture listed? At what performance?          │
│   Check QNN SDK reference benchmarks (if QNN EP):               │
│     - Does QNN vendor claim better numbers for this model?       │
└─────────────────────────┬────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 5 — WORK ITEMS                                             │
│   For each [MISSING_CAPABILITY] or [WRONG_DEFAULT] gap:          │
│     - Draft GitHub issue with: title, body, repro, expected,     │
│       actual, proposed fix, ORT source pointer                   │
│     - Estimate implementation complexity (S/M/L/XL)             │
│   For [BUG]: file with full repro script                         │
│   For [CALIBRATION_DATA]: suggest dataset and eval protocol      │
│   For [EP_LIMITATION]: file with QNN/DML SDK reference           │
└──────────────────────────────────────────────────────────────────┘
```

---

### Key external tools to invoke

```bash
# A. ORT transformer optimizer (the "gold standard" for transformer models)
python -c "
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
opts = FusionOptions('bert')   # or 'gpt2', 'clip', etc.
opts.enable_attention = True
opts.enable_gelu = True
model = optimizer.optimize_model(
    'export.onnx', model_type='bert',
    num_heads=12, hidden_size=768,
    optimization_options=opts
)
model.save_model_to_file('ort_optimized.onnx')
"

# B. Olive (end-to-end, EP-aware)
olive run --config olive_recipe.json
# olive recipe template: see skills/optimization-research/templates/olive_qnn.json

# C. onnxsim (structural simplification)
python -m onnxsim export.onnx simplified.onnx

# D. onnxoptimizer
python -c "
import onnxoptimizer, onnx
m = onnx.load('export.onnx')
passes = onnxoptimizer.get_available_passes()
m2 = onnxoptimizer.optimize(m, passes)
onnx.save(m2, 'onnxopt.onnx')
"
```

---

### Gap report format (`gap_analysis.md`)

```markdown
# Optimization Gap Analysis: <model_id> on <ep>

Date: <timestamp>
winml-cli version: <version>
ORT version: <version>

## Summary
| Tool | Latency p50 | Accuracy | Config notes |
|---|---|---|---|
| winml best (autoconfig) | 28.3ms | 0.953 | W8A16, entropy, 256 samples |
| ORT transformer optimizer | 19.1ms | 0.951 | model_type=bert, all fusions |
| Olive QNN recipe | 17.8ms | 0.948 | W8A8 + attention fusion |
| **Gap** | **10.5ms (37%)** | — | — |

## Gap 1: [MISSING_CAPABILITY] FusedMatMul with rotary embedding
**What external tool does:** ...
**What winml does:** ...
**ORT source:** `onnxruntime/python/tools/transformers/fusion_rotary_attention.py`
**Proposed fix:** Add RotaryAttentionFusion to FusionPipe capability registry
**Estimated effort:** M

## Gap 2: [WRONG_DEFAULT] attention-fusion disabled by default
...
```

---

### GitHub issue template

```markdown
title: [optimization-gap] <model_arch>/<ep>: <gap description>

body:
## Summary
<one-sentence description of what's missing>

## Reproduction
```bash
# Install
uv pip install winml-cli

# Baseline (winml current)
winml build -c config.json -m <model-id> -o winml_out/
winml perf -m winml_out/model.onnx --ep <ep> --warmup 10 --iterations 50

# Better result (external)
<commands to reproduce the external result>
```

## Expected vs actual
- External tool achieves: <latency>ms at <accuracy>
- winml achieves:         <latency>ms at <accuracy>
- Gap: <delta>ms (<pct>%)

## Root cause
<what the external tool does that winml doesn't>

## ORT source reference
<link to relevant ORT optimizer code>

## Proposed fix direction
<what capability / default change / bug fix would close this gap>

## Complexity estimate
S / M / L / XL
```

---

### What this skill does NOT do
- Does not make code changes to winml-cli itself (files issues only)
- Does not run production benchmarks (uses quick screening methodology)
- Does not replace formal performance testing with validated hardware

### Cross-references
- `autoconfig` provides the winml baseline to compare against
- Issues filed here feed `adding-ep-support` and `contributing-a-skill` workflows
- Use `ep-compatibility-check` to confirm EP availability before running external benchmarks

---


---

## ConvNext Autoconfig POC — Rigorous Ablation Results

**Source:** `C:\tmp\autoconfig-demo\ablation.py` — 4-phase rigorous ablation experiment
**Measurement:** `winml perf --ep cpu --warmup 10 --iterations 50` — pure inference latency, no preprocessing
**Design:** 3 independent runs per config; promotion threshold = max(3%, 2×σ_baseline); correctness gate (`winml eval --samples 20`) per config
**Report:** `C:\tmp\autoconfig-demo\report.html` | **Config:** `C:\tmp\autoconfig-demo\config_cpu_optimal.json`

### Graph structure (facebook/convnext-tiny-224, opset 17)

**Op counts (raw export):** 287 nodes total
```
Add×72  Mul×54  Transpose×42  MatMul×36  LayerNormalization×23
Conv×22  Div×18  Erf×18  ReduceMean×1  Gemm×1
```

**ConvNext block structure** (traced from first DW-Conv):
```
DW-Conv(7x7, g=96)  → Transpose
→ LayerNormalization (native, already fused at export)
→ MatMul(C→4C)      → Add(bias)
→ [GELU: Div → Erf → Add(1) → Mul → Mul(0.5)]   ← 18 unfused in export
→ MatMul(4C→C)      → Add(bias)   [Gemm after ORT L2]
→ Mul (layer scale) → Add (residual)
→ Transpose (back to NCHW)
```

**Conv breakdown:** 4 regular (1×stem 4x4, 3×downsample 2x2 stride-2), 18×DW-Conv 7x7

**Transpose patterns:**
```
19× Conv → Transpose → LayerNormalization     (NCHW→NHWC for LN)
15× Mul  → Transpose → Add                   (NHWC→NCHW for residual)
 4× LayerNormalization → Transpose → Conv    (NHWC→NCHW for next DW-Conv)
 2× Add  → Transpose → Conv
 2× Add  → Transpose → LayerNormalization
```
→ ConvNext is a **Transpose-sandwich** model: alternates NCHW (Conv) and NHWC (LN) layout

**Observed graph transformation (export.onnx → model.onnx after winml build, baseline config):**
| Op | export.onnx | model.onnx (baseline) | Change |
|---|---|---|---|
| `com.microsoft/Gelu` | 0 | 18 | +18 |
| `Gemm` | 1 | 37 | +36 |
| `MatMul` | 36 | 0 | −36 |
| `Add` | 72 | 18 | −54 |
| `Mul` | 54 | 18 | −36 |
| `Div`, `Erf` | 18 each | 0 | −18 each |
| `Reshape` | 0 | 72 | +72 |

**Observation (confirmed):** The baseline `model.onnx` (no user fusion flags) already differs substantially from `export.onnx`. GELU and MatMul+Add are fused before any user capability flag is applied.

**Open question (unresolved):** The `ORTGraphPipe` design (graph.py) is supposed to disable `GeluFusion`/`GeluFusionL2`/`LayerNormFusion` in the baseline via `optimization.disable_specified_optimizers`. Yet the baseline output clearly contains `com.microsoft/Gelu`. This contradiction is unresolved — possible explanations include: ORT name mismatch in disabled list, a different code path fusing GELU, or the export step (via HF Optimum) applying fusion before winml. **This must be investigated before any mechanistic claims about "ORT L2 already does X" are written in user-facing reports.**

---

### Ablation results (rigorous, Phase 0–4)

**Clean baseline:** 43.7ms p50 (base_0 + base_1, 6 runs, all within 42.5–45.4ms)

| config | p50 mean | Δ vs baseline | runs (ms) | verdict |
|---|---|---|---|---|
| base_0 | 43.0ms | −0.6ms | 43.8 / 42.7 / 42.5 | baseline |
| base_1 | 44.3ms | +0.6ms | 43.2 / 44.3 / 45.4 | baseline |
| base_2 | 73.5ms | +29.8ms | 47.2 / **127.1** / 46.2 | outlier run (system spike) |
| opset_18 | 48.0ms | +4.3ms | 50.2 / 44.0 / 49.7 | neutral |
| **opset_19** | **160.3ms** | **+116ms** | **147.6 / 145.8 / 187.4** | **⚠️ SEVERE REGRESSION** |
| **opset_20** | **131.0ms** | **+87ms** | **135.7 / 129.8 / 127.5** | **⚠️ SEVERE REGRESSION** |
| **opset_21** | **170.3ms** | **+126ms** | **190.1 / 164.9 / 155.8** | **⚠️ SEVERE REGRESSION** |
| **opset_22** | **85.0ms** | **+41ms** | **70.9 / 93.9 / 90.2** | **confirmed regression** |
| no_cf_17 | 51.8ms | +8.1ms | 56.4 / 49.0 / 49.9 | mild regression |
| base_mid | 49.4ms | +5.8ms | 51.3 / 51.1 / 45.9 | baseline (mid-exp drift) |
| gelu_only | 52.5ms | +8.9ms | 53.0 / 55.6 / 49.1 | mild regression |
| ln_only | 57.2ms | +13.6ms | **79.3** / 47.9 / 44.5 | inconclusive (outlier) |
| conv_add | 50.2ms | +6.5ms | 47.3 / 55.9 / 47.4 | inconclusive |
| conv_act | 51.2ms | +7.5ms | 45.2 / 41.9 / **66.4** | inconclusive (outlier) |
| **matmul_add** | **81.7ms** | **+38.0ms** | **63.0 / 70.8 / 111.2** | **CONFIRMED REGRESSION** |
| transpose_opt | 45.5ms | +1.8ms | 42.3 / 52.3 / 41.8 | neutral |
| nchwc | 45.4ms | +1.7ms | 43.4 / 48.0 / 44.7 | neutral |
| matmul_scale | 56.9ms | +13.3ms | 51.5 / 58.1 / 61.2 | probable mild regression |
| base_end | 48.3ms | +4.7ms | 45.3 / 56.7 / 43.1 | baseline (end-of-exp drift) |

**Phase 3 outcome:** No candidates met promotion threshold (29.4ms needed). Baseline is optimal.

---

### Confirmed findings (statistically defensible)

**1. `matmul-add-fusion` is a confirmed regression on ConvNext CPU (+38ms)**
- All 3 independent runs: 63.0 / 70.8 / 111.2ms — each far above the highest clean baseline run (45.4ms)
- Not attributable to system noise (no run-to-run overlap with baseline distribution)
- Mechanism hypothesis: baseline already converts MatMul+Add→Gemm (37 Gemm in model.onnx); applying matmul-add-fusion on top may create redundant or conflicting kernel dispatch. Unconfirmed — requires profiling.

**2. `transpose-optimizer` is NEUTRAL on pure inference latency**
- Runs: 42.3 / 52.3 / 41.8ms — overlapping with clean baseline (42.5–45.4ms)
- ⚠️ **CORRECTION OF EARLIER FINDING:** A previous 8-iteration search (using `winml eval`) reported +270ms. That was a measurement artifact — `winml eval` includes HF preprocessing pipeline overhead and has no warmup. It measures *application startup + preprocessing + inference*, not *inference alone*. With `winml perf` (warmup=10, iter=50, pure inference): transpose_opt = baseline. Do not cite the +270ms in any report.

**3. `nchwc-transformer` is neutral on this model**
- NCHWc SIMD layout: 43.4 / 48.0 / 44.7ms — no benefit for ConvNext CPU inference.

**4. opset=18 is neutral**
- Same node count (251) as opset=17 — no graph structure changes. Mean slightly above baseline (48ms) is within machine variance.

**5. No flag improved latency beyond noise. Baseline is the optimal config.**

---

### ⚠️ Critical finding: ORT performance cliff at opset 19 (ConvNext CPU)

**Experiment:** tested opset 17–22, all with identical graph structure (251 nodes, same op counts)

| opset | mean p50 | slowdown |
|---|---|---|
| 17 | 43.7ms | — (baseline) |
| 18 | 48.0ms | 1.1× |
| **19** | **160.3ms** | **3.7×** |
| **20** | **131.0ms** | **3.0×** |
| **21** | **170.3ms** | **3.9×** |
| **22** | **85.0ms** | **1.9×** |

**Key facts:**
- All runs within each opset are consistent (no outliers) — this is real, not noise
- Graph structure is **byte-for-byte identical**: Reshape×72, Transpose×42, Gemm×37, LN×23, Conv×22 for ALL opsets
- The performance difference is entirely in ORT's runtime execution path, not the graph

**Mechanism: CONFIRMED ROOT CAUSE — ORT `kMaxSupportedOpset` gates Transpose Optimizer**

Source: `onnxruntime/core/optimizer/transpose_optimization/optimizer_api.h`
```cpp
constexpr int64_t kMaxSupportedOpset = 18;  // ORT v1.14.x — bumped each ORT release
```

Entry point `onnx_transpose_optimization::Optimize()` → `MakeOptimizerContext()`:
```cpp
if (*opset > kMaxSupportedOpset) {
    return std::nullopt;  // entire Transpose Optimizer skipped silently
}
```

ConvNext has 42 Transpose nodes (NCHW↔NHWC sandwich in every block). The Transpose Optimizer normally:
- Pushes Transposes through Add×18, Mul×18 (layer-scale + residual) across block boundaries
- Cancels adjacent inverse pairs

When bypassed (opset > kMaxSupportedOpset), all 42 Transposes execute as full memory-layout copies → 3–4× systemic slowdown.

**ORT optimization level experiment (definitive proof):**

| Session opt level | opset=17 | opset=19 | ratio | explanation |
|---|---|---|---|---|
| DISABLE_ALL | 47.5ms | **355ms** | **7.5×** | No Transpose Optimizer → all 42 Transposes raw |
| ENABLE_BASIC | 289ms | 315ms | 1.1× | Both slow (re-optimizing pre-fused graph) |
| ENABLE_EXTENDED | 209ms | 241ms | 1.2× | Better but no layout transform |
| **ENABLE_ALL** | 216ms | **215ms** | **1.0×** | Transpose Optimizer runs on both → full parity |

**`kMaxSupportedOpset` version history:**

| ORT version | kMaxSupportedOpset | opset ≥ N disabled |
|---|---|---|
| v1.14.x | **18** | ≥ 19 |
| v1.16.x | 19 | ≥ 20 |
| v1.17.x | 20 | ≥ 21 |
| v1.18.x | 21 | ≥ 22 |
| main/HEAD | **26** | fully covered |

**Classification for optimization-research skill:** `[KNOWN_TRADEOFF]` (intentional design: ORT bumps the ceiling with each ONNX opset release)
- winml-cli ships a specific ORT build → its `kMaxSupportedOpset` is fixed
- winml-cli's **default opset=17 is correct and essential** — it is the safe zone for all current ORT builds
- Raising opset requires ensuring the shipping ORT version has `kMaxSupportedOpset ≥ target_opset`
- Do NOT raise default opset without verifying `kMaxSupportedOpset` in the shipped ORT

**Call chain:**
```
InferenceSession::Initialize()
  → TransposeOptimizer::ApplyImpl()         [transpose_optimizer.cc:18]
      → onnx_transpose_optimization::Optimize()
          → MakeOptimizerContext()
              → if opset > kMaxSupportedOpset: return nullopt  ← THE GATE
```

---

### Inconclusive / do not report

These show elevated means but cannot be confirmed as regressions given machine variance (p90 = 2–3× p50 throughout):
- `ln_only`, `conv_add`, `conv_act`: each has ≥1 extreme outlier run; other runs are baseline-level
- `gelu_only`: consistently 49–56ms, possibly a mild regression but no outlier; 3 runs insufficient to separate from drift
- `matmul_scale`: all 3 runs elevated (51–61ms), but concurrent baseline also drifted (+5ms); net delta ~+8ms, weak signal

Do not write these as confirmed regressions in user-facing reports. Label as "inconclusive" or omit.

---

### Measurement methodology correction (winml eval vs winml perf)

| Tool | What it measures | Latency for ConvNext CPU |
|---|---|---|
| `winml eval` (no warmup, includes preprocessing) | Application-level: model load + HF preprocessing + inference × N | ~67ms/sample |
| `winml perf --warmup 10 --iterations 50` | Pure inference: steady-state kernel execution only | ~43.7ms p50 |
| Difference | HF preprocessing + JIT warmup overhead | ~23ms |

**Rule for autoconfig skill:** Always use `winml perf` with `--warmup 10 --iterations 50` for latency measurements in experiments. Never use `winml eval` latency to compare configs.

---

### Key insight for autoconfig skill

- CPU EP on ConvNext: no extra flag tested improved latency. Baseline (no fusions beyond what ORT L2 applies unconditionally) is optimal.
- The only actionable finding is: **do not add `matmul-add-fusion` for ConvNext on CPU** (or any model where baseline already uses Gemm).
- QNN/DML: not yet tested. Guidance on those EPs requires separate validated experiments.

---

### `winml analyze` gaps discovered

These are cases where analyzing the graph *before* running autoconfig would have prevented wasted search iterations:

**Gap 1: "Already fused" vs "fuseable" not distinguished**
- ConvNext has `LayerNormalization` as a native op (already fused at PyTorch export)
- `layer-norm-fusion` targets the *decomposed* ReduceMean→Sub→... pattern
- `winml analyze` reports `OP/ai.onnx/LayerNormalization` without indicating it's already in canonical form
- **Impact:** user enables `layer-norm-fusion` thinking it will help; it does nothing (but builds take longer)
- **Fix:** analyze should tag ops as `already_canonical` vs `fuseable_subgraph`

**Gap 2: DW-Conv not distinguished from regular Conv**
- ConvNext has 18×7x7 DW-Conv (group=C) and 4×regular Conv (group=1)
- `winml analyze` reports all as `OP/ai.onnx/Conv` (undifferentiated)
- QNN EP supports DW-Conv natively (important for NPU efficiency), but EP support classification is per op type, not per `groups` value
- **Impact:** user cannot tell whether Conv ops are the DW or regular variant; EP support may differ
- **Fix:** analyze should emit `OP/ai.onnx/Conv[depthwise]` vs `OP/ai.onnx/Conv[regular]`

**Gap 3: Transpose-sandwich pattern not detected**
- 42 Transpose nodes in ConvNext form a clear `Conv→Transpose→LN→...→Transpose` repeating pattern
- `transpose-optimizer` turns this into NHWC chains (good for GPU/NPU, bad for CPU)
- `winml analyze` reports Transpose as just `OP/ai.onnx/Transpose` with no structural context
- **Impact:** user cannot predict whether `transpose-optimizer` will help or hurt without running it
- **Fix:** analyze should detect `transpose_sandwich_depth: N` and emit a warning for CPU EP

**Gap 4: ORT L2 baseline fusions not surfaced**
- After ORT Level 2 optimization (which runs unconditionally), the graph already has fused Gelu, Gemm
- The analyze command runs on the *pre-optimize* export.onnx, not the actual optimized model
- `winml analyze` sees 36×MatMul in export.onnx but the real model at inference has 37×Gemm
- **Impact:** analyze output doesn't reflect what the model actually looks like when running
- **Fix:** analyze should optionally run on `optimized.onnx` (post-ORT-L2), not just `export.onnx`

**Gap 5: MatMul semantic not classified**
- 36 MatMul ops are all MLP dense layers (4C→C or C→4C expansion)
- No attention MatMuls present (ConvNext has no self-attention)
- QNN handles dense-layer MatMul differently from attention-context MatMul
- `winml analyze` reports `OP/ai.onnx/MatMul` without semantic classification
- **Fix:** analyze could detect MatMul role heuristically (shapes: attention = square-ish, MLP = wide fan-out)

---



### Why skill eval matters

Mobius has no skill eval mechanism — it tests models but not skills themselves. This is a gap.
A SKILL.md can have correct content but still cause the agent to give wrong guidance if the
trigger description is poorly written or the structure is confusing. Skill eval catches this.

### Two eval dimensions

| Dimension | What it checks | When to run |
|---|---|---|
| **Static (content quality)** | description trigger phrases, command accuracy, cross-reference validity | Every PR that modifies a SKILL.md |
| **Dynamic (agent behavior)** | Given a user scenario + skill injected, does the agent produce the right commands and diagnosis? | On significant content changes; periodically |

Static eval = the review checklist in `contributing-a-skill`.
Dynamic eval = test cases in `evals/eval.yaml` per skill, run with `winml skill eval`.

### `winml skill` — new CLI subcommand

The eval system is built into winml-cli itself as a new `skill` subcommand.
This keeps the toolchain self-contained and enables CI integration without external dependencies.

**Command surface:**
```bash
winml skill check  [--skill <name>]   # static: lint + auto-verify all commands in SKILL.md
winml skill gen-evals [--skill <name>] # auto-research: generate eval.yaml from SKILL.md content
winml skill eval   [--skill <name>]   # dynamic: run agent behavior tests
winml skill list                      # list all skills with pass/fail status
```

#### `winml skill check` — auto-research via command extraction

This is the "code change that does auto research":

1. **Parse SKILL.md** — extract every code block containing `winml <command>` patterns
2. **Verify flags exist** — run `winml <command> --help` and check each flag is present
3. **Verify cross-references** — confirm every `.agents/skills/<name>/SKILL.md` path exists
4. **Verify trigger coverage** — count quoted phrases in `description` frontmatter (must be ≥3)
5. **Optionally run commands** — with `--dry-run-commands`, execute each command on a
   canary model to verify it doesn't crash

Example output:
```
winml skill check --skill debug-accuracy-drop

Checking debug-accuracy-drop...
  ✓ description: 4 trigger phrases found
  ✓ winml eval --mode compare     [flag verified against eval --help]
  ✓ winml analyze -m ... --ep qnn [flag verified against analyze --help]
  ✗ winml perf --monitor          [flag '--monitor' not found in perf --help]  ← STALE
  ✓ cross-ref: ep-compatibility-check/SKILL.md exists
  ✗ cross-ref: validate-before-ship/SKILL.md [file missing]  ← BROKEN LINK
Summary: 2 issues found
```

Key insight: **every time winml-cli flags change, `winml skill check` automatically
detects which skills have stale commands** — no manual audit needed.

Implementation sketch (`src/winml/modelkit/commands/skill.py`):
```python
import re, subprocess
from pathlib import Path
import click

SKILLS_DIR = Path(__file__).parents[5] / "skills"
WINML_CMD_PATTERN = re.compile(r'^\s*(winml\s+\w[\w\-]*\s+[^\n]+)', re.MULTILINE)

def extract_commands(skill_md: str) -> list[str]:
    """Extract all 'winml <subcommand> ...' lines from code blocks."""
    in_block = False
    commands = []
    for line in skill_md.splitlines():
        if line.strip().startswith("```"):
            in_block = not in_block
        elif in_block and line.strip().startswith("winml "):
            commands.append(line.strip())
    return commands

def verify_flag(command_line: str) -> tuple[bool, str]:
    """Check flags in a command line exist in --help output."""
    parts = command_line.split()
    subcommand = parts[1]
    flags = [p for p in parts[2:] if p.startswith("--")]
    result = subprocess.run(["winml", subcommand, "--help"],
                            capture_output=True, text=True)
    help_text = result.stdout
    for flag in flags:
        if flag not in help_text:
            return False, f"flag '{flag}' not found in {subcommand} --help"
    return True, "ok"

@click.group("skill")
def skill_cmd():
    """Manage and evaluate winml-cli skills."""

@skill_cmd.command("check")
@click.option("--skill", default=None, help="Skill name to check (default: all)")
@click.option("--dry-run-commands", is_flag=True, help="Execute commands on canary model")
def check(skill, dry_run_commands):
    """Static check: verify commands and cross-references in SKILL.md files."""
    targets = [SKILLS_DIR / skill] if skill else list(SKILLS_DIR.iterdir())
    for skill_dir in targets:
        skill_md = (skill_dir / "SKILL.md").read_text()
        for cmd in extract_commands(skill_md):
            ok, msg = verify_flag(cmd)
            status = "✓" if ok else "✗ STALE"
            click.echo(f"  {status}  {cmd[:60]}")
```

#### `winml skill gen-evals` — LLM-powered eval case generation

Auto-generates `evals/eval.yaml` from SKILL.md content using an LLM:

1. **Extract trigger phrases** from `description` frontmatter
2. **Extract symptom→fix tables** from SKILL.md sections
3. **Prompt an LLM** to generate (user scenario, expected commands) pairs
4. **Write `evals/eval.yaml`** in PromptFoo format

This is "auto research": the LLM reads the skill and generates adversarial cases
that challenge the agent — including negative cases where the agent should NOT
recommend something.

```bash
winml skill gen-evals --skill debug-accuracy-drop --model gpt-4o --count 5
# Writes: skills/debug-accuracy-drop/evals/eval.yaml (auto-generated)
# Human review before committing
```

The generated eval.yaml is a starting point — contributors review and refine before
committing. Over time, real user questions (from GitHub issues) can be mined and
added as additional eval cases.

#### `winml skill eval` — agent behavior testing

Runs the eval cases and reports results:

```bash
winml skill eval --skill debug-accuracy-drop
# Uses evals/eval.yaml + injects SKILL.md as system prompt
# Reports pass/fail per test case
```

Internally shells out to PromptFoo (if installed) or uses a lightweight built-in runner
that calls the configured LLM API directly.

### Directory layout

Each skill carries its own eval cases:
```
skills/
  debug-accuracy-drop/
    SKILL.md
    evals/
      eval.yaml     ← agent behavior test cases (hand-written or gen-evals output)
```

### eval.yaml format (PromptFoo)

```yaml
# skills/debug-accuracy-drop/evals/eval.yaml
description: "Agent behavior eval for debug-accuracy-drop skill"

prompts:
  - "{{user_message}}"

providers:
  - id: openai:gpt-4o
    config:
      systemPrompt: |
        You are a WinML CLI assistant. Use the following skill:
        ---
        {{skill_content}}

tests:
  - description: "Low cosine after W8A8 — should isolate to quantize stage"
    vars:
      user_message: "I quantized my model to W8A8 and cosine similarity is 0.87. What's wrong?"
    assert:
      - type: contains
        value: "winml eval --mode compare"
      - type: icontains
        value: "quantize"
      - type: icontains
        value: "w8a16"              # should suggest escalating precision

  - description: "NPU vs CPU discrepancy — should point to op fallback"
    vars:
      user_message: "My model gives different results on QNN NPU vs CPU after compile"
    assert:
      - type: contains
        value: "winml analyze"
      - type: icontains
        value: "partial"            # mention partial op fallback
      - type: icontains
        value: "compile"            # blame compile stage, not quantize

  - description: "Drop after optimize only — should NOT blame calibration"
    vars:
      user_message: "cosine similarity dropped after winml optimize, I haven't quantized yet"
    assert:
      - type: contains
        value: "winml eval --mode compare"
      - type: icontains
        value: "optimize"
      - type: not-icontains
        value: "calibration"        # calibration is irrelevant here
```

### Minimum eval cases per skill

| Skill | Min cases | Key assertions |
|---|---|---|
| `ep-compatibility-check` | 3 | Recommends 3-layer check in order; gives fallback when EP absent |
| `debug-accuracy-drop` | 4 | Correctly isolates pipeline stage; suggests precision escalation |
| `validate-before-ship` | 3 | Lists all 6 gates; handles waiver scenario |
| `optimize-for-device` | 3 | Applies latency-budget vs accuracy-budget framework correctly |
| `prepare-for-winapp` | 2 | Produces manifest.json structure; includes CPU fallback |
| `adding-model-support` | 2 | Suggests L1→L5 order; correct recipe structure |
| `contributing-a-skill` | 2 | Flags missing trigger phrases; flags pseudocode commands |

### What "passing" means

An eval case passes when all assertions hold. Recommended pass threshold before merging:
- All `contains` / `icontains` assertions pass
- All `not-icontains` (negative) assertions pass (agent does NOT give wrong advice)

The negative assertions are the most valuable — they catch the agent confidently giving
wrong guidance (e.g., blaming calibration for an optimize-stage drop).

### Running evals

```bash
# Install PromptFoo
npm install -g promptfoo

# Run eval for a single skill
cd skills/debug-accuracy-drop
promptfoo eval --config evals/eval.yaml

# Run all skill evals
for dir in skills/*/; do
  if [ -f "$dir/evals/eval.yaml" ]; then
    promptfoo eval --config "$dir/evals/eval.yaml"
  fi
done
```

---

## Implementation notes

### Directory structure
```
skills/
  use-winml-cli/              ← existing, extend
    SKILL.md
    evals/eval.yaml
  optimize-for-device/        ← new (consumer)
    SKILL.md
    evals/eval.yaml
  debug-accuracy-drop/        ← new (consumer)
    SKILL.md
    evals/eval.yaml
  prepare-for-winapp/         ← new (consumer, partial dep on winml package feature)
    SKILL.md
    evals/eval.yaml
  ep-compatibility-check/     ← new (consumer)
    SKILL.md
    evals/eval.yaml
  validate-before-ship/       ← new (consumer)
    SKILL.md
    evals/eval.yaml
  adding-model-support/       ← new (contributor)
    SKILL.md
    evals/eval.yaml
  adding-ep-support/          ← new (contributor)
    SKILL.md
    evals/eval.yaml
  contributing-a-skill/       ← new (contributor)
    SKILL.md
    evals/eval.yaml
  autoconfig/                 ← new (consumer — autoresearch loop for external users)
    SKILL.md
    evals/eval.yaml
  optimization-research/      ← new (internal — deep gap analysis for winml-cli team)
    SKILL.md
    templates/olive_qnn.json
    templates/olive_dml.json
    evals/eval.yaml
```

### Priority order for implementation
**Code changes first (unblocks agentic skill execution):**
0. `winml eval --format json` — critical: enables all accuracy-related agentic flows
0. `winml analyze --format json` — enables EP compatibility agentic flows
0. `winml perf --format json` — enables performance SLA agentic flows

**Consumer skills:**
1. `ep-compatibility-check` — lowest risk, pure existing commands, high value for new users
2. `debug-accuracy-drop` — closes clearest pain point, existing `eval --mode compare`
3. `validate-before-ship` — most complete checklist, builds on 1+2
4. `optimize-for-device` — needs good hardware reference data to be accurate
5. `prepare-for-winapp` — needs `winml package` feature or clear workaround documented
6. `autoconfig` — depends on #847/#848/#849 + most complex skill to implement

**Contributor skills:**
6. `contributing-a-skill` — enables community contributions to the skill ecosystem
7. `adding-model-support` — most impactful for model coverage growth
8. `adding-ep-support` — lower frequency, but needed for new EP onboarding

### Required code changes for agentic skill execution

The three changes that turn skills from documentation into agentic programs:

**1. `winml eval --format json`**

File: `src/winml/modelkit/commands/eval.py`

Add `--format` option and emit structured JSON to stdout:
```json
{
  "mode": "compare",
  "model": "path/to/quantized.onnx",
  "model_id": "microsoft/resnet-50",
  "metrics": {
    "cosine_similarity": 0.87,
    "sqnr_db": 28.3,
    "psnr_db": 31.1,
    "max_abs_diff": 0.042
  },
  "task_metric": { "top1_accuracy": 0.741 },
  "threshold_pass": false
}
```

**2. `winml analyze --format json`**

File: `src/winml/modelkit/commands/analyze.py`

Already supports `--output file.json`. Add `--format json` to also print to stdout
(mirrors pattern from `winml inspect` and `winml sys`):
```json
{
  "ep": "qnn",
  "model": "path/to/model.onnx",
  "summary": { "supported": 142, "partial": 3, "unsupported": 1 },
  "partial_ops": ["MultiHeadAttention", "LayerNorm", "Softmax"],
  "unsupported_ops": ["CustomRotaryEmbedding"]
}
```

**3. `winml perf --format json`**

File: `src/winml/modelkit/commands/perf.py`

Already writes JSON to file via `-o`. Add `--format json` stdout output:
```json
{
  "model": "path/to/model.onnx",
  "ep": "qnn",
  "device": "npu",
  "iterations": 100,
  "latency_ms": { "p50": 18.3, "p90": 21.7, "p99": 28.4, "mean": 18.9 },
  "throughput_rps": 54.6
}
```

These three changes are ~50 lines of code each, follow the existing pattern from
`winml inspect --format json` and `winml sys --format json`, and unlock the full
agentic execution model for all consumer skills.

### Sizing estimate (per skill)
Each SKILL.md based on Mobius patterns (~8–14KB):
- ~200 lines prose + decision tables
- ~50 lines code examples
- Cross-reference section

### Relationship to existing `use-winml-cli` skill
The new skills are **task-scoped** (problem → solution) vs the existing skill which is
**tool-scoped** (here's what each command does). They complement, not replace each other.
The existing skill should add cross-references to the new skills in its "Common patterns" section.

---

## QNN NPU Catalog Sweep — Findings & Feature Gaps (2026-06-13)

Source: 8-model catalog sweep via autoconfig POC (C:\tmp\autoconfig-demo\catalog_qnn_sweep.py)

### Cross-model results

| Model | Arch | Baseline p50 | Best p50 | Gain | Best config |
|-------|------|-------------|----------|------|-------------|
| microsoft/resnet-18 | resnet | 0.96ms | 0.96ms | — | baseline (opset17) |
| google/vit-base-patch16-224 | vit | 9.04ms | 9.04ms | — | baseline (opset17) |
| apple/mobilevit-small | mobilevit | 12.07ms | **8.62ms** | +29% | opset21+conv_fusions |
| facebook/dinov2-small | dinov2 | 6.56ms | **4.98ms** | +24% | opset21 |
| hustvl/yolos-small | yolos | 78.69ms | — | timeout | — |
| distilbert SST-2 | distilbert | 19.48ms | 19.48ms | — | baseline |
| all-MiniLM-L6-v2 | bert | 5.81ms | 5.81ms | — | baseline |
| deepset/roberta-base-squad2 | roberta | 14.94ms | 14.72ms | 1.5% | opset21 |

### Validated KB findings

**npu-001 refined**: opset21 benefit is architecture-gated:
- ✅ Conv + residual connections: +25–31% (mobilevit, dinov2, convnext)
- ❌ Pure transformer (ViT, YOLOS): -7% or neutral
- ⚪ NLP BERT-family: neutral

**npu-006 NEW — CRITICAL**: Conv fusions (conv-bn/add/activation) cause catastrophic QNN NPU CPU fallback
- ResNet-18 with conv fusions: 0.96ms → 132ms (+4900% regression)
- MobileViT: safe (no regression)
- Severity: critical — can produce 50x+ regression silently

**npu-007 NEW**: DVFS thermal noise makes CV gate unreliable on QNN NPU
- New bench protocol: 3 sessions × 500 iters + 30s cool-down + median p50 + >10% noise floor

### Feature gaps (winml-cli backlog items)

**Gap A: winml analyze — Conv fusion QNN safety check**
winml analyze should detect Conv-dominant topologies and warn when conv-bn/add/activation
fusions are configured for QNN NPU target. Currently no pre-build detection of this hazard.
- Command to add: warning in analyze output when ep=qnn AND conv_fusion_pass is enabled AND model has >N Conv ops
- Priority: HIGH (silent 50x regression risk)

**Gap B: budget-aware sweep in autoconfig**
Large models (YOLOS, ~78ms/inf) cause sweep timeout with current fixed budget.
Need: per-hypothesis time estimation → auto-skip models that exceed budget, log as "timeout" not failure.
- Affects: autoconfig POC and any future winml sweep command

**Gap C: winml perf DVFS-aware session averaging**
winml perf should natively support session-level median aggregation for QNN NPU.
Current single-session variance is dominated by DVFS thermal state, not model performance.
- Flag proposal: --sessions 3 --cool-down 30 --signal median-p50
- This would make winml perf output trustworthy for optimization decisions on Snapdragon X Elite

---

## Feature Request: FusedConv detection + unfuse-for-qnn (2026-06-15)

### Problem

用户可能从外部拿到一个已经做过 Conv fusion 的 ONNX 模型，或者 autoconfig 实验里开了 conv-add-activation-fusion flag。
这类模型在 QNN NPU 上跑起来特别慢（ResNet-18 实测 +4900% regression），但没有任何报错，用户完全不知道原因。

### Root cause

conv-add-activation-fusion 生成的是 ORT 扩展 op FusedConv（非标准 ONNX op）。
QNN EP 不认识这个 op，所有 FusedConv 节点全部 fallback 到 CPU，PCIe round-trip 开销极大。

conv-bn-fusion 不同：它把 BN 参数数学吸收进 Conv weight，不产生新 op 类型，结果仍是标准 Conv，**不可逆**。

### Proposed feature

**1. winml analyze — FusedConv detection**

winml analyze -m model.onnx --ep qnn 扫描图中所有节点，
如果发现 FusedConv 节点且目标 EP 为 QNN，输出警告：

`
⚠ QNN NPU: 23 FusedConv nodes detected.
  FusedConv is an ORT-internal op not supported by QNN EP — these nodes will fall back to CPU.
  Recommend: run winml optimize --unfuse-conv to expand back to standard ONNX ops.
`

**2. winml optimize --unfuse-conv**

新增 optimize pass：把 FusedConv 节点拆回 Conv + Add + <Activation>。
- Lossless（权重不变，只拆 op 结构）
- 输出标准 ONNX，QNN EP 可正常映射 HTP kernel
- 适用场景：BYOM 用户带入已做过 fusion 的模型

**Implementation notes**
- 检测：
ode.op_type == "FusedConv" 即可定位
- 拆分：读 FusedConv attribute ctivation 字段 → 插入对应 Relu/Sigmoid/Tanh 节点
- 不处理 conv-bn-fusion 产生的模型（那个无法反向，只能重新从 FP32 export）

### Priority
MEDIUM — 默认 flag 是关的，不是高频路径，但对 BYOM 场景（拿到别人优化过的模型）有实际价值。
