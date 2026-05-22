🐛 ModelKit Bug Bash Guide
=======================

[Reference: Modelkit BugBash](https://github.com/microsoft/ModelKit/blob/qiowu/bugbash/bug-bash-guide.md#modelkit-bug-bash-guide)

**Release**: v0.0.1.dev1 | **Date**: 2026-04-02

___

## Welcome!
Welcome to the **ModelKit Bug Bash** ! ModelKit is an **open‑source CLI on GitHub** that converts and optimizes PyTorch and Hugging Face models into high‑quality ONNX models for Windows ML. The current scope focuses on **classic deep learning models** (e.g., CNNs, vision transformers, NLP classifiers, segmentation). **LLMs and generative models are out of scope**—do not test GPT, LLaMA, Phi, Mistral, Stable Diffusion, or other decoder‑only or seq2seq generative architectures.

## 📋 Prerequisites 
### Required Software
| **Component** | **How to Get It** |
|-----------|--------------|
| **Windows 11** (x64 or ARM64) | Windows 11 24H2+ required for NPU support |
| **UV**|Install [UV](https://github.com/astral-sh/uv)|
|**Windows APP SDK Runtime 1.8**| [Latest Windows App SDK downloads - Windows apps](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads)|
| **Modelkit (python wheel)** | Download [winml_modelkit-0.0.1.dev1-py3-none-any.whl](https://microsoft.sharepoint-df.com/:u:/r/teams/WinPD/Shared%20Documents/Forms/Gallery.aspx?id=%2Fteams%2FWinPD%2FShared%20Documents%2FModelKit%2Fwinml%5Fmodelkit%2D0%2E0%2E1%2Edev1%2Dpy3%2Dnone%2Dany%2Ewhl&parent=%2Fteams%2FWinPD%2FShared%20Documents%2FModelKit&p=true&share=cQqnvDjbLu18QZ%5FhHSiX%2D2f3EgUCQzr1M%2DQKvecLbJEsxiAn7g)|

### Required Hardware
**This bug bash targets NPU only.** We recommend testing on one of the following NPU devices:
| Device | EP | Flag |
| --- | --- | --- |
| Snapdragon X Elite (Qualcomm) | QNN | `--ep qnn --device npu` |
| Intel AI Boost (Meteor Lake / Lunar Lake) | OpenVINO | `--ep openvino --device npu` |
| AMD Ryzen AI (Phoenix / Hawk Point / Strix) | VitisAI | `--ep vitisai --device npu` |
**No NPU?** Use `--device auto` — ModelKit will fall back to the best available device (GPU → CPU). Note that `winml compile` requires NPU and cannot run without one.

---

##⚡ Reporting Bugs
[Modelkit and Local Model Agent Skills Bugs.loop](https://microsoft.sharepoint.com/:fl:/s/b35b0ac6-fbb0-43c1-abcc-4f55dd436ab2/IQCb2-g6XPRRTbcEz5TGPcqaATQAIt_nFDWhOLj5BRXwISY?e=kmgnzR&nav=cz0lMkZzaXRlcyUyRmIzNWIwYWM2LWZiYjAtNDNjMS1hYmNjLTRmNTVkZDQzNmFiMiZkPWIlMjFSWno0enY0YzhFdVR1eXFBZlpoeFVDRmJ4NTkxelZKSG93TUM5X1NqLUpTQS1BUEpPQ2RaU2JCZzlVckV2YUkwJmY9MDE0VUhFUUlFMzNQVURVWEhVS0ZHM09CR1BTVEREM1NVMiZjPSUyRiZhPUxvb3BBcHAmcD0lNDBmbHVpZHglMkZsb29wLXBhZ2UtY29udGFpbmVy)

---
## 🧩 Modelkit Setup

Download the wheel file shared for this bug bash: `winml_modelkit-0.0.1.dev1-py3-none-any.whl`

```bash
# Create a Python 3.10 virtual environment
uv venv --python 3.10
.\.venv\Scripts\activate

# Install Modelkit from wheel
uv pip install '.\<your path>\winml_modelkit-0.0.1.dev1-py3-none-any.whl'

# Sanity check — verify NPU device and EP are available
winml sys --list-device --list-ep
```

> **Dependency**: ModelKit depends on onnxruntime-windowsml 1.23.x. Please ensure the Execution Providers (EPs) installed

> **NPU required for E2E tests.** Run `winml sys --list-device --list-ep` to confirm your NPU and EP are detected:
> - Snapdragon X Elite → `QNNExecutionProvider` listed
> - Intel AI Boost → `OpenVINOExecutionProvider` listed
> - AMD Ryzen AI → `VitisAIExecutionProvider` listed
>
> If no NPU is available, use `--device auto` for perf, config, build, and eval — ModelKit will fall back to the best available device. `winml compile` requires NPU and should be marked SKIP if no NPU is present.

---

##🤖 Quick Start and Core Feature Tests

### 📝 Path A: Run with Claude Code, Github Copilot.
Paste this prompt into your coding agent (Claude Code, Cursor, Copilot, etc.) to run everything automatically:

#### Prompt A — PERF: HuggingFace end-to-end only

Use this for a quick smoke test of the auto-pipeline perf path.

```text
I have winml-modelkit installed from wheel (`uv pip install winml_modelkit-0.0.1.dev1-py3-none-any.whl`) in the current venv.

Before starting, run `winml sys --list-device --list-ep` to identify your NPU type and note
the corresponding EP flag:
  - Snapdragon X Elite  → --ep qnn --device npu
  - Intel AI Boost      → --ep openvino --device npu
  - AMD Ryzen AI        → --ep vitisai --device npu

If no NPU is detected, use `--device auto` throughout.

MODEL SELECTION:
  Run `winml hub` to get the built-in model catalog. Randomly pick:
    - MODEL_A: one image-classification model
    - MODEL_B: one token-classification model
  Run `winml inspect -m <MODEL_A>` and `winml inspect -m <MODEL_B>`.
  If either fails, pick a different model.

Important: run all `winml` commands directly — do not pipe output (e.g. no `| tail`, `| head`, `| tee`).
The EP requires an unpiped process; piping causes a crash that looks like a real failure.

Run the following tests, capture output, and report pass/fail.
Stop and flag any failure before continuing.

1. SYSTEM INFO
   Run: winml sys --list-device --list-ep
   Pass: your NPU device and its EP are listed.

2. INSPECT
   Run: winml inspect -m MODEL_A
   Run: winml inspect -m MODEL_B
   Pass: model task, loader/exporter/inference class, and support status all printed.

3. PERF — HuggingFace end-to-end
   Run: winml perf -m MODEL_A --device npu --iterations 100
   Run: winml perf -m MODEL_B --device npu --iterations 100
   Pass: auto-runs full pipeline for each model; reports NPU latency and throughput.

4. PERF — live hardware monitor
   Run: winml perf -m MODEL_A --device npu --monitor --iterations 1000
   Pass: live NPU utilization chart shown during run; final latency table printed.

After all tests, produce a summary table:
  | # | Test | Model | Status | Notes |
  showing PASS / FAIL / SKIP for each item,
  including which MODEL_A, MODEL_B, and EP were used.
```

---

#### Prompt B — Config + Build + Perf with ONNX

Use this to test the config/build pipeline and then benchmark both the compiled ONNX and the HuggingFace auto-pipeline.

```text
I have winml-modelkit installed from wheel (`uv pip install winml_modelkit-0.0.1.dev1-py3-none-any.whl`) in the current venv.

Before starting, run `winml sys --list-device --list-ep` to identify your NPU type and note
the corresponding EP flag:
  - Snapdragon X Elite  → --ep qnn --device npu
  - Intel AI Boost      → --ep openvino --device npu
  - AMD Ryzen AI        → --ep vitisai --device npu

If no NPU is detected, use `--device auto` for all commands that accept a device flag.
`winml compile` (inside `winml build`) requires NPU; mark build steps SKIP if no NPU is present.

MODEL SELECTION:
  Run `winml hub` to get the built-in model catalog. Randomly pick:
    - MODEL_A: one image-classification model
    - MODEL_B: one token-classification model
  Run `winml inspect -m <MODEL_A>` and `winml inspect -m <MODEL_B>`.
  If either fails, pick a different model. Substitute EP_FLAGS with your device's EP flag.

Important: run all `winml` commands directly — do not pipe output (e.g. no `| tail`, `| head`, `| tee`).
The EP requires an unpiped process; piping causes a crash that looks like a real failure.

Run the following tests, capture output, and report pass/fail.
Stop and flag any failure before continuing.

1. SYSTEM INFO
   Run: winml sys --list-device --list-ep
   Pass: your NPU device and its EP are listed.

2. INSPECT
   Run: winml inspect -m MODEL_A
   Run: winml inspect -m MODEL_B --verbose
   Pass: model task, loader/exporter/inference class, and support status all printed.

3. CONFIG + BUILD — MODEL_A
   Run: winml config -m MODEL_A --device npu --precision int8 -o model_a_config/config.json
   Run: winml build -c model_a_config/config.json -m MODEL_A -o model_a_config/
   Pass: config.json generated; build completes all stages (export→optimize→quantize→compile) targeting NPU.

4. CONFIG + BUILD — MODEL_B
   Run: winml config -m MODEL_B --device npu --precision int8 -o model_b_config/config.json
   Run: winml build -c model_b_config/config.json -m MODEL_B -o model_b_config/
   Pass: config.json generated; build completes all stages targeting NPU.

5. PERF — direct ONNX (built artifact)
   Run: winml perf -m model_a_config/<compiled_filename>.onnx --device npu --iterations 100
   Run: winml perf -m model_b_config/<compiled_filename>.onnx --device npu --iterations 100
   Pass: reports P50/P90/Avg latency and throughput on NPU for each model.

6. PERF — HuggingFace end-to-end
   Run: winml perf -m MODEL_A --device npu --iterations 100
   Run: winml perf -m MODEL_B --device npu --iterations 100
   Pass: auto-runs full pipeline for each model; reports NPU latency and throughput.

After all tests, produce a summary table:
  | # | Test | Model | Status | Notes |
  showing PASS / FAIL / SKIP for each item,
  including which MODEL_A, MODEL_B, and EP_FLAGS were used,
  followed by any commands that need investigation.
```

---

#### Prompt C — Full pipeline (all features)

Use this for a complete end-to-end bug bash covering all commands.

```text
I have winml-modelkit installed from wheel (`uv pip install winml_modelkit-0.0.1.dev1-py3-none-any.whl`) in the current venv.
All tests target NPU. Before starting, run `winml sys --list-device --list-ep` to identify your NPU type and use
the corresponding EP flag throughout:
  - Snapdragon X Elite  → --ep qnn --device npu
  - Intel AI Boost      → --ep openvino --device npu
  - AMD Ryzen AI        → --ep vitisai --device npu

If no NPU is detected, use `--device auto` for all commands that accept a device flag — ModelKit will fall back
to the best available device (GPU → CPU). Only `winml compile` (step 8) requires NPU; mark that step SKIP if
no NPU is available.

Important: run all `winml` commands directly — do not pipe output (e.g. no `| tail`, `| head`, `| tee`).
The EP requires an unpiped process; piping causes a crash that looks like a real failure.

Run through the following ModelKit core feature tests in order, executing each command,
capturing its output, and reporting pass/fail with a brief summary. Stop and
flag any failure before continuing — do not skip errors silently.

MODEL SELECTION (do this before starting):
  Run `winml hub` to get the full built-in model catalog. From the output,
  randomly pick:
    - MODEL_A: one image-classification model
    - MODEL_B: one token-classification model (avoid text-classification — eval is broken, see known issue #216)
  Then run `winml inspect -m <MODEL_A>` and `winml inspect -m <MODEL_B>`.
  If either fails, pick a different model. Use these two models throughout
  all tests below (substitute wherever you see MODEL_A / MODEL_B).
  Also substitute EP_FLAGS with the EP flag for your device (e.g., --ep qnn --device npu).

1. SYSTEM INFO
   Run: winml sys
   Run: winml sys --list-device --list-ep
   Pass: your NPU device and its EP are listed.

2. INSPECT
   Run: winml inspect -m MODEL_A
   Run: winml inspect -m MODEL_B --verbose
   Pass: model task, loader/exporter/inference class, and support status all printed.

3. HUB
   Run: winml hub
   Run: winml hub --task image-classification
   Run: winml hub --model MODEL_A
   Pass: catalog table shown; per-model detail includes accuracy info.

4. EXPORT
   Run: winml export -m MODEL_A -o model_a/model.onnx
   Pass: ONNX file produced at the specified path; no errors.

5. ANALYZE
   Note: winml analyze requires --device NPU (uppercase). Replace the --device portion of EP_FLAGS with NPU.
   Run: winml analyze --model model_a/model.onnx --ep <your-ep> --device NPU
   Run: winml analyze --model model_a/model.onnx --ep <your-ep> --device NPU --information
   Run: winml analyze --model model_a/model.onnx --ep <your-ep> --device NPU --run-unknown-op
   Pass: operator compatibility report shown for your NPU EP; --run-unknown-op runs without crash.

6. OPTIMIZE
   Run: winml optimize --list-capabilities
   Run: winml optimize --list-rewrites
   Run: winml optimize -m model_a/model.onnx -o model_a/model_opt.onnx
   Pass: optimized ONNX produced; file size equal or smaller than input.

--- Steps 7–14: use --device npu if NPU is available, otherwise --device auto (except compile, which requires NPU). ---

7. QUANTIZE
   Run: winml quantize -m model_a/model_opt.onnx --precision int8 -o model_a/model_int8.onnx
   Run: winml quantize -m model_a/model_opt.onnx --weight-type int8 --activation-type uint16 -o model_a/model_w8a16.onnx
   Pass: each completes; output ONNX contains QDQ nodes.
   Note: re-running with the same -o path will crash with FileExistsError (known issue #185) — delete old output first.

8. COMPILE
   Run: winml compile --list
   Run: winml compile -m model_a/model_int8.onnx --output-dir model_a/compiled/ EP_FLAGS
   Run: winml compile -m model_a/model_int8.onnx --output-dir model_a/compiled_noquant/ EP_FLAGS --no-quantize
   Pass: compiled ONNX produced in output dir.

9. PERF — direct ONNX
   Run: winml perf -m model_a/compiled/<compiled_filename>.onnx --device npu --iterations 100
   Pass: reports P50/P90/Avg latency and throughput on NPU.

10. PERF — HuggingFace end-to-end
    Run: winml perf -m MODEL_A --device npu --iterations 100
    Run: winml perf -m MODEL_B --device npu --iterations 100
    Pass: auto-runs full pipeline for each model; reports NPU latency and throughput.

11. PERF — live hardware monitor
    Run: winml perf -m MODEL_A --device npu --monitor --iterations 1000
    Pass: live NPU utilization chart shown during run; final latency table printed.

12. CONFIG + BUILD — MODEL_A
    Run: winml config -m MODEL_A --device npu --precision int8 -o model_a_config/config.json
    Run: winml build -c model_a_config/config.json -m MODEL_A -o model_a_config/
    Pass: build completes all stages (export→optimize→quantize→compile) targeting NPU.

13. CONFIG + BUILD — MODEL_B
    Run: winml config -m MODEL_B --device npu --precision int8 -o model_b_config/config.json
    Run: winml build -c model_b_config/config.json -m MODEL_B -o model_b_config/
    Pass: config.json generated; build completes all stages targeting NPU.

14. EVAL
    Note: Only the following models support a built-in default eval dataset:
      image-classification: microsoft/resnet-50, facebook/convnext-tiny-224
      text-classification:  Intel/bert-base-uncased-mrpc
      token-classification: dslim/bert-base-NER, dbmdz/bert-large-cased-finetuned-conll03-english,
                            Babelscape/wikineural-multilingual-ner
    Other models require a custom --dataset config. Use one of the above as MODEL_A/MODEL_B,
    or skip this step if neither model is in the list above.
    Run: winml eval -m MODEL_A --device npu --samples 100
    Run: winml eval -m MODEL_B --device npu --samples 100
    Pass: accuracy metric reported without error (uses auto-detected dataset).

After all tests, produce a summary table:
  | # | Test | Model | Status | Notes |
  showing PASS / FAIL / SKIP for each item,
  including which MODEL_A, MODEL_B, and EP_FLAGS were used,
  followed by any commands that need investigation.
```

### 📝 Path B: Run CLI in your terminal

Run through each section below and note any failures. Report issues with the exact command, output, and machine spec.
#### Accepted inputs

- **HuggingFace model ID** (e.g., `microsoft/resnet-50`) — model weights are downloaded on first run.
- **Local ONNX file** (e.g., `model.onnx`) — produced by `winml export` or `winml build`, or any ONNX file you already have on hand.

#### The golden rule: inspect first

Before running any pipeline command on a model, always verify it is supported:

```bash
winml inspect -m <model-id>
```

If `inspect` prints an error or shows `Unsupported`, **stop and skip that model**. Only models that pass inspect are valid inputs for export, analyze, perf, build, and eval.

#### Recommended test models

Use models from the built-in catalog (`winml hub`) or from the following.

**Built-in hub models** (`winml hub` to list all):

| Model ID | Task | Architecture |
|----------|------|--------------|
| `microsoft/resnet-50` | image-classification | resnet |
| `google/vit-base-patch16-224` | image-classification | vit |
| `microsoft/swin-large-patch4-window7-224` | image-classification | swin |
| `facebook/convnext-tiny-224` | image-classification | convnext |
| `rizvandwiki/gender-classification` | image-classification | vit |
| `ProsusAI/finbert` | text-classification | bert |
| `Intel/bert-base-uncased-mrpc` | text-classification | bert |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | text-classification | roberta |
| `dslim/bert-base-NER` | token-classification | bert |
| `dbmdz/bert-large-cased-finetuned-conll03-english` | token-classification | bert |
| `Babelscape/wikineural-multilingual-ner` | token-classification | bert |
| `w11wo/indonesian-roberta-base-posp-tagger` | token-classification | roberta |
| `microsoft/table-transformer-detection` | object-detection | table-transformer |
| `mattmdjaga/segformer_b2_clothes` | image-segmentation | segformer |
| `nvidia/segformer-b1-finetuned-ade-512-512` | image-segmentation | segformer |
| `nvidia/segformer-b2-finetuned-ade-512-512` | image-segmentation | segformer |
| `nvidia/segformer-b5-finetuned-ade-640-640` | image-segmentation | segformer |


---

### 💡 Using one command to build models

### Perf (`winml perf`)

```bash

# HuggingFace end-to-end (auto-pipeline) on NPU
winml perf -m microsoft/resnet-50 --device npu --iterations 100

# Live NPU utilization monitor
winml perf -m microsoft/resnet-50 --device npu --monitor --iterations 1000
```

**Options**:
- `-m/--model` — HuggingFace model ID or local `.onnx` file
- `--task` — explicit task (auto-detected if not specified)
- `--iterations` — benchmark iterations (default: 100)
- `--warmup` — warmup iterations excluded from stats (default: 10)
- `--device` — target device; always specify `npu` in this bug bash
- `--precision` — `auto`, `int8`, `int16`, or `w{x}a{y}` (default: auto)
- `--ep` — force specific EP (use `qnn` for NPU)
- `-o/--output` — output JSON file path
- `--batch-size` — input batch size (default: 1)
- `--shape-config` — JSON file with shape overrides
- `--no-quantize` — skip quantization during auto build
- `--rebuild` — force rebuild of cached artifacts
- `--ignore-cache` — build in temp folder, discard after run
- `--monitor` — live NPU utilization chart during benchmark
- `--op-tracing [basic|detail]` — operator-level profiling (requires `onnxruntime-qnn`; see known issue #217 — may crash or produce empty trace)
- `-v/--verbose`

**Pass criteria**:
- All variants report P50/P90/Avg latency and throughput on NPU
- `--monitor` shows live NPU utilization chart during run

---

### 💡 Using pipeline to build models

### System Info (`winml sys`)

```bash
winml sys
winml sys --list-device --list-ep
```

**Options**:
- `-f/--format [text|json|compact]` — output format (default: text)
- `-v/--verbose` — additional diagnostic information
- `--list-device` — list available devices in priority order
- `--list-ep` — list available execution providers

**Pass criteria**: Your NPU device and its EP (QNN / OpenVINO / VitisAI) are listed.

---

### Hub (`winml hub`)

```bash
winml hub
winml hub --model-type bert
winml hub --task image-classification
winml hub --model microsoft/resnet-50
winml hub --output catalog.json
```

**Options**:
- `-t/--model-type` — filter by architecture (e.g., `bert`, `vit`)
- `-k/--task` — filter by task (e.g., `text-classification`)
- `-m/--model` — show detail for a specific model
- `-o/--output` — save results to JSON file

**Pass criteria**: Catalog table displayed; per-model detail includes accuracy verdict.

---

### Inspect (`winml inspect`)

> Run this before testing any model. If inspect fails, skip the model entirely.

```bash
winml inspect -m microsoft/resnet-50
winml inspect -m microsoft/resnet-50 --verbose
winml inspect -m microsoft/resnet-50 --hierarchy
```

**Options**:
- `-m/--model` — HuggingFace model ID or local ONNX file path (required)
- `-f/--format [table|json]` — output format (default: table)
- `-v/--verbose` — show full configuration details
- `-t/--task` — override auto-detected task
- `-H/--hierarchy` — show HF module hierarchy (random weights, no download)

**Pass criteria**: Model task, loader/exporter class, inference class, and support status all printed without error.

---

### Config + Build (`winml config` + `winml build`)

```bash
# Generate NPU config and build
winml config -m microsoft/resnet-50 --device npu --precision int8 -o resnet_config/config.json
winml build -c resnet_config/config.json -m microsoft/resnet-50 -o resnet_config/

# Text model on NPU
winml config -m dslim/bert-base-NER --device npu --precision int8 -o bert_config/config.json
winml build -c bert_config/config.json -m dslim/bert-base-NER -o bert_config/

# Use global cache
winml build -c resnet_config/config.json -m microsoft/resnet-50 --use-cache
```

**`winml config` options**:
- `-m/--model` — HuggingFace model ID or `.onnx` file
- `-t/--task` — override auto-detected task
- `-d/--device npu` — always use `npu` in this bug bash
- `--ep [qnn|openvino|vitisai]` — EP matching your device
- `-p/--precision` — `int8`, `int16`, or `w{x}a{y}` recommended for NPU
- `-o/--output` — output JSON file (default: stdout)
- `--no-quant` — exclude quantization from config
- `--no-compile` — exclude compilation from config
- `--shape-config` — JSON with shape overrides

**`winml build` options**:
- `-c/--config` — WinMLBuildConfig JSON (required)
- `-m/--model` — HuggingFace model ID or `.onnx` file (required)
- `-o/--output-dir` — output directory
- `--use-cache` — use global cache `~/.cache/winml/`
- `--rebuild` — force rebuild
- `--no-quant`, `--no-compile`, `--no-optimize` — skip stages
- `--no-analyze` — skip analyzer loop
- `--max-optim-iterations` — max autoconf re-optimization rounds (default: 3)

**Pass criteria**: Config JSON generated; build completes all stages (export → optimize → quantize → compile) targeting NPU.

---

### Perf (`winml perf`)

```bash
# Direct ONNX benchmark on NPU
winml perf -m resnet_compiled/resnet_quant_int8_qnn_ctx.onnx --device npu --iterations 100

# HuggingFace end-to-end (auto-pipeline) on NPU
winml perf -m microsoft/resnet-50 --device npu --iterations 100

# Live NPU utilization monitor
winml perf -m microsoft/resnet-50 --device npu --monitor --iterations 1000
```

**Options**:
- `-m/--model` — HuggingFace model ID or local `.onnx` file
- `--task` — explicit task (auto-detected if not specified)
- `--iterations` — benchmark iterations (default: 100)
- `--warmup` — warmup iterations excluded from stats (default: 10)
- `--device` — target device; always specify `npu` in this bug bash
- `--precision` — `auto`, `int8`, `int16`, or `w{x}a{y}` (default: auto)
- `--ep` — force specific EP (use `qnn` for NPU)
- `-o/--output` — output JSON file path
- `--batch-size` — input batch size (default: 1)
- `--shape-config` — JSON file with shape overrides
- `--no-quantize` — skip quantization during auto build
- `--rebuild` — force rebuild of cached artifacts
- `--ignore-cache` — build in temp folder, discard after run
- `--monitor` — live NPU utilization chart during benchmark
- `--op-tracing [basic|detail]` — operator-level profiling (requires `onnxruntime-qnn`; see known issue #217 — may crash or produce empty trace)
- `-v/--verbose`

**Pass criteria**:
- All variants report P50/P90/Avg latency and throughput on NPU
- `--monitor` shows live NPU utilization chart during run

---

### Eval (`winml eval`)

```bash
winml eval -m microsoft/resnet-50 --device npu --samples 100
winml eval -m dslim/bert-base-NER --device npu --samples 100
winml eval --task image-classification --schema
```

**Options**:
- `-m/--model` — HuggingFace model ID or `.onnx` file
- `--model-id` — HuggingFace model ID when `-m` points to an `.onnx` file
- `--dataset` — HF dataset path (auto-selected per task if omitted)
- `--task` — override auto-detected task
- `--device npu` — always use `npu` in this bug bash
- `--samples` — number of samples (default: 100)
- `--split` — dataset split (default: validation)
- `--shuffle/--no-shuffle`
- `--streaming` — stream dataset instead of downloading fully
- `-o/--output` — output JSON file path
- `--schema` — print expected dataset schema for the given task and exit

**Models with built-in default dataset** (no `--dataset` needed):
- image-classification: `microsoft/resnet-50`, `facebook/convnext-tiny-224`
- text-classification: `Intel/bert-base-uncased-mrpc` *(but see known issue #216 — avoid for eval)*
- token-classification: `dslim/bert-base-NER`, `dbmdz/bert-large-cased-finetuned-conll03-english`, `Babelscape/wikineural-multilingual-ner`

Other models require `--dataset <hf-dataset-path>`. Skip this command if the model has no default dataset.

**Pass criteria**: Accuracy metric reported without error on NPU.

---

### 💡 Using primitive command to build model step by step

### Export (`winml export`)

```bash
winml export -m microsoft/resnet-50 -o resnet_onnx/model.onnx
```

**Options**:
- `-m/--model` — HuggingFace model ID or local path (required)
- `-o/--output` — output ONNX file path (required)
- `-v/--verbose` — verbose 8-step output (crashes on Windows cp1252 terminals; see known issue #214)
- `--with-report` — generate markdown + JSON reports alongside ONNX
- `--clean-onnx` — produce a clean ONNX without embedded metadata
- `--no-hierarchy` — skip embedding hierarchy metadata
- `--dynamo` — use PyTorch 2.0+ dynamo export
- `--torch-module` — include specific `torch.nn` modules in hierarchy (comma-separated)
- `-t/--task` — override auto-detected task
- `--input-specs` — JSON file with custom input specifications
- `--export-config` — ONNX export configuration JSON
- `--shape-config` — JSON with shape overrides

**Pass criteria**: ONNX file produced at the specified path; no errors.

---

### Analyze (`winml analyze`)

Use the EP matching your device:

```bash
# Snapdragon X Elite (QNN)
winml analyze --model resnet_onnx/model.onnx --ep qnn --device NPU --information
winml analyze --model resnet_onnx/model.onnx --ep qnn --device NPU --run-unknown-op

# Intel AI Boost (OpenVINO)
winml analyze --model resnet_onnx/model.onnx --ep openvino --device NPU --information

# AMD Ryzen AI (VitisAI)
winml analyze --model resnet_onnx/model.onnx --ep vitisai --device NPU --information

# Save output to file
winml analyze --model resnet_onnx/model.onnx --ep qnn --device NPU --output results.json
```

**Options**:
- `--model` — path to ONNX model (required)
- `--ep` — target EP: use `qnn` for NPU
- `--device [CPU|GPU|NPU]` — target device; always use `NPU` (uppercase) in this bug bash
- `-v/--verbose` / `-q/--quiet`
- `--output` — save JSON output to file
- `--information/--no-information` — include detailed recommendations (default: enabled)
- `--run-unknown-op/--no-run-unknown-op` — run unknown ops on local machine (default: enabled)
- `--save-node [partial|unsupported]` — save specific node types for further analysis
- `--htp-metadata` — path to HTP metadata JSON for enhanced pattern extraction

**Pass criteria**: EP compatibility report shown for your NPU; `--run-unknown-op` runs without crash.

> **Known issue #194**: Exit code is 1 when any EP has unknown operators, even if QNN NPU reports 100% support. Check the report content, not just the exit code.

---

### Optimize (`winml optimize`)

```bash
winml optimize --list-capabilities
winml optimize --list-rewrites
winml optimize -m resnet_onnx/model.onnx -o resnet_optimized.onnx
```

**Options**:
- `-l/--list-capabilities` — list all registered capabilities and exit
- `--list-rewrites` — list available pattern rewrite families and exit
- `-m/--model` — input ONNX model file
- `-o/--output` — output path (default: `{input}_opt.onnx`)
- `-c/--config` — YAML/JSON config file
- `-v/--verbose`
- `--enable-*/--disable-*` — toggle individual capabilities (see `--list-capabilities`)

**Pass criteria**: Optimized ONNX produced; file size equal or smaller than input.

---

### Quantize (`winml quantize`)

```bash
winml quantize -m resnet_optimized.onnx --precision int8 -o resnet_quant_int8.onnx
winml quantize -m resnet_optimized.onnx --weight-type int8 --activation-type uint16 -o resnet_quant_w8a16.onnx
```

**Options**:
- `-m/--model` — input ONNX model file (required)
- `-o/--output` — output path (default: `{input}_qdq.onnx`)
- `-p/--precision [int8|int16|w{x}a{y}]` — precision shorthand
- `--weight-type [uint8|int8|uint16|int16]` — explicit weight type (overrides `--precision`)
- `--activation-type [uint8|int8|uint16|int16]` — explicit activation type (overrides `--precision`)
- `--samples` — calibration samples (default: 10)
- `--method [minmax|entropy|percentile]` — calibration method (default: minmax)
- `--per-channel` — per-channel quantization
- `--symmetric` — symmetric quantization
- `-v/--verbose`

**Pass criteria**: Each command completes; output ONNX contains QDQ nodes.

> **Known issue #185 (P1)**: Re-running with the same `-o` path crashes with `FileExistsError` if a `.onnx.data` sidecar already exists. Delete the old output file and its `.data` sidecar before re-running.

> **Known issue #193 (P1)**: Output from standalone `winml quantize` may fail NPU compilation (QNN MaxPool NHWC layout error). Use `winml build` for the full NPU pipeline.

---

### Compile (`winml compile`)

```bash
# List available compilers
winml compile --list

# Snapdragon X Elite (QNN)
winml compile -m resnet_quant_int8.onnx --output-dir resnet_compiled/ --ep qnn --device npu
winml compile -m resnet_quant_int8.onnx --output-dir resnet_compiled/ --ep qnn --device npu --no-quantize

# Intel AI Boost (OpenVINO)
winml compile -m resnet_quant_int8.onnx --output-dir resnet_compiled/ --ep openvino --device npu

# AMD Ryzen AI (VitisAI)
winml compile -m resnet_quant_int8.onnx --output-dir resnet_compiled/ --ep vitisai --device npu
```

**Options**:
- `-m/--model` — input ONNX model (required unless `--list`)
- `--output-dir` — output directory (default: same as input)
- `-d/--device` — target device; always `npu` in this bug bash (default: npu)
- `--ep` — use `qnn` for NPU
- `--quantize/--no-quantize` — enable/disable internal quantization (default: enabled)
- `--validate/--no-validate` — validate compiled model (default: enabled)
- `-v/--verbose`
- `--compiler [ort|qairt]` — compiler backend (default: ort)
- `--qnn-sdk-root` — path to QAIRT SDK root
- `--embed` — embed EP context in ONNX (default: external `.bin` file)
- `--list` — list available compilers for selected device and exit

**Pass criteria**: Compiled ONNX produced in output directory.

> **Known issue #186 (P1)**: `--ep qnn` silently falls back to OpenVINO when QNN SDK is not available. Verify your EP is correctly listed in `winml sys --list-ep` before compiling.

---

### Perf (`winml perf`)

```bash
# Direct ONNX benchmark on NPU
winml perf -m resnet_compiled/resnet_quant_int8_qnn_ctx.onnx --device npu --iterations 100

# HuggingFace end-to-end (auto-pipeline) on NPU
winml perf -m microsoft/resnet-50 --device npu --iterations 100

# Live NPU utilization monitor
winml perf -m microsoft/resnet-50 --device npu --monitor --iterations 1000
```

**Options**:
- `-m/--model` — HuggingFace model ID or local `.onnx` file
- `--task` — explicit task (auto-detected if not specified)
- `--iterations` — benchmark iterations (default: 100)
- `--warmup` — warmup iterations excluded from stats (default: 10)
- `--device` — target device; always specify `npu` in this bug bash
- `--precision` — `auto`, `int8`, `int16`, or `w{x}a{y}` (default: auto)
- `--ep` — force specific EP (use `qnn` for NPU)
- `-o/--output` — output JSON file path
- `--batch-size` — input batch size (default: 1)
- `--shape-config` — JSON file with shape overrides
- `--no-quantize` — skip quantization during auto build
- `--rebuild` — force rebuild of cached artifacts
- `--ignore-cache` — build in temp folder, discard after run
- `--monitor` — live NPU utilization chart during benchmark
- `--op-tracing [basic|detail]` — operator-level profiling (requires `onnxruntime-qnn`; see known issue #217 — may crash or produce empty trace)
- `-v/--verbose`

**Pass criteria**:
- All variants report P50/P90/Avg latency and throughput on NPU
- `--monitor` shows live NPU utilization chart during run

---

### Eval (`winml eval`)

```bash
winml eval -m microsoft/resnet-50 --device npu --samples 100
winml eval -m dslim/bert-base-NER --device npu --samples 100
winml eval --task image-classification --schema
```

**Options**:
- `-m/--model` — HuggingFace model ID or `.onnx` file
- `--model-id` — HuggingFace model ID when `-m` points to an `.onnx` file
- `--dataset` — HF dataset path (auto-selected per task if omitted)
- `--task` — override auto-detected task
- `--device npu` — always use `npu` in this bug bash
- `--samples` — number of samples (default: 100)
- `--split` — dataset split (default: validation)
- `--shuffle/--no-shuffle`
- `--streaming` — stream dataset instead of downloading fully
- `-o/--output` — output JSON file path
- `--schema` — print expected dataset schema for the given task and exit

**Models with built-in default dataset** (no `--dataset` needed):
- image-classification: `microsoft/resnet-50`, `facebook/convnext-tiny-224`
- text-classification: `Intel/bert-base-uncased-mrpc` *(but see known issue #216 — avoid for eval)*
- token-classification: `dslim/bert-base-NER`, `dbmdz/bert-large-cased-finetuned-conll03-english`, `Babelscape/wikineural-multilingual-ner`

Other models require `--dataset <hf-dataset-path>`. Skip this command if the model has no default dataset.

**Pass criteria**: Accuracy metric reported without error on NPU.

---

## Known Issues (found during bug bash)

| # | Issue | Severity | Area | Description | Workaround |
|---|-------|----------|------|-------------|------------|
| 1 | [#192](https://github.com/microsoft/ModelKit/issues/192) | P1 | `winml perf --module` | `AttributeError: ResNetModel has no attribute 'resnet'` — module path construction bug in `perf.py` when using `--module` on ResNet | Avoid `--module` on ResNet; try BERT (`BertAttention`) instead |
| 2 | [#193](https://github.com/microsoft/ModelKit/issues/193) | P1 | `winml quantize` → NPU | ONNX produced by standalone `winml quantize` fails NPU compilation (QNN MaxPool NHWC layout error); `winml build` pipeline is unaffected | Use `winml build` for NPU targets |
| 3 | [#185](https://github.com/microsoft/ModelKit/issues/185) | P1 | `winml quantize` re-run | Crashes with `FileExistsError` when output `.onnx.data` sidecar from a previous run already exists | Delete the old `.onnx` and `.onnx.data` files before re-running |
| 4 | [#186](https://github.com/microsoft/ModelKit/issues/186) | P1 | `winml compile --ep qnn` | Silently falls back to OpenVINO when QNN SDK is not installed; output file is still named `*_qnn_ctx.onnx`; downstream `winml perf` crashes | Verify QNN SDK is present via `winml sys` before compiling |
| 5 | [#194](https://github.com/microsoft/ModelKit/issues/194) | P2 | `winml analyze` exit code | Exits with code 1 when any EP has unknown operators, even when QNN NPU reports 100% support — may break CI pipelines | Check report content, not just exit code |
| 6 | [#195](https://github.com/microsoft/ModelKit/issues/195) | P2 | `winml perf --module` | `--module` expects a **class name** (e.g., `BertAttention`), not a module path; unclear error when wrong format is used | Run `winml inspect -m <model> --hierarchy` to discover valid class names |
| 7 | [#175](https://github.com/microsoft/ModelKit/issues/175) | P2 | `winml perf` vs `winml build` | `winml perf -m <hf-id>` and `winml config` + `winml build` can produce different export results for the same model | Use `winml build` output for production; file a repro if you observe discrepancies |
| 8 | [#182](https://github.com/microsoft/ModelKit/issues/182) | P2 | `winml analyze --run-unknown-op` | Static analyzer still creates single-node models for unknown QDQ ops even when `--run-unknown-op` is enabled | No workaround; may report incorrect support status for those ops |
| 9 | [#214](https://github.com/microsoft/ModelKit/issues/214) | P1 | `winml export --verbose` | `UnicodeEncodeError: 'charmap' codec can't encode character` — emoji in verbose output crashes on Windows (cp1252 encoding) | Omit `--verbose` flag; #208 fix incomplete for export command |
| 10 | [#215](https://github.com/microsoft/ModelKit/issues/215) | P2 | `winml analyze --device` | `--device` only accepts uppercase (`NPU`/`GPU`/`CPU`); all other commands accept lowercase — causes confusing errors | Use uppercase: `--device NPU` |
| 11 | [#216](https://github.com/microsoft/ModelKit/issues/216) | P1 | `winml eval` text-classification | `RuntimeError: Label alignment failed` — auto-selected dataset labels don't match model's label set (e.g. finbert positive/negative/neutral vs GLUE) | Avoid text-classification models for eval; image-classification and token-classification are unaffected |
| 12 | [#217](https://github.com/microsoft/ModelKit/issues/217) | P2 | `winml perf --op-tracing` | `--op-tracing` may crash or produce an empty trace; requires `onnxruntime-qnn` to be installed separately | Omit `--op-tracing` unless specifically testing this feature |

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `winml sys` | System info + device/EP inventory |
| `winml inspect -m <model>` | Verify model is supported (**run this first**) |
| `winml hub` | Browse built-in validated model catalog |
| `winml export -m <model> -o dir/` | Export to ONNX |
| `winml analyze --model <model.onnx> --ep qnn --device NPU` | Analyze QNN NPU compatibility |
| `winml optimize -m <model.onnx> -o out.onnx` | Apply graph optimizations |
| `winml quantize -m <model.onnx> --precision int8` | Insert QDQ quantization nodes |
| `winml compile -m <model.onnx> --ep qnn --device npu` | Compile for NPU |
| `winml perf -m <model> --device npu` | Benchmark on NPU |
| `winml perf -m <model> --device npu --monitor` | Benchmark with live NPU chart |
| `winml config -m <model> --device npu -o config.json` | Generate NPU build config |
| `winml build -c config.json -m <model>` | Build all stages for NPU |
| `winml eval -m <model> --device npu` | Evaluate accuracy on NPU |
| `winml --help` / `winml <cmd> --help` | Command reference |
