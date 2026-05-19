# WinML CLI

[![ModelKit CI](https://github.com/microsoft/winml-cli/actions/workflows/modelkit-ci.yml/badge.svg)](https://github.com/microsoft/winml-cli/actions/workflows/modelkit-ci.yml)
![Status](https://img.shields.io/badge/status-early%20access-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

**WinML CLI** is a CLI toolkit to build **portable, performant, and high-quality** models for Windows ML. It covers the entire journey from pretrained model to on-device inference — export, optimization, quantization, compilation, and benchmarking — across **all execution providers**, regardless of silicon.

---

## :dart: WinML CLI Is Right for You If

- [x] You want to build models that run on **any Windows device** — Qualcomm, Intel, AMD, NVIDIA, or CPU
- [x] You want to benchmark a model with **one command** — latency, throughput, and live hardware utilization
- [x] You want to catch compatibility issues **ahead of time** — unsupported ops, shape mismatches, EP gaps
- [x] You want **deep insights** into your model — I/O shapes, task mapping, operator coverage per EP
- [x] You want a **repeatable and traceable** model building process — config-driven, inspectable at every stage
- [x] You want **AI agents** to build and profile models for you — agent-ready skills for coding assistants

---

## :desktop_computer: Supported Hardware

| Execution Provider | Hardware | Status | EP Flag | Device Flag |
|:-------------------|:---------|:------:|:--------|:------------|
| **QNN** | Qualcomm NPU (Snapdragon X Elite) | 🟢 Ready | `--ep qnn` | `--device npu` |
| **OpenVINO** | Intel NPU (Meteor Lake / Lunar Lake) | 🟢 Ready | `--ep openvino` | `--device npu` |
| **VitisAI** | AMD NPU (Ryzen AI) | 🟢 Ready | `--ep vitisai` | `--device npu` |
| **NvTensorRTRTX** | NVIDIA discrete GPUs | 🔶 Planned | `--ep nv_tensorrt_rtx` | `--device gpu` |
| **MIGraphX** | AMD discrete GPUs | 🔶 Planned | `--ep migraphx` | `--device gpu` |
| **Dml** | Hardware-agnostic GPU backend | 🔶 Planned | `--ep dml` | `--device gpu` |
| **CPU** | Cross-platform fallback | ⚪ Always available | `--ep cpu` | `--device cpu` |

> **Tip:** Use `--device auto` and WinML CLI picks the best available device — NPU first, then GPU, then CPU.

---

## :clipboard: Prerequisites

### Required Software

| **Component** | **How to Get It** |
|-----------|--------------|
| **Windows 11** (x64 or ARM64) | Windows 11 24H2+ required for NPU support |
| **UV** | Install [UV](https://github.com/astral-sh/uv) |
| **Windows App SDK Runtime 1.8** | [Latest Windows App SDK downloads](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads) |
| **WinML CLI** (Python wheel) | [Releases](https://github.com/microsoft/winml-cli/releases) |

### Required Hardware

**WinML CLI targets NPU.** We recommend testing on one of the following NPU devices:

| Device | EP | Flag |
|--------|-----|------|
| Snapdragon X Elite (Qualcomm) | QNN | `--ep qnn --device npu` |
| Intel AI Boost (Meteor Lake / Lunar Lake) | OpenVINO | `--ep openvino --device npu` |
| AMD Ryzen AI (Phoenix / Hawk Point / Strix) | VitisAI | `--ep vitisai --device npu` |

**No NPU?** Use `--device auto` — WinML CLI will fall back to the best available device (GPU → CPU). Note that `winml compile` requires NPU and cannot run without one.

### Accepted Inputs

- **HuggingFace model ID** (e.g., `microsoft/resnet-50`) — weights are downloaded on first run
- **Local ONNX file** (e.g., `model.onnx`) — from `winml export`, `winml build`, or any ONNX you already have

### The Golden Rule: Inspect First

Before running any pipeline command, always verify the model is supported:

```bash
winml inspect -m <model-id>
```

If `inspect` prints an error or shows `Unsupported`, **skip that model**. Only models that pass inspect are valid inputs for export, analyze, build, perf, and eval.

---

## :package: Installation

WinML CLI requires **Python 3.11** and is distributed as a Python wheel. We recommend [uv](https://docs.astral.sh/uv/) for fast, reproducible environment setup.

**1. Create a Python 3.11 environment**

```bash
uv venv --python 3.11
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\activate

# Windows (Git Bash / WSL)
source .venv/Scripts/activate
```

**2. Install from wheel**

```bash
uv pip install winml_cli-<version>-py3-none-any.whl
```

**3. Verify your environment**

```bash
winml sys --list-device --list-ep
```

Confirm that your target device and EP appear in the output:

- **Snapdragon X Elite** — look for `QNNExecutionProvider`
- **Intel AI Boost** — look for `OpenVINOExecutionProvider`
- **AMD Ryzen AI** — look for `VitisAIExecutionProvider`

If no NPU is detected, you can still use WinML CLI with `--device auto` for most commands. The only exception is `winml compile`, which requires an NPU device.

---

## :wrench: Commands

| Category | Commands | Purpose |
|:---------|:---------|:--------|
| **Primitives** | `inspect` `export` `optimize` `quantize` `compile` | Single-stage building blocks |
| **Pipeline** | `config` `build` `perf` `eval` `run`\* | End-to-end orchestration |
| **Insights** | `analyze` `debug`\* | Diagnostics and compatibility |
| **Utilities** | `hub` `cache`\* `doctor`\* `setting`\* `sys` | Catalog, cache, and environment |

\* = coming soon

<details>
<summary><strong>Primitives</strong> — one stage at a time</summary>

**`winml inspect`** — Discover model metadata. Prints the task, model class, input/output tensor names and shapes, and execution provider compatibility. No weights are loaded — this reads only the model configuration, making it fast and lightweight. Always run inspect first to verify a model is supported.

**`winml export`** — Convert a source model to ONNX. Takes a Hugging Face model ID (or local checkpoint) and produces a standards-compliant ONNX file with hierarchy-preserving metadata.

**`winml optimize`** — Fuse operators, simplify graphs, and prepare for target EPs. Takes an ONNX model and an optimization config (typically generated by `winml analyze`) and applies graph-level transformations: operator fusion, constant folding, shape inference, and EP-specific rewrites.

**`winml quantize`** — Compress to low-bit precision. Reduces model size and inference latency by converting weights and activations from FP32 to INT8 (or other low-bit formats). After quantization, the model is portable — it can run on any ONNX Runtime backend.

**`winml compile`** — Generate device-specific binaries. Takes a quantized ONNX model and produces EP-specific compiled artifacts (for example, QNN context binaries for Qualcomm NPU). This step locks the model to a specific device but delivers the lowest possible inference latency.

</details>

<details>
<summary><strong>Pipeline</strong> — orchestrated workflows</summary>

**`winml config`** — Auto-detect optimal settings into a JSON config. Inspects the model and generates a complete build specification: task, I/O shapes, optimization flags, quantization parameters, and target EP settings. The config file is reviewable, editable, and version-controllable — the single source of truth for your build.

**`winml build`** — Orchestrate the full pipeline. Takes a config file and executes every stage in sequence: export, analyze, optimize, quantize, and compile. Two commands (`config` + `build`) replace eight manual steps.

**`winml perf`** — Benchmark latency, throughput, and hardware utilization. Runs inference on the target device and reports latency percentiles (p50, p90, p99), throughput (inferences per second), and optionally live hardware monitoring (CPU, RAM, NPU utilization) with the `--monitor` flag. Can accept a local ONNX file or a Hugging Face model ID.

**`winml eval`** — Measure model accuracy against reference datasets. Compares the output of your optimized/quantized model against the original to quantify any accuracy loss introduced by the pipeline.

**`winml run`** — End-to-end inference with pre/post processing. *(Coming soon.)*

</details>

<details>
<summary><strong>Insights</strong> — understand what is happening inside</summary>

**`winml analyze`** — Lint operators, check EP compatibility, and generate optimization config. The analyzer has two components: the **Linter** (like ESLint for ONNX) checks every operator against target EPs and classifies each as supported, partial, or unsupported. **AutoConf** detects suboptimal patterns and generates the optimization config that the optimizer consumes. Together they form the analyze-optimize loop.

**`winml debug`** — Interactive model debugging and layer-by-layer inspection. *(Coming soon.)*

</details>

<details>
<summary><strong>Utilities</strong> — catalog, cache, and environment</summary>

**`winml catalog`** — Browse the curated built-in model catalog.

**`winml cache`** — Manage built model artifacts and pipeline outputs. View, clean, or selectively remove cached models and intermediate files.

**`winml doctor`** — Diagnose environment issues. Checks runtimes, execution providers, and dependencies to identify configuration problems.

**`winml setting`** — Configure WinML CLI preferences. Set default EPs, output directories, and other global options.

**`winml sys`** — System information and capability reporting. Prints detected hardware, available EPs, Python version, and installed package versions.

</details>

---

## :rocket: Quick Start

### Inspect a Model

The fastest way to get started is to inspect a model. Let's look at ResNet-50:

```bash
winml inspect -m microsoft/resnet-50
```

This prints the model's metadata without downloading weights:

- **Task**: `image-classification` — what the model does
- **Model class**: `ResNetForImageClassification` — the architecture
- **Input tensors**: names, data types, and shapes (e.g., `pixel_values: float32 [1, 3, 224, 224]`)
- **Output tensors**: names, data types, and shapes (e.g., `logits: float32 [1, 1000]`)

If inspect succeeds, the model is supported and you can proceed with the rest of the pipeline.

> **Golden rule: always inspect first.** Before running export, build, perf, or any other pipeline command, verify the model is supported with `winml inspect`.

### Build with Primitive Commands

This walkthrough builds **ConvNeXT** (`facebook/convnext-base-224`) step by step using primitive commands. ConvNeXT is a family of CNN models inspired by Vision Transformers, introduced by Meta in 2022 — it offers high accuracy while retaining the efficiency of CNNs.

#### Phase 1: Inspect

```bash
winml inspect -m facebook/convnext-base-224
```

#### Phase 2: Build a Portable Model

**Export** from PyTorch to ONNX:

```bash
winml export -m facebook/convnext-base-224 -o convnext/model.onnx -v
```

**Analyze** for EP compatibility:

```bash
winml analyze -m convnext/model.onnx --optim-config optim.json
```

**Optimize** the graph using the analyzer's config:

```bash
winml optimize -m convnext/model.onnx -c optim.json -o convnext/model_opt.onnx
```

**Quantize** to INT8:

```bash
winml quantize -m convnext/model_opt.onnx -o convnext/model_opt_int8.onnx
```

#### Phase 3: Benchmark on Device

**Compile** for NPU (generates device-specific binaries):

```bash
winml compile -m convnext/model_opt_int8.onnx --ep qnn -o convnext/model_compiled.onnx
```

**Benchmark on NPU** — note the latency:

```bash
winml perf -m convnext/model_compiled.onnx --ep qnn --iterations 100
```

**Benchmark on CPU** for comparison:

```bash
winml perf -m convnext/model_opt.onnx --ep cpu --iterations 100
```

Compare the two numbers to see the performance difference between NPU and CPU inference.

### Build with Config + Build

Same model, different approach. Instead of running each command manually, use the config-driven pipeline. Think of it like CMake: `config` generates a build plan, `build` executes it.

**Generate the build config:**

```bash
winml config -m facebook/convnext-base-224 -o convnext_config.json
```

This creates a JSON file containing all settings for every pipeline step — task, I/O shapes, optimization flags, quantization parameters — all auto-detected from the model.

**Build the model:**

```bash
winml build -c convnext_config.json -m facebook/convnext-base-224 -o convnext_build/
```

This orchestrates the full pipeline — export, analyze, optimize, quantize, compile — all in one go. Same result as the manual steps above, but in two commands.

**Benchmark the result:**

```bash
winml perf -m convnext_build/model.onnx --ep qnn --iterations 100
```

The config file is the single source of truth for your build. Version-control it, share it with teammates, edit it to override settings, and replay builds deterministically on any machine.

### Benchmark in One Command

The simplest way to evaluate a model — one command, zero setup:

```bash
winml perf -m facebook/convnext-base-224 --device npu --monitor
```

WinML CLI handles everything behind the scenes: download the model from Hugging Face, export to ONNX, optimize the graph, and run the benchmark on your NPU. The `--monitor` flag enables live hardware monitoring — real-time CPU utilization, RAM usage, and NPU activity alongside the latency results.

This is ideal for quick smoke tests: does the model run on this device, and how fast is it?

---

## :arrows_counterclockwise: The BYOM Workflow

The **Build Your Own Model** (BYOM) workflow is the philosophy behind WinML CLI. It defines how a source model becomes a production-ready, device-optimized artifact.

### The Pipeline

```
Source Model --> Export --> Analyze --> Optimize --> Quantize --> Compile --> Benchmark
```

![BYOM Workflow](docs/assets/workflow-only.svg)

Each arrow is a WinML CLI command. You can enter the pipeline at any stage (for example, start with a local ONNX file and skip export), exit early (stop after optimization if you do not need quantization), or loop back to repeat a stage with different settings.

### Primitive Commands vs. Config-Driven Pipeline

|  | **Primitive Commands** | **Config-Driven Pipeline** |
|:--|:--|:--|
| **Steps** | One command **per stage** | Two steps: **config** + **build** |
| **Control** | Start from any stage; try different settings to fix errors or tweak performance | Repeatable, tweakable, version-controllable |
| **Best for** | **Flexible** workflow | Production-ready **delivery** |
| **When to use** | Exploring, debugging, prototyping | CI/CD, batch builds, team workflows |
| **Lifecycle** | "Coding" phase | Polish |

---

## :clipboard: Built-in Models

Run `winml catalog` to browse the full catalog interactively.

<details>
<summary><strong>Click to expand the full model catalog</strong></summary>

| Model ID | Task | Architecture |
|:---------|:-----|:-------------|
| `microsoft/resnet-50` | image-classification | ResNet |
| `google/vit-base-patch16-224` | image-classification | ViT |
| `microsoft/swin-large-patch4-window7-224` | image-classification | Swin |
| `facebook/convnext-tiny-224` | image-classification | ConvNeXT |
| `rizvandwiki/gender-classification` | image-classification | ViT |
| `ProsusAI/finbert` | text-classification | BERT |
| `Intel/bert-base-uncased-mrpc` | text-classification | BERT |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | text-classification | RoBERTa |
| `dslim/bert-base-NER` | token-classification | BERT |
| `dbmdz/bert-large-cased-finetuned-conll03-english` | token-classification | BERT |
| `Babelscape/wikineural-multilingual-ner` | token-classification | BERT |
| `w11wo/indonesian-roberta-base-posp-tagger` | token-classification | RoBERTa |
| `microsoft/table-transformer-detection` | object-detection | Table Transformer |
| `mattmdjaga/segformer_b2_clothes` | image-segmentation | SegFormer |
| `nvidia/segformer-b1-finetuned-ade-512-512` | image-segmentation | SegFormer |
| `nvidia/segformer-b2-finetuned-ade-512-512` | image-segmentation | SegFormer |
| `nvidia/segformer-b5-finetuned-ade-640-640` | image-segmentation | SegFormer |

</details>

These models are verified against WinML CLI's full pipeline and serve as reliable starting points. You are not limited to this list — any Hugging Face model that passes `winml inspect` is a valid input.

For models not in this table, run `winml inspect -m <model-id>` to verify support before proceeding.

---

## :warning: Scope & Limitations

### What WinML CLI supports

WinML CLI targets **classic deep learning models** — CNNs, encoders, vision transformers, NLP classifiers, token classifiers, object detection models, and segmentation models.

Supported tasks include:
- Image classification (ResNet, ViT, Swin, ConvNeXT)
- Text classification (BERT, RoBERTa)
- Token classification / NER (BERT, RoBERTa)
- Object detection (Table Transformer)
- Image segmentation (SegFormer)

### What WinML CLI does not support

**LLMs and generative models are not in scope.** Do not use WinML CLI with GPT, LLaMA, Phi, Mistral, Stable Diffusion, or any model with a decoder-only or sequence-to-sequence generative architecture. LLM support (with LoRA) is planned for Q3-Q4 2026.

### Known constraints

- `winml compile` requires an NPU device. If no NPU is available, skip the compile step and use `--device auto` for benchmarking.
- Some models may export successfully but fail during optimization or quantization due to unsupported operator patterns. The analyzer will flag these issues.
- Performance numbers vary by device, driver version, and EP version. Always benchmark on your target hardware.

---

## :world_map: Roadmap

| Milestone | Target | Highlights |
|:----------|:-------|:-----------|
| 🟡 **Kickoff** | Q4 2025 | Internal prototype, core primitive commands |
| 🟢 **Early Access** | Q1 2026 | First external testers, config + build pipeline, hub catalog |
| 🔵 **Public Beta** | Q2 2026 | Open source, agent skills, Foundry Toolkit integration |
| 🟣 **RC** | Q3-Q4 2026 | **LLM support** (with LoRA), broader device coverage, MLIR |

<details>
<summary><strong>Click to expand roadmap details</strong></summary>

**Q4 2025 — Kickoff**
- Primitive commands: `inspect`, `export`, `optimize`, `quantize`, `compile`
- QNN, OpenVINO, and VitisAI execution provider support
- Internal validation with ResNet, BERT, ViT, SegFormer families

**Q1 2026 — Early Access**
- Pipeline commands: `config`, `build`, `perf`, `eval`
- Analyzer with auto-configuration loop
- Built-in model catalog (`winml catalog`)
- Live hardware monitoring (`--monitor`)

**Q2 2026 — Public Beta**
- Open source release
- Agent-ready skills for coding assistants (Claude Code, Cursor, Copilot)
- Foundry Toolkit for VS Code integration

**Q3-Q4 2026 — Release Candidate**
- LLM support (decoder-only architectures with LoRA adapters)
- NvTensorRTRTX, MIGraphX, and Dml execution providers
- MLIR-based optimization backend
- Public SDK and framework APIs

</details>

---

## :lock: Data / Telemetry

Official WinML CLI releases can collect anonymous usage telemetry to
help improve the product. Telemetry is classified as **Optional**. A
one-time prompt on your first run asks for consent (default: accept —
press Enter to enable, type `n` to decline).

Dev installs (`pip install -e .` or running from a source checkout)
never send telemetry.

**Control** — edit `%USERPROFILE%\.winml\config.json`:

- Set `telemetry.consent` to `"disabled"` to opt out
- Set `telemetry.consent` to `"enabled"` to opt in
- Delete the file to re-show the first-run prompt on the next run

Telemetry is automatically disabled in CI / non-TTY environments
regardless of the stored decision.

See [docs/Privacy.md](docs/Privacy.md) for the full list of what is and
is not collected, event schemas, CI auto-disable behavior, and storage
locations.

---

## :handshake: Contributions and Feedback

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For feature requests or bug reports, please file a [GitHub Issue](https://github.com/microsoft/winml-cli/issues).

---

## :balance_scale: Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## :page_facing_up: License

This project is licensed under the [MIT License](LICENSE.txt).

---

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft
sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
