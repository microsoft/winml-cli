# ModelKit

**CLI toolkit to build portable, performant and high-quality models for Windows ML — bridging the gap between pretrained models and on-device inference.**

ModelKit takes a pretrained model from Hugging Face (or a local ONNX file) and prepares it for on-device inference through a complete pipeline: export to ONNX, optimize the graph, quantize to low-bit precision, compile for target hardware, and benchmark on device. One toolkit covers every execution provider — QNN, OpenVINO, VitisAI, TensorRT, MIGraphX, DirectML, and CPU — so you never need a separate vendor toolchain per silicon.

---

## Table of Contents

- [ModelKit Is Right for You If](#modelkit-is-right-for-you-if)
- [Supported Hardware](#supported-hardware)
- [Installation](#installation)
- [Commands](#commands)
- [Quick Start](#quick-start)
- [The BYOM Workflow](#the-byom-workflow)
- [Built-in Models](#built-in-models)
- [Scope & Limitations](#scope--limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## ModelKit Is Right for You If

**You want to build models that run on any Windows device.**
ModelKit produces ONNX models that target the Windows ML runtime. A model prepared through ModelKit runs on any supported execution provider — Qualcomm NPU, Intel NPU, AMD NPU, NVIDIA GPU, or CPU — without per-device rework. Build once, run anywhere.

**You want to benchmark a model with one command.**
Run `winml perf -m <model-id> --device npu` and ModelKit handles everything behind the scenes — download, export, optimize, and benchmark. You get latency percentiles, throughput numbers, and live hardware utilization in seconds.

**You want to catch compatibility issues ahead of time.**
The built-in analyzer checks every operator against your target execution provider before you deploy, not after. It classifies operators as supported, partial, or unsupported — and generates an optimization config that fixes the issues automatically.

**You want deep insights into your model.**
`winml inspect` reveals task, model class, I/O tensor shapes, and EP compatibility without loading weights. `winml analyze` goes deeper — linting operators, detecting suboptimal patterns, and identifying performance bottlenecks in the graph.

**You want a repeatable and traceable model building process.**
`winml config` captures every build setting — task, I/O shapes, optimization flags, quantization parameters — into a single JSON file. Check it into source control, share it with your team, and replay builds deterministically on any machine. Think CMake for models.

**You want AI agents to build and profile models for you.**
ModelKit provides built-in skills that coding agents (Claude Code, Cursor, Copilot, and others) can consume. Agents can drive the entire build pipeline programmatically — from model selection through optimization to benchmarking.

---

## Supported Hardware

ModelKit supports the major execution providers in the Windows ML ecosystem. The three NPU providers are fully supported today; GPU and CPU providers are coming in future releases.

| Execution Provider | Hardware | Status | Device Flag |
|---|---|---|---|
| **QNN** | Qualcomm NPU (Snapdragon X Elite) | 🟢 Ready | `--ep qnn --device npu` |
| **OpenVINO** | Intel NPU (Meteor Lake / Lunar Lake) | 🟢 Ready | `--ep openvino --device npu` |
| **VitisAI** | AMD NPU (Ryzen AI) | 🟢 Ready | `--ep vitisai --device npu` |
| **TensorRT** | NVIDIA discrete GPUs | 🔶 Planned | `--ep tensorrt --device gpu` |
| **MIGraphX** | AMD discrete GPUs | 🔶 Planned | `--ep migraphx --device gpu` |
| **DirectML** | Hardware-agnostic GPU backend | 🔶 Planned | `--ep dml --device gpu` |
| **CPU** | Cross-platform fallback | ⚪ Always available | `--ep cpu --device cpu` |

**Automatic device selection.** If you are unsure which EP to use, pass `--device auto`. ModelKit will detect the best available device on your machine and select the appropriate execution provider automatically, falling back through NPU, GPU, and finally CPU.

---

## Installation

ModelKit requires **Python 3.10** and is distributed as a Python wheel. We recommend using [uv](https://docs.astral.sh/uv/) for fast, reproducible environment setup.

### Step 1: Create a virtual environment

```bash
uv venv --python 3.10
```

This creates a `.venv` directory with an isolated Python 3.10 environment. Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\activate

# Windows (Git Bash / WSL)
source .venv/Scripts/activate
```

### Step 2: Install ModelKit

```bash
uv pip install winml_modelkit-0.0.1.dev1-py3-none-any.whl
```

This installs the `winml` CLI and all required dependencies (ONNX Runtime, Hugging Face Transformers, optimization libraries, and more).

### Step 3: Sanity check

Verify that your system is ready:

```bash
winml sys --list-device --list-ep
```

This command prints your system information, detected hardware devices, and available execution providers. Confirm that your target device and EP appear in the output:

- **Snapdragon X Elite** — look for `QNNExecutionProvider`
- **Intel AI Boost** — look for `OpenVINOExecutionProvider`
- **AMD Ryzen AI** — look for `VitisAIExecutionProvider`

If no NPU is detected, you can still use ModelKit with `--device auto` for most commands. The only exception is `winml compile`, which requires an NPU device to generate device-specific binaries.

---

## Commands

ModelKit organizes its commands into four categories: **Primitives** for individual pipeline stages, **Pipeline** for orchestrated workflows, **Insights** for diagnostics, and **Utilities** for housekeeping.

### Summary

| Category | Commands | Purpose |
|---|---|---|
| **Primitives** | `inspect`, `export`, `optimize`, `quantize`, `compile` | Individual building blocks — one command per pipeline stage |
| **Pipeline** | `config`, `build`, `perf`, `eval`, `run`* | Orchestration, benchmarking, and evaluation |
| **Insights** | `analyze`, `debug`* | Diagnostics, compatibility checking, and debugging |
| **Utilities** | `cache`, `doctor`, `setting`, `sys` | Environment management and housekeeping |

*\* Coming soon*

### Primitives

These are the individual building blocks of the model preparation pipeline. Each command handles exactly one stage. You can run them standalone, reorder them, or compose them into custom workflows.

**`winml inspect`** — Discover model metadata. Prints the task, model class, input/output tensor names and shapes, and execution provider compatibility. No weights are loaded — this reads only the model configuration, making it fast and lightweight. Always run inspect first to verify a model is supported before feeding it into the pipeline.

**`winml export`** — Convert a source model to ONNX. Takes a Hugging Face model ID (or local checkpoint) and produces a standards-compliant ONNX file with hierarchy-preserving metadata. This is the entry point for any model that starts as a PyTorch checkpoint.

**`winml optimize`** — Fuse operators, simplify graphs, and prepare for target EPs. Takes an ONNX model and an optimization config (typically generated by `winml analyze`) and applies graph-level transformations: operator fusion, constant folding, shape inference, and EP-specific rewrites.

**`winml quantize`** — Compress to low-bit precision. Reduces model size and inference latency by converting weights and activations from FP32 to INT8 (or other low-bit formats). After quantization, the model is portable — it can run on any ONNX Runtime backend.

**`winml compile`** — Generate device-specific binaries. Takes a quantized ONNX model and produces EP-specific compiled artifacts (for example, QNN context binaries for Qualcomm NPU). This step locks the model to a specific device but delivers the lowest possible inference latency.

### Pipeline

Orchestration commands that chain primitives together, plus benchmarking and evaluation tools.

**`winml config`** — Auto-detect optimal settings into a JSON config. Inspects the model and generates a complete build specification: task, I/O shapes, optimization flags, quantization parameters, and target EP settings. The config file is reviewable, editable, and version-controllable — it is the single source of truth for your build.

**`winml build`** — Orchestrate the full pipeline. Takes a config file and executes every stage in sequence: export, analyze, optimize, quantize, and compile. Two commands (`config` + `build`) replace eight manual steps.

**`winml perf`** — Benchmark latency, throughput, and hardware utilization. Runs inference on the target device and reports latency percentiles (p50, p90, p99), throughput (inferences per second), and optionally live hardware monitoring (CPU, RAM, NPU utilization) with the `--monitor` flag. Can accept either a local ONNX file or a Hugging Face model ID — in the latter case, it handles the full pipeline automatically.

**`winml eval`** — Measure model accuracy against reference datasets. Compares the output of your optimized/quantized model against the original to quantify any accuracy loss introduced by the pipeline.

**`winml run`** — End-to-end inference with pre/post processing. *(Coming soon.)*

### Insights

Diagnostic tools for understanding what is happening inside your model.

**`winml analyze`** — Lint operators, check EP compatibility, and generate optimization config. The analyzer has two components. The **Linter** is like ESLint for ONNX — it checks every operator against target EPs and classifies each as supported (green), partial (gray), or unsupported (red). **AutoConf** detects suboptimal patterns in the graph and generates the optimization config that the optimizer consumes. Together they form the analyze-optimize loop.

**`winml debug`** — Interactive model debugging and layer-by-layer inspection. *(Coming soon.)*

### Utilities

Housekeeping and environment management commands.

**`winml cache`** — Manage built model artifacts and pipeline outputs. View, clean, or selectively remove cached models and intermediate files.

**`winml doctor`** — Diagnose environment issues. Checks runtimes, execution providers, and dependencies to identify configuration problems.

**`winml setting`** — Configure ModelKit preferences. Set default EPs, output directories, and other global options.

**`winml sys`** — System information and capability reporting. Prints detected hardware, available EPs, Python version, and installed package versions. The first command to run after installation.

---

## Quick Start

This section walks you through ModelKit from simplest to most detailed, using real commands and real models.

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

If inspect succeeds, the model is supported and you can proceed with the rest of the pipeline. If it prints an error or `Unsupported`, skip that model — it is not yet compatible with ModelKit.

**Golden rule: always inspect first.** Before running export, build, perf, or any other pipeline command, verify the model is supported with `winml inspect`.

### Build with Primitive Commands

This walkthrough builds **ConvNeXT** (`facebook/convnext-base-224`) step by step using primitive commands. ConvNeXT is a family of CNN models inspired by Vision Transformers, introduced by Meta in 2022 — it offers high accuracy while retaining the efficiency of CNNs.

The workflow has three phases: **Inspect**, **Build a Portable Model**, and **Benchmark on Device**.

#### Phase 1: Inspect

```bash
winml inspect -m facebook/convnext-base-224
```

This tells you everything about the model — task, model class, I/O shapes — without loading weights.

#### Phase 2: Build a Portable Model

**Export** from PyTorch to ONNX:

```bash
winml export -m facebook/convnext-base-224 -o convnext/model.onnx -v
```

The `-v` flag enables verbose output so you can see the export progress and any warnings.

**Analyze** for EP compatibility:

```bash
winml analyze -m convnext/model.onnx --optim-config optim.json
```

The analyzer checks every operator against the target EPs. It tells you what is supported, what is partial, what needs fixing — and it writes an optimization config (`optim.json`) that captures the recommended fixes.

**Optimize** the graph using the analyzer's config:

```bash
winml optimize -m convnext/model.onnx -c optim.json -o convnext/model_opt.onnx
```

The analyzer told you what to fix; the optimizer fixes it. This applies operator fusion, constant folding, and EP-specific rewrites.

**Quantize** to INT8:

```bash
winml quantize -m convnext/model_opt.onnx -o convnext/model_opt_int8.onnx
```

After this step you have a portable model — it can run on any ONNX Runtime backend.

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

Compare the two numbers. You should see roughly a **25x speedup** — the quantized model on NPU versus the original on CPU. Same model, same accuracy, completely different performance.

### Build with Config + Build

Same model, different approach. Instead of running each command manually, use the config-driven pipeline.

**Generate the build config:**

```bash
winml config -m facebook/convnext-base-224 -o convnext_config.json
```

This creates a JSON file containing all settings for every pipeline step — task, I/O shapes, optimization flags, quantization parameters — all auto-detected from the model. Open it in any editor to review or adjust. The config follows the same pattern as CMake: `winml config` is like `cmake -B build` (generate the specification), and `winml build` is like `cmake --build build` (execute it).

**Build the model:**

```bash
winml build -c convnext_config.json -m facebook/convnext-base-224 -o convnext_build/
```

This orchestrates the full pipeline — export, analyze, optimize, quantize, compile — all in one go. Same result as the eight manual steps above, but in two commands.

**Benchmark the result:**

```bash
winml perf -m convnext_build/model.onnx --ep qnn --iterations 100
```

Same model, same quality — two commands instead of eight.

The config file is the single source of truth for your build. You can version-control it, share it with teammates, edit it to override settings, and replay builds deterministically on any machine.

### Benchmark in One Command

The simplest way to evaluate a model — one command, zero setup:

```bash
winml perf -m facebook/convnext-base-224 --device npu --monitor
```

ModelKit handles everything behind the scenes: download the model from Hugging Face, export to ONNX, optimize the graph, and run the benchmark on your NPU. The `--monitor` flag enables live hardware monitoring — you will see real-time CPU utilization, RAM usage, and NPU activity alongside the latency results.

This is ideal for quick smoke tests: does the model run on this device, and how fast is it? Think of it as the QA step — validate, benchmark, deliver.

---

## The BYOM Workflow

The **Build Your Own Model** (BYOM) workflow is the philosophy behind ModelKit. It defines how a source model becomes a production-ready, device-optimized artifact.

### The Pipeline

```
Source Model → Export → Analyze → Optimize → Quantize → Compile → Benchmark
```

Each arrow is a ModelKit command. You can enter the pipeline at any stage (for example, start with a local ONNX file and skip export), exit early (stop after optimization if you do not need quantization), or loop back to repeat a stage with different settings.

### Three Quality Gates

The pipeline embeds three quality gates — checkpoints where ModelKit validates the model before proceeding. These three steps define the quality of your output.

**Analyze — Portability Gate.**
Does the model run on the target EP? The analyzer lints every operator and checks compatibility against your target execution providers. If an operator is unsupported, you find out here — before you spend time on optimization and quantization. The analyzer also generates the optimization config that feeds into the next stage.

**Optimize — Performance Gate.**
Is the graph efficient enough? The optimizer applies graph-level transformations (fusion, constant folding, shape inference, EP-specific rewrites) to produce a model that runs efficiently on the target hardware. Compare perf numbers before and after optimization to measure the improvement.

**Evaluate — Fidelity Gate.**
Is the model still accurate after quantization? Compressing from FP32 to INT8 reduces size and improves latency, but it can degrade accuracy. The eval command measures the difference so you can make an informed decision about the quality-performance tradeoff.

### The Analyze-Optimize Loop

The analyzer and optimizer work as a pair. The analyzer's **Linter** (like ESLint for ONNX) identifies compatibility issues and classifies operators. The analyzer's **AutoConf** (like GNU AutoConf for ONNX) detects suboptimal patterns and generates a fix config. The optimizer consumes that config and applies the transformations.

If the first pass does not achieve full compatibility, you can iterate: analyze again, review the updated config, optimize again. This loop is what makes models portable across execution providers.

---

## Built-in Models

ModelKit ships with a curated catalog of tested models. Run `winml hub` to list all available models.

| Model ID | Task | Architecture |
|---|---|---|
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

These models are verified against ModelKit's full pipeline and serve as reliable starting points for testing and experimentation. You are not limited to this list — any Hugging Face model that passes `winml inspect` is a valid input.

**Golden rule: inspect first.** Before running any pipeline command on a model not in this table, run `winml inspect -m <model-id>` to verify it is supported.

---

## Scope & Limitations

### What ModelKit supports

ModelKit targets **classic deep learning models** — CNNs, encoders, vision transformers, NLP classifiers, token classifiers, object detection models, and segmentation models. These are the architectures that run well on the ONNX Runtime execution providers available today.

Supported tasks include:
- Image classification (ResNet, ViT, Swin, ConvNeXT)
- Text classification (BERT, RoBERTa)
- Token classification / NER (BERT, RoBERTa)
- Object detection (Table Transformer)
- Image segmentation (SegFormer)

### What ModelKit does not support

**LLMs and generative models are not in scope.** Do not use ModelKit with GPT, LLaMA, Phi, Mistral, Stable Diffusion, or any model with a decoder-only or sequence-to-sequence generative architecture. LLM support (with LoRA) is planned for Q3-Q4 2026.

### Accepted inputs

ModelKit accepts two types of input:

- **Hugging Face model ID** (e.g., `microsoft/resnet-50`) — model weights are downloaded automatically on first use and cached locally.
- **Local ONNX file** (e.g., `model.onnx`) — produced by `winml export`, `winml build`, or any other ONNX exporter.

### Known constraints

- `winml compile` requires an NPU device. It cannot run without one. If no NPU is available, skip the compile step and use `--device auto` for benchmarking.
- Some models may export successfully but fail during optimization or quantization due to unsupported operator patterns. The analyzer will flag these issues.
- Performance numbers vary by device, driver version, and EP version. Always benchmark on your target hardware.

---

## Roadmap

**Q4 2025 — Project Kickoff.**
Initial development of the ModelKit CLI, core pipeline commands, and execution provider integrations.

**Q1 2026 — Early Access & Feedback.**
Internal release to partner teams. Validation across QNN, OpenVINO, and VitisAI execution providers. Bug bash and usability testing with real-world models.

**Q2 2026 — Public Beta.**
Open source release. Coding agent skills for Claude Code, Cursor, Copilot, and other AI-assisted development tools. Integration with AI Toolkit (AITK) for Visual Studio Code.

**Q3-Q4 2026 — Release Candidate.**
LLM support with LoRA adapters. Expanded device coverage for GPU and additional NPU platforms. MLIR integration for next-generation compiler backends.

---

## Contributing

Contributing guidelines are coming soon. If you are interested in contributing to ModelKit, please reach out to the WinPD team for early access and collaboration opportunities.

---

## License

MIT
