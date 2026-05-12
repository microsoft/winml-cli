# ModelKit

**CLI toolkit to build portable, performant and high-quality models for Windows ML.**

![Status](https://img.shields.io/badge/status-early%20access-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

ModelKit bridges the gap between pretrained models and on-device inference.
Export from HuggingFace, optimize graphs, quantize weights, compile to device-specific binaries,
and benchmark — all from a single CLI. No separate vendor toolchain per silicon.
Built-in quality gates catch compatibility problems, suboptimal operators, and quantization
regressions — and suggest fixes automatically — before the model ever reaches a device.

---

## :dart: ModelKit Is Right for You If

- [x] You want to build models that run on **any Windows device** — Qualcomm, Intel, AMD, NVIDIA, or CPU
- [x] You want to benchmark a model with **one command** — latency, throughput, and live hardware utilization
- [x] You want to catch compatibility issues **ahead of time** — unsupported ops, shape mismatches, EP gaps
- [x] You want **deep insights** into your model — I/O shapes, task mapping, operator coverage per EP
- [x] You want a **repeatable and traceable** model building process — config-driven, inspectable at every stage
- [x] You want **AI agents** to build and profile models for you — agent-ready skills and structured JSON output

---

## :desktop_computer: Supported Hardware

| Provider | Hardware | Device Flag | Status |
|:---------|:---------|:------------|:------:|
| **QNN** | Qualcomm GPU and NPU | `--device npu` | :green_circle: Ready |
| **OpenVINO** | Intel CPU, iGPU, dGPU, and NPU | `--device npu` | :green_circle: Ready |
| **VitisAI** | AMD NPU | `--device npu` | :green_circle: Ready |
| **TensorRT** | NVIDIA discrete GPUs | `--device gpu` | :large_orange_diamond: Planned |
| **MIGraphX** | AMD discrete GPUs | `--device gpu` | :large_orange_diamond: Planned |
| **DirectML** | Hardware-agnostic GPU backend | `--device gpu` | :large_orange_diamond: Planned |
| **CPU** | Cross-platform fallback | `--device cpu` | :white_circle: Always |

> **Tip:** Use `--device auto` and ModelKit picks the best available device — NPU first, then GPU, then CPU.

---

## :package: Installation

**1. Create a Python 3.10 environment**

```bash
uv venv --python 3.10
.venv\Scripts\activate        # Windows
```

**2. Install from wheel**

```bash
uv pip install winml_modelkit-<version>-py3-none-any.whl
```

**3. Verify your environment**

```bash
winml sys --list-device --list-ep
```

This prints detected devices, available execution providers, and library versions — a quick sanity check before you start building.

---

## :wrench: Commands

| Category | Commands | Purpose |
|:---------|:---------|:--------|
| **Primitives** | `inspect` `export` `optimize` `quantize` `compile` | Single-stage building blocks |
| **Pipeline** | `config` `build` `perf` `eval` `run`\* | End-to-end orchestration |
| **Insights** | `analyze` `debug`\* | Diagnostics and compatibility |
| **Utilities** | `hub` `cache` `doctor` `setting` `sys` | Catalog, cache, and environment |

\* = coming soon

<details>
<summary><strong>Primitives</strong> — one stage at a time</summary>

| Command | Description |
|:--------|:------------|
| `winml inspect` | Discover model metadata, task, I/O shapes, and EP support |
| `winml export` | Convert a HuggingFace model to ONNX with hierarchy-preserving metadata |
| `winml optimize` | Fuse operators, simplify graphs, prepare for target EP |
| `winml quantize` | Compress to low-bit precision (int8, int16, mixed `w{x}a{y}`) with calibration |
| `winml compile` | Generate device-specific binaries (e.g., QNN context binaries) |

</details>

<details>
<summary><strong>Pipeline</strong> — orchestrated workflows</summary>

| Command | Description |
|:--------|:------------|
| `winml config` | Auto-detect task, I/O shapes, and optimal settings into a JSON build config |
| `winml build` | Execute the full pipeline: export, analyze, optimize, quantize, compile |
| `winml perf` | Benchmark latency, throughput, and hardware utilization with `--monitor` |
| `winml eval` | Evaluate model accuracy against reference datasets (ImageNet, GLUE, etc.) |
| `winml run`\* | End-to-end inference with pre/post processing |

</details>

<details>
<summary><strong>Insights</strong> — understand what is happening inside</summary>

| Command | Description |
|:--------|:------------|
| `winml analyze` | Lint operators, check EP compatibility, generate optimization config |
| `winml debug`\* | Interactive model debugging and layer-by-layer inspection |

</details>

<details>
<summary><strong>Utilities</strong> — environment and catalog</summary>

| Command | Description |
|:--------|:------------|
| `winml hub` | Browse the curated built-in model catalog with accuracy verdicts |
| `winml cache` | Manage built model artifacts and pipeline outputs |
| `winml doctor` | Diagnose environment issues (runtimes, providers, dependencies) |
| `winml setting` | Configure ModelKit preferences |
| `winml sys` | System information, device list, and EP capability reporting |

</details>

---

## :rocket: Quick Start

### :mag: Inspect a Model

Before building anything, ask ModelKit what it knows about your model:

```bash
winml inspect -m microsoft/resnet-50
```

ModelKit resolves the task, model class, I/O tensor shapes, and the export/quantize/compile
strategy — everything the build pipeline will use. Add `--format json` for machine-readable
output that agents and scripts can consume directly.

```bash
# JSON output for automation
winml inspect -m microsoft/resnet-50 --format json

# List all supported tasks
winml inspect --list-tasks

# Inspect with a specific task override
winml inspect -m google-bert/bert-base-uncased --task fill-mask
```

> **Golden rule:** always run `winml inspect -m <model>` before any pipeline command.

---

### :package: Build with Primitive Commands

Use individual commands for fine-grained control. Here is a ConvNeXT walkthrough:

```bash
# 1. Export HuggingFace model to ONNX
winml export -m facebook/convnext-base-224 -o convnext.onnx

# 2. Check EP compatibility
winml analyze -m convnext.onnx --device npu

# 3. Optimize the graph
winml optimize -m convnext.onnx -o convnext_opt.onnx

# 4. Quantize for NPU (int16 weights + int16 activations)
winml quantize -m convnext_opt.onnx --device npu --precision w16a16

# 5. Compile to device binary
winml compile -m convnext_opt_qdq.onnx --device npu

# 6. Benchmark
winml perf -m convnext_opt_qdq.onnx --device npu --iterations 100
```

Each step produces an inspectable artifact — you can stop, examine, tweak, and resume at any point.

---

### :gear: Build with Config + Build

Think of it like CMake: **`config` generates a build plan, `build` executes it.**

```bash
# Generate a build config (auto-detects task, shapes, quant settings)
winml config -m facebook/convnext-base-224 --device npu -o build_config.json

# Execute the full pipeline in one shot
winml build -c build_config.json -m facebook/convnext-base-224 -o output/
```

The config file is a plain JSON document — edit it to override precision, skip stages, or target
a different EP. You can also generate a config without downloading weights:

```bash
# Config from model type alone (no download required)
winml config --model-type bert --device npu -o bert_config.json

# Config with custom precision and no compilation stage
winml config -m microsoft/resnet-50 --precision w8a16 --no-compile -o resnet_config.json
```

The `build` command reads this config and orchestrates the full pipeline — export, analyze,
optimize, quantize, and compile — with a single invocation. Use `--no-quant` or `--no-compile`
to skip stages on the fly without editing the config file.

---

### :zap: Benchmark in One Command

Point `perf` at any model — HuggingFace ID or local `.onnx` file — and get latency stats instantly:

```bash
winml perf -m microsoft/resnet-50 --device npu --iterations 100

# With live hardware utilization chart
winml perf -m microsoft/resnet-50 --device npu --iterations 100 --monitor
```

The `--monitor` flag renders a live terminal chart showing NPU/GPU utilization, CPU%, and memory
during the benchmark run.

```bash
# Benchmark with custom precision and batch size
winml perf -m model.onnx --device npu --precision w8a16 --batch-size 4

# Force a specific execution provider
winml perf -m model.onnx --ep qnn --iterations 200 --warmup 20

# Save results to JSON for CI/CD integration
winml perf -m model.onnx --device npu -o results/perf_report.json
```

`perf` reports median latency, P90/P95/P99 percentiles, throughput (inferences/sec), and
memory usage. JSON output makes it easy to track regressions across builds.

---

## :arrows_counterclockwise: The BYOM Workflow

```
 Source Model
      |
      v
  +---------+     +---------+     +----------+     +----------+     +---------+
  | Export  | --> | Analyze | --> | Optimize | --> | Quantize | --> | Compile |
  +---------+     +---------+     +----------+     +----------+     +---------+
                      |                |                                  |
                      v                v                                  v
                  Portability      Performance                       Benchmark
                   Report          Report                            & Evaluate
```

Three quality gates guard the pipeline:

| Gate | Pillar | What It Catches |
|:-----|:-------|:----------------|
| :shield: **Analyze** | Portability | Unsupported ops, shape mismatches, EP compatibility gaps |
| :zap: **Optimize** | Performance | Suboptimal operator patterns, fusion opportunities, graph simplifications |
| :bar_chart: **Evaluate** | Fidelity | Accuracy regressions from quantization and compilation |

**How it works in practice:**

1. **Export** converts a HuggingFace (or custom) model into ONNX format with rich metadata
2. **Analyze** scans every operator against the target EP's capability matrix and flags issues
3. **Optimize** applies graph transformations — operator fusion, constant folding, layout optimization
4. **Quantize** compresses weights and activations to int8/int16 with calibration data
5. **Compile** produces device-specific binaries (e.g., QNN context binaries for Qualcomm NPUs)

At each stage, artifacts are saved to disk and can be inspected or edited before continuing.
The `build` command chains all five stages together, with the analyzer running in a loop to
auto-configure optimizations for maximum EP coverage.

---

## :clipboard: Built-in Models

Run `winml hub` to browse the full catalog interactively. Use `winml hub --model <id>` for per-model details.

<details>
<summary><strong>Click to expand the full model catalog</strong></summary>

| Model ID | Architecture | Task |
|:---------|:-------------|:-----|
| `ProsusAI/finbert` | bert | text-classification |
| `Intel/bert-base-uncased-mrpc` | bert | text-classification |
| `dslim/bert-base-NER` | bert | token-classification |
| `dbmdz/bert-large-cased-finetuned-conll03-english` | bert | token-classification |
| `Babelscape/wikineural-multilingual-ner` | bert | token-classification |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | roberta | text-classification |
| `w11wo/indonesian-roberta-base-posp-tagger` | roberta | token-classification |
| `google/vit-base-patch16-224` | vit | image-classification |
| `rizvandwiki/gender-classification` | vit | image-classification |
| `microsoft/swin-large-patch4-window7-224` | swin | image-classification |
| `facebook/convnext-tiny-224` | convnext | image-classification |
| `microsoft/resnet-50` | resnet | image-classification |
| `microsoft/table-transformer-detection` | table-transformer | object-detection |
| `mattmdjaga/segformer_b2_clothes` | segformer | image-segmentation |
| `nvidia/segformer-b1-finetuned-ade-512-512` | segformer | image-segmentation |
| `nvidia/segformer-b2-finetuned-ade-512-512` | segformer | image-segmentation |
| `nvidia/segformer-b5-finetuned-ade-640-640` | segformer | image-segmentation |

</details>

> Every model in the catalog has been validated end-to-end: export, quantize, device deployment, and accuracy verification.

---

## :warning: Scope & Limitations

| :white_check_mark: Supported | :x: Not Supported |
|:------------------------------|:-------------------|
| CNNs (ResNet, ConvNeXT, Swin) | LLMs (GPT, LLaMA, Phi, Mistral) |
| Vision Transformers (ViT, DeiT) | Diffusion models (Stable Diffusion, SDXL) |
| NLP classifiers (BERT, RoBERTa) | Decoder-only / seq2seq generative models |
| Token classifiers (NER, POS tagging) | Multi-modal generative models |
| Object detection (DETR, Table Transformer) | Models requiring custom CUDA kernels |
| Semantic segmentation (SegFormer) | Training or fine-tuning workflows |
| Image super-resolution (Swin2SR, ESRGAN) | |

> LLM and generative model support is on the roadmap.

---

## :world_map: Roadmap

| Milestone | Target | Highlights |
|:----------|:-------|:-----------|
| :yellow_circle: **Kickoff** | Q4 2025 | Internal prototype, core primitive commands |
| :green_circle: **Early Access** | Q1 2026 | First external testers, config + build pipeline, hub catalog |
| :blue_circle: **Public Beta** | Q2 2026 | Open source, agent skills, AI Toolkit integration |
| :purple_circle: **RC** | Q3-Q4 2026 | LLM + LoRA support, GPU and NPU expansion, MLIR backend |

<details>
<summary><strong>Click to expand roadmap details</strong></summary>

**Q4 2025 — Kickoff**
- Primitive commands: `inspect`, `export`, `optimize`, `quantize`, `compile`
- QNN, OpenVINO, and VitisAI execution provider support
- Internal validation with ResNet, BERT, ViT, SegFormer families

**Q1 2026 — Early Access**
- Pipeline commands: `config`, `build`, `perf`, `eval`
- Analyzer with auto-configuration loop
- Built-in model catalog (`winml hub`) with accuracy verdicts
- Live hardware monitoring (`--monitor`)

**Q2 2026 — Public Beta**
- Open source release
- Agent-ready skills for coding assistants
- AI Toolkit for VS Code integration
- Expanded model catalog: depth estimation, super-resolution, CLIP

**Q3-Q4 2026 — Release Candidate**
- LLM support (decoder-only architectures with LoRA adapters)
- TensorRT, MIGraphX, and DirectML execution providers
- MLIR-based optimization backend
- Public SDK and framework APIs

</details>

---

## :handshake: Contributing

*Coming soon.* We are working on contribution guidelines and will open the process during Public Beta.

---

## :page_facing_up: License

[MIT](../../LICENSE)
