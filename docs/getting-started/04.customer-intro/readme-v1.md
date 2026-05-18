# ModelKit

CLI toolkit to build portable, performant and high-quality models for Windows ML — bridging the gap between pretrained models and on-device inference.

## ModelKit Is Right for You If

- You want to build models that run on **any Windows device**
- You want to benchmark a model with **one command**
- You want to catch compatibility issues **ahead of time**
- You want **deep insights** into your model
- You want a **repeatable and traceable** model building process
- You want **AI agents** to build and profile models for you

## Supported Hardware

| Provider | Hardware | Status |
|----------|----------|--------|
| QNN | Qualcomm GPU and NPU | :green_circle: Ready |
| OpenVINO | Intel CPU, iGPU, dGPU, and NPU | :green_circle: Ready |
| VitisAI | AMD NPU | :green_circle: Ready |
| TensorRT | NVIDIA discrete GPUs | :large_orange_diamond: Planned |
| MIGraphX | AMD discrete GPUs | :large_orange_diamond: Planned |
| DirectML | Hardware-agnostic GPU backend | :large_orange_diamond: Planned |
| CPU | Cross-platform fallback | :white_circle: Always |

> No NPU? Use `--device auto` — ModelKit falls back to GPU, then CPU.

## Installation

```bash
# Create a Python 3.10 virtual environment
uv venv --python 3.10
.venv/Scripts/activate

# Install from wheel
uv pip install winml_modelkit-<version>-py3-none-any.whl

# Sanity check — verify devices and execution providers
winml sys --list-device --list-ep
```

## Commands

| Category | Commands | Purpose |
|----------|----------|---------|
| Primitives | `inspect`, `export`, `optimize`, `quantize`, `compile` | Single-stage operations |
| Pipeline | `config`, `build`, `perf`, `eval`, `run`* | End-to-end workflows |
| Insights | `analyze`, `debug`* | Analysis and debugging |
| Utilities | `cache`, `doctor`, `setting`, `sys` | Environment management |

\* = coming soon

## Quick Start

### Inspect a Model

```bash
winml inspect -m microsoft/resnet-50
```

### Build with Primitive Commands

```bash
# Export HuggingFace model to ONNX
winml export -m facebook/convnext-base-224

# Analyze portability
winml analyze -m model.onnx

# Optimize graph
winml optimize -m model.onnx

# Quantize for NPU
winml quantize -m model.onnx --device npu

# Benchmark
winml perf -m model.onnx --device npu --iterations 100
```

### Build with Config + Build

Like CMake: `config` generates a build plan, `build` executes it.

```bash
# Generate a build config
winml config -m facebook/convnext-base-224 --device npu

# Execute the build
winml build
```

### Benchmark in One Command

```bash
winml perf -m facebook/convnext-base-224 --device npu --iterations 100 --monitor
```

## The BYOM Workflow

```
Source Model → Export → Analyze → Optimize → Quantize → Compile → Benchmark
```

Three quality gates guard the pipeline:

- **Analyze** — portability: catches unsupported ops and shape issues before they reach hardware
- **Optimize** — performance: graph transformations that reduce latency
- **Evaluate** — fidelity: measures accuracy loss from quantization and compilation

## Built-in Models

Run `winml hub` to see the full catalog.

| Model ID | Task | Architecture |
|----------|------|--------------|
| `microsoft/resnet-50` | image-classification | resnet |
| `google/vit-base-patch16-224` | image-classification | vit |
| `microsoft/swin-large-patch4-window7-224` | image-classification | swin |
| `facebook/convnext-tiny-224` | image-classification | convnext |
| `rizvandwiki/gender-classification` | image-classification | vit |
| `ProsusAI/finbert` | text-classification | bert |
| `dslim/bert-base-NER` | token-classification | bert |
| `microsoft/table-transformer-detection` | object-detection | table-transformer |
| `mattmdjaga/segformer_b2_clothes` | image-segmentation | segformer |
| `nvidia/segformer-b1-finetuned-ade-512-512` | image-segmentation | segformer |

> Golden rule: always run `winml inspect -m <model>` before any pipeline command.

## Scope & Limitations

- **Supported**: classic deep learning models — CNNs, vision transformers, NLP classifiers, token classifiers, object detection, segmentation
- **Not supported**: LLMs (GPT, LLaMA, Phi, Mistral), diffusion models (Stable Diffusion), or any decoder-only / seq2seq generative architecture
- LLM support is on the roadmap

## Roadmap

| Milestone | Target | Highlights |
|-----------|--------|------------|
| Kickoff | Q4 2025 | Internal prototype |
| Early Access | Q1 2026 | First external testers |
| Public Beta | Q2 2026 | Open source, agent skills, AITK integration |
| RC | Q3-Q4 2026 | LLM + LoRA, more devices, MLIR backend |

## Contributing

*Coming soon.*

## License

MIT
