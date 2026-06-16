# WinML CLI

[![WinML CLI CI](https://github.com/microsoft/winml-cli/actions/workflows/modelkit-ci.yml/badge.svg)](https://github.com/microsoft/winml-cli/actions/workflows/modelkit-ci.yml)
![Status](https://img.shields.io/badge/status-preview-blue)
[![PyPI release](https://img.shields.io/pypi/v/winml-cli)](https://pypi.org/project/winml-cli/)
![License](https://img.shields.io/badge/license-MIT-green)

**Windows ML CLI** is a command line tool for building **portable, performant, and high-quality** AI models for Windows ML. It takes you from a source model — whether from Hugging Face or your own pipeline — to a hardware-optimized artifact in a reproducible workflow.

Purpose-built for Windows hardware diversity, the CLI handles conversion, graph optimization, and compilation across AMD, Intel, NVIDIA, and Qualcomm targets. The CLI fits naturally into CI/CD pipelines so teams can validate and ship models easily.

---

## What you can do

- **Build once, run across hardwares.** Compose your own workflow from primitive commands (`export`, `analyze`, `optimize`, `quantize`, `compile`), or use an auto-generated config with `winml build` - both produce portable models that run across hardware.
- **Drill into the details.** Deep insights into operator compatibility, shape mismatches, graph optimizations, and EP-aware tuning at any stage of the pipeline.
- **AI-ready.** CLI-driven tools with built-in skills, friendly to work with mainstream agents.

## What you get out of the box

- **All Windows ML EPs supported.** Every [supported execution provider](https://microsoft.github.io/winml-cli/latest/concepts/eps-and-devices/#eps-winml-cli-supports) is available behind the same commands.
- **Curated model catalog.** A [verified set of models](https://microsoft.github.io/winml-cli/latest/reference/supported-models/) that run across all Windows ML EPs - a reliable starting point.
- **Bring your own ONNX.** Not only for converting from PyTorch - bring an [existing ONNX model](https://microsoft.github.io/winml-cli/latest/tutorials/build-from-onnx/) to get operator-compatibility insights and optimize it based on the analysis.

---

## 🎯 Getting Started

### Prerequisites

| Component | Details |
|---|---|
| Windows | Windows 11 24H2 or later (required for NPU support; earlier versions work for CPU/GPU) |
| Python | 3.11 |
| Package manager | [uv](https://github.com/astral-sh/uv) |
| WinML CLI | [PyPI](https://pypi.org/project/winml-cli/) |

### Installation

WinML CLI requires **Python 3.11** and is distributed as a Python wheel. We recommend [uv](https://docs.astral.sh/uv/) for fast, reproducible environment setup.

**1. Create an environment**

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

**2. Install winml-cli**

```bash
uv pip install winml-cli
```

**3. Verify your environment**

```bash
uv run winml sys --list-device --list-ep
```

This command enumerates available compute devices and execution providers on your machine. If an expected device or execution provider is missing, `winml sys` is the right place to diagnose it. See [winml sys](https://microsoft.github.io/winml-cli/latest/commands/sys/) for the full flag reference and troubleshooting tips.

## 🚀 Quick Start

### Inspect the model

Before downloading any models, confirm that winml-cli recognises the model:

```bash
uv run winml inspect -m microsoft/resnet-50
```

💡 Tip: Always inspect before build to catch unsupported architectures early.

### Build the model

```bash
uv run winml build -m microsoft/resnet-50 -o resnet_out/ --no-quant
```

`winml build` runs all pipeline steps in sequence — export, optimize, quantize. You can start a model build without a config file, or provide one to configure each step in the sequence (see [`winml config`](https://microsoft.github.io/winml-cli/latest/commands/config/) to customize). All intermediate artifacts land in `resnet_out/`. For more details, see [Output Layout - Windows ML CLI](https://microsoft.github.io/winml-cli/latest/reference/output-layout/#file-categories).

### Benchmark the model

```bash
uv run winml perf -m resnet_out/model.onnx --device auto --iterations 50 --monitor
```

`--device auto` lets the CLI resolve the best available device on your machine — NPU first, then GPU, then CPU.

---

## 🔀 Try Other Ways

- **Use with AI Agent** — See details at [Use with AI Agent - Windows ML CLI](https://microsoft.github.io/winml-cli/latest/getting-started/agent-skill/).
- **UI Quickstart** — See also [UI Quickstart - Windows ML CLI](https://microsoft.github.io/winml-cli/latest/getting-started/ui-quickstart/).

---

## 📚 Learn More
- **[Full Documentation](https://microsoft.github.io/winml-cli/latest/)** — Access the complete wiki for detailed guides, API references, and troubleshooting.
- **[Supported Models](https://microsoft.github.io/winml-cli/latest/reference/supported-models/)** — Browse the curated catalog of verified models that run across all Windows ML EPs.
- **[Execution provider compatibility](https://microsoft.github.io/winml-cli/latest/reference/supported-models/#execution-provider-compatibility)** — Browse the compatible EP alias and device combinations.


---

## 🤝 Contributing

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For feature requests or bug reports, please file a [GitHub Issue](https://github.com/microsoft/winml-cli/issues).

### Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

### License

This project is licensed under the [MIT License](LICENSE.txt).
