# Installation

winml-cli is a Python toolkit for converting and optimizing PyTorch models to ONNX format, targeting deployment on the [Windows ML](https://learn.microsoft.com/en-us/windows/ai/windows-ml/) runtime. It supports multiple hardware backends including QNN (Qualcomm NPU), OpenVINO (Intel CPU/GPU), DirectML, and ONNX Runtime. To get started you need a Windows machine, Python 3.10, and the `uv` package manager.

## Prerequisites

| Component | Details |
|---|---|
| Windows | Windows 11 24H2 or later (required for NPU support) |
| Hardware | Copilot+PC with NPU (40+ TOPS recommended for NPU acceleration; CPU/DirectML works without an NPU) |
| Python | 3.10 (the project pins `requires-python = ">=3.10,<3.11"`) |
| Package manager | [`uv`](https://github.com/astral-sh/uv) |
| Version control | `git` |

!!! note "No NPU?"
    You can follow most of these docs without NPU hardware. Most winml-cli commands (`build`, `compile`, `perf`, `analyze`) accept `--device auto` and fall back to CPU or DirectML automatically. `winml eval` accepts only `cpu|gpu|npu` (no `auto`), so pass `--device cpu` explicitly there. The end-to-end tutorial documents an explicit CPU fallback path.

## Install

```bash
git clone https://github.com/microsoft/winml-cli.git
cd winml-cli
uv python install 3.10
uv sync
```

Cloning the repository pulls down all source code and configuration. `uv python install 3.10` downloads and pins the exact Python version the project requires. `uv sync` creates an isolated virtual environment and installs all declared dependencies from `pyproject.toml` in a single step. No separate `pip install` or manual venv activation is needed.

## Verify

```bash
uv run winml sys
```

Expected output (abbreviated):

```text
╭──────────────────────────────────╮
│   winml-cli System Information    │
╰──────────────────────────────────╯

Environment
  Python Version    3.10.x
  OS                Windows 11
  Machine           AMD64

ML Libraries
  Library        Version   Status
  torch          2.x.x     OK
  onnx           1.x.x     OK

Available Devices (priority order)
  #1  NPU   ...
  #2  GPU   ...
  #3  CPU   ...

Available Execution Providers
  QNNExecutionProvider           -> NPU
  DmlExecutionProvider           -> GPU
  CPUExecutionProvider           -> CPU
```

This command enumerates available compute devices and execution providers on your machine. If an expected device or SDK is missing, `winml sys` is the right place to diagnose it. See [winml sys](../commands/sys.md) for the full flag reference and troubleshooting tips.

## Optional extras

Two optional dependency groups are available for hardware-specific backends:

- `--extra openvino` — installs [OpenVINO](https://docs.openvino.ai/) for inference on Intel CPU and GPU targets.
- `--extra qnn` — installs `onnxruntime-qnn` for Qualcomm NPU support. Note: the `onnxruntime-qnn` package requires Python 3.11 or later, so this extra will not install any packages under the project's default Python 3.10 environment. It is reserved for future use when the project broadens its Python version support.

To install an extra:

```bash
uv sync --extra openvino
```

Both extras can be combined:

```bash
uv sync --extra openvino --extra qnn
```

## Next steps

- **[Quickstart](quickstart.md)** — export your first model in 5 minutes.
- **[End-to-End Tour](end-to-end.md)** — full pipeline targeting whatever hardware you have (NPU / GPU / CPU).
- **[How winml-cli Works](../concepts/how-it-works.md)** — the mental model.
