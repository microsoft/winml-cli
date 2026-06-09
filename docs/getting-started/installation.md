# Installation

## Prerequisites

| Component | Details |
|---|---|
| Windows | Windows 11 24H2 or later (required for NPU support) |
| Hardware | Copilot+PC with NPU (40+ TOPS recommended for NPU acceleration; CPU/DirectML works without an NPU) |
| Python | 3.11 (the project pins `requires-python = ">=3.11,<3.12"`) |
| Package manager | [`uv`](https://github.com/astral-sh/uv) |
| Version control | `git` |

!!! note "No NPU?"
    You can follow most of these docs without NPU hardware. All winml-cli commands accept `--device auto` and fall back to CPU or DirectML automatically. The end-to-end tutorial documents an explicit CPU fallback path.

## Install

```bash
uv python install 3.11
uv pip install winml-cli
```

`uv python install 3.11` downloads and pins the exact Python version the project requires. `uv pip install winml-cli` installs the latest release from PyPI into a managed environment. No separate venv activation is needed.

!!! tip "Install from source (for development)"
    If you want to contribute or run the latest unreleased code:

    ```bash
    git clone https://github.com/microsoft/winml-cli.git
    cd winml-cli
    uv sync
    ```

## Verify

```bash
winml sys
```

Expected output (abbreviated):

```text
╭──────────────────────────────────╮
│   winml-cli System Information    │
╰──────────────────────────────────╯

Environment
  Python Version    3.11.x
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

??? note "Optional extras (hardware-specific backends)"

    Two optional dependency groups are available:

    - `--extra openvino` — installs [OpenVINO](https://docs.openvino.ai/) for inference on Intel CPU and GPU targets.
    - `--extra qnn` — installs `onnxruntime-qnn` for Qualcomm NPU support.

    To install an extra:

    ```bash
    uv pip install winml-cli[qnn]
    ```

    Both extras can be combined:

    ```bash
    uv pip install winml-cli[openvino,qnn]
    ```

## Next steps

- **[Quickstart](quickstart.md)** — export your first model in 5 minutes.
- **[End-to-End Tour](end-to-end.md)** — full pipeline targeting whatever hardware you have (NPU / GPU / CPU).
- **[How winml-cli Works](../concepts/how-it-works.md)** — the mental model.
