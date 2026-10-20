# Installation

## Prerequisites

| Component | Details |
|---|---|
| Windows | Windows 11 24H2 or later (required for NPU support) |
| Hardware | Device with CPU, GPU, or NPU |
| Python | 3.11 recommended; 3.14 experimental (x64, standard GIL build) |
| Package manager | [`uv`](https://github.com/astral-sh/uv) |
| Version control | `git` |

!!! note "No NPU?"
    You can follow most of these docs without NPU hardware. All winml-cli commands accept `--device auto` and fall back to CPU or DirectML automatically. The tutorials document explicit CPU fallback paths.

## Install

```bash
uv python install 3.11
uv pip install winml-cli
```

`uv python install 3.11` downloads and pins the exact Python version the project requires. `uv pip install winml-cli` installs the latest release from PyPI into a managed environment. No separate venv activation is needed.

!!! warning "Windows on Arm requires x64 Python"
    winml-cli supports **x64 (AMD64) Python only**. On an Arm64 device, native Arm64 Python cannot install the dependencies because PyTorch and the Windows ML runtime packages publish no `win_arm64` wheels — `uv sync` fails with an unsupported-platform error. Install an x64 build of Python 3.11 and point `uv` at it:

    ```powershell
    uv python install cpython-3.11-windows-x86_64-none
    uv sync --python cpython-3.11-windows-x86_64-none
    ```

    The x64 interpreter runs under Windows emulation on Arm64 hardware; `winml sys` still reports the Arm64 machine, and the NPU/GPU/CPU providers work normally.

!!! tip "Install from source (for development)"
    If you want to contribute or run the latest unreleased code:

    ```bash
    git clone https://github.com/microsoft/winml-cli.git
    cd winml-cli
    uv sync
    ```

### Experimental Python 3.14

Python 3.11 remains the recommended production interpreter. Python 3.14 is an
experimental source-build target; passing CPU tests does not certify every
GPU/NPU provider or model. Python 3.12 and 3.13 are accepted by package metadata
but have not been validated in this upgrade. Free-threaded builds and native
Arm64 Python are not supported.

From a source checkout, use a separate environment:

```powershell
uv python install cpython-3.14-windows-x86_64-none
$env:UV_PROJECT_ENVIRONMENT = "temp\python314"
uv sync --python cpython-3.14-windows-x86_64-none --all-extras
uv run --no-sync python -m winml.modelkit.cli --help
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

The explicit x64 interpreter is required on Windows on Arm too. Python 3.14
reports the Arm64 host from an emulated x64 process; provider DLL selection must
use the interpreter's architecture, not just `platform.machine()`.

Python 3.14 selects newer NumPy, SciPy, ONNX, and Windows ML runtime versions;
older interpreters retain their existing dependency constraints. PyArrow and
pandas remain exactly pinned to preserve the Parquet writer and metadata
format. This does not guarantee identical regenerated ONNX model bytes:
ONNX 1.20 changes the default IR/opset. Internal rule generators explicitly keep
IR 11 for compatibility with the previous artifact baseline, but operator and
shape-inference changes still require comparison of regenerated rules and models.
Test each execution provider needed by your workload before adopting 3.14.

For source validation, install development dependencies and run the suite:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "temp\python314"
$env:WINML_CACHE_DIR = (Join-Path $PWD "temp\python314-cache")
uv sync --python cpython-3.14-windows-x86_64-none --all-extras --all-groups
uv run --no-sync python -m pytest tests skills\adding-model-support\tests skills\auto-optimize\tests
uv run --no-sync python -m pytest tests\e2e -m "e2e or e2e_run or e2e_serve" --no-cov
uv run --no-sync python -m pytest tests\e2e -m "not e2e and not e2e_run and not e2e_serve" --no-cov
Remove-Item Env:UV_PROJECT_ENVIRONMENT
Remove-Item Env:WINML_CACHE_DIR
```

The second pytest command explicitly enables marked end-to-end tests, including
the normally disabled run/serve groups; the third covers unmarked tests in that
directory without the directory's default E2E skip. Some require model downloads,
additional fixtures, or compatible
hardware/provider packages; skips and environment failures are not passes.
If a native provider terminates pytest, rerun the affected files in separate
processes to preserve results for the rest of the suite.
Use a different `WINML_CACHE_DIR` when testing another interpreter concurrently:
simultaneous exports must not write the same cached model artifacts.

## Verify

```bash
winml sys
```

Expected output (abbreviated):

```text
+------------------------------------+
|   winml-cli System Information     |
+------------------------------------+

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

This command enumerates available compute devices and execution providers on your machine. If an expected device or execution provider is missing, `winml sys` is the right place to diagnose it. See [winml sys](../commands/sys.md) for the full flag reference and troubleshooting tips.

## Next steps

- **[Quickstart](quickstart.md)** — export your first model in 5 minutes.
- **[End-to-End Tour](quickstart.md)** — full pipeline targeting whatever hardware you have (NPU / GPU / CPU).
- **[How winml-cli Works](../concepts/how-it-works.md)** — the mental model.
