# Installation

## Prerequisites

| Component | Details |
|---|---|
| Windows | Windows 11 24H2 or later (required for NPU support) |
| Hardware | Device with CPU, GPU, or NPU |
| Python | Standard GIL: x64 3.14 for the full CLI; native ARM64 3.13 for the runtime preview |
| Package manager | [`uv`](https://github.com/astral-sh/uv) |
| Version control | `git` |

!!! note "No NPU?"
    You can follow most of these docs without NPU hardware. All winml-cli commands accept `--device auto` and fall back to CPU or DirectML automatically. The tutorials document explicit CPU fallback paths.

## Install

On a Windows ARM64 machine, follow [Windows on Arm: choose an environment](#windows-on-arm-choose-an-environment) instead. The following quick install is for x64 Windows:

```bash
uv python install 3.14
uv venv --python 3.14
uv pip install winml-cli
```

`uv python install 3.14` downloads a managed interpreter; `uv venv --python 3.14` creates `.venv`, which `uv pip install` detects without activation. The repository root pins x64 CPython 3.14 in `.python-version`. Published releases may have different requirements until this migration is released.

!!! tip "Install from source (for development)"
    If you want to contribute or run the latest unreleased code:

    ```bash
    git clone https://github.com/microsoft/winml-cli.git
    cd winml-cli
    uv sync
    ```

### Windows on Arm: choose an environment

Both options run on the **same Windows ARM64 machine**. Start with the one
your workload needs; install both if you switch between them.

| Use case | Interpreter and wheel | Source project |
|---|---|---|
| Native ONNX or GGUF inference, Hugging Face export and CPU builds | ARM64 Python 3.13, `win_arm64` | `_envs\arm64` |
| Dataset evaluation, `eval --mode compare`, or `analyze` with Parquet rules | x64 Python 3.14 under emulation, `win_amd64` | Repository root |

The x64 environment supplies Hugging Face `datasets` and PyArrow, which the
native profile omits. This is a Python dependency difference, not a hardware
limitation. `WINMLCLI_RULES_DIR` selects where to find rules; it does not
install a Parquet reader.

**From a checkout:** choose the directory, then use the same commands in either
one. Stay at the repository root for x64, or run `cd _envs\arm64` for native
ARM64. `uv sync` uses that project's interpreter pin to create its own `.venv`.

```powershell
uv sync
uv run --no-sync python -c "import sysconfig; print(sysconfig.get_platform())"
uv run --no-sync winml sys --list-device
```

The architecture check prints `win-amd64` at the root and `win-arm64` in
`_envs\arm64`. Once both environments exist, switching directories is enough:
`uv run --no-sync winml ...` uses the environment for the current project.

**Installing a published release instead:** once a release includes both wheels,
create separate environments outside the checkout. `uv` selects the wheel
matching each interpreter. Run `winml.exe` from the environment you need:

```powershell
uv venv --python cpython-3.13-windows-aarch64-none .venv-arm64
uv pip install --python .venv-arm64\Scripts\python.exe --only-binary winml-cli `
    --find-links https://download-r2.pytorch.org/whl/cpu/torch/ `
    --find-links https://download-r2.pytorch.org/whl/cpu/torchvision/ winml-cli
.\.venv-arm64\Scripts\winml.exe sys --list-device

uv venv --python cpython-3.14-windows-x86_64-none .venv-x64
uv pip install --python .venv-x64\Scripts\python.exe --only-binary winml-cli winml-cli
.\.venv-x64\Scripts\winml.exe eval --help
```

#### Native Windows ARM64 runtime preview

The [ARM project](../../_envs/arm64/pyproject.toml) owns its interpreter pin
and dependencies. From the repository root:

```powershell
cd _envs\arm64
uv sync
```

One install includes CPU Torch/TorchVision, Transformers, Optimum ONNX,
ONNX Script and pandas. The source project configures the official PyTorch
CPU index for Torch/TorchVision. A wheel cannot configure the installer's
indexes, so the wheel installation above supplies the two PyTorch package
pages explicitly. Neither `datasets` nor PyArrow is installed on ARM64.
Build the ARM wheel from a complete checkout; build the release source
distribution from the repository root.

To verify local CPU inference from the ARM directory:

```powershell
uv run --no-sync winml perf -m ..\..\tests\assets\resnet_fp32\model.onnx `
    --device cpu --ep cpu --iterations 1 --warmup 0 --no-memory
```

The ARM64 preview has been tested with ONNX inference on CPU, DirectML and
QNN, and GGUF CPU generation. See the [wheel validation guide](../../_envs/arm64/README.md)
for release artifacts and how to compare real workloads on both architectures.

### Python 3.14 runtime policy

Python 3.14 is the default development and CI runtime. Root package metadata requires
Python 3.14 or later; Python 3.11, 3.12, and 3.13 are not supported by the x64 project.
Later Python releases have not been validated. Free-threaded builds are not
supported. The separate native ARM64 project uses standard CPython 3.13. This policy does not imply every GPU/NPU
provider or model passes: known pipeline, cache, dataset, and QNN failures
remain separate validation work.

From a source checkout, create or update the project environment:

```powershell
uv sync --all-extras
uv run --no-sync python -m winml.modelkit.cli --help
```

The root pin selects x64 even on Windows on Arm. Python 3.14
reports the Arm64 host from an emulated x64 process; provider DLL selection must
use the interpreter's architecture, not just `platform.machine()`.

NumPy, SciPy, ONNX, and Windows ML runtime now use one dependency constraint
set compatible with Python 3.14. PyArrow and
pandas remain exactly pinned to preserve the Parquet writer and metadata
format. This does not guarantee identical regenerated ONNX model bytes:
ONNX 1.20 changes the default IR/opset. Internal rule generators explicitly keep
IR 11 for compatibility with the previous artifact baseline, but operator and
shape-inference changes still require comparison of regenerated rules and models.
Test each execution provider needed by your workload before adopting 3.14.

For source validation, install development dependencies and run the suite:

```powershell
$env:WINML_CACHE_DIR = (Join-Path $PWD "temp\python314-cache")
uv sync --all-extras --all-groups
uv run --no-sync python -m pytest tests skills\adding-model-support\tests skills\auto-optimize\tests
uv run --no-sync python -m pytest tests\e2e -m "e2e or e2e_run or e2e_serve" --no-cov
uv run --no-sync python -m pytest tests\e2e -m "not e2e and not e2e_run and not e2e_serve" --no-cov
Remove-Item Env:WINML_CACHE_DIR
```

The second pytest command explicitly enables marked end-to-end tests, including
the normally disabled run/serve groups; the third covers unmarked tests in that
directory without the directory's default E2E skip. Some require model downloads,
additional fixtures, or compatible
hardware/provider packages; skips and environment failures are not passes.
If a native provider terminates pytest, rerun the affected files in separate
processes to preserve results for the rest of the suite.
Use a different `WINML_CACHE_DIR` when running another build job concurrently:
simultaneous exports must not write the same cached model artifacts.

The optional Qualcomm QAIRT SDK backend runs its vendor-pinned compiler in a
separate Python 3.11 subprocess environment. This is not a supported winml-cli
runtime or a second project test matrix. Migrating that SDK environment requires
a compatible vendor SDK, Torch, and ONNX stack; the current vendor requirements
include Torch 2.4.1, NumPy below 2, and ONNX below 1.17, which cannot simply be
installed in the CLI's 3.14 environment. This exception applies only to the
optional QAIRT SDK compiler backend, not the normal ORT QNN/NPU path.
The CLI itself runs on 3.14.
The standalone SDK compiler script also has a narrowly scoped Ruff target override
so formatting the CLI cannot introduce syntax unsupported by the vendor runtime.

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
  Python Version    3.14.x
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
