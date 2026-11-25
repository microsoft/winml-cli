# Installation

## Prerequisites

| Component | Details |
|---|---|
| Windows | Windows 11 24H2 or later (required for NPU support) |
| Hardware | Device with CPU, GPU, or NPU |
| Python | 3.11 |
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

!!! warning "Full CLI on Windows on Arm"
    Use x64 Python 3.11 under emulation for the complete export/evaluation toolchain. For native inference, see the [ARM64 runtime preview](#native-windows-arm64-runtime-preview).

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

### Native Windows ARM64 runtime preview

The preview runs `winml perf` on local ONNX models and on a local GGUF using
Windows ML Tasks and the llama.cpp CPU backend. The x64 and ARM64 wheels have the same distribution name
and version. The **Python process architecture**, not the host CPU, selects
the wheel and its dependencies. The CLI payload is Python and package data;
the native DLLs come from the platform-specific dependencies. Do not rename an
`any` wheel or install x64 DLLs into an ARM64 environment.

To reproduce from this checkout (PowerShell, no global activation or sync):

```powershell
foreach ($platform in 'win_arm64', 'win_amd64') {
    uv build --wheel "-C=--build-option=--plat-name=$platform" --out-dir dist\windows-preview
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
uv venv --python cpython-3.11-windows-aarch64-none temp\arm-runtime
$scripts = 'temp\arm-runtime\Scripts'
$wheel = Get-Item dist\windows-preview\winml_cli-0.5.0-py3-none-win_arm64.whl
uv pip install --python "$scripts\python.exe" --only-binary :all: $wheel.FullName
uv pip check --python "$scripts\python.exe"
& "$scripts\python.exe" -c "import platform, sysconfig; print(platform.machine(), sysconfig.get_platform())"
& "$scripts\winml.exe" sys --list-device
```

The architecture check must print `ARM64 win-arm64`. To verify **inference**
rather than discovery, use a local GGUF or the small pinned test fixture:

```powershell
curl.exe -fL "https://huggingface.co/ggml-org/tiny-llamas/resolve/99dd1a73db5a37100bd4ae633f4cfce6560e1567/stories260K.gguf" -o temp\stories260K.gguf
& "$scripts\winml.exe" perf -m temp\stories260K.gguf --runtime winml-runtime --device cpu `
    --prompt "Once upon a time" --context-length 128 --max-new-tokens 4 `
    --iterations 1 --warmup 0 --no-memory --format json --output temp\arm-result.json
```

Check `benchmark_info.effective_device` (`cpu`), `requests[0].generated_tokens`
(positive), and `response_text` in the JSON report. This is a smoke test, not
an accuracy or performance benchmark.

For classic ONNX, use the small fixtures **from the repository root**:

```powershell
$fp32 = 'tests\assets\resnet_fp32\model.onnx'
$w8a8 = 'tests\assets\resnet_w8a8\model.onnx'
& "$scripts\winml.exe" perf -m $fp32 --device cpu --ep cpu --iterations 1 --warmup 0 --no-memory
& "$scripts\winml.exe" perf -m $fp32 --device gpu --ep dml --iterations 1 --warmup 0 --no-memory
& "$scripts\winml.exe" perf -m $w8a8 --device npu --ep qnn --iterations 1 --warmup 0 --no-memory
```

On a Snapdragon X Elite, all three commands completed without the export stack.
The QNN command can provision an EP through Windows ML Catalog on first use.
In a separate ORT profile of the same fixtures, DirectML ran one fused node;
QNN ran one fused partition with the input quantize and output dequantize on CPU.
Selecting `--device npu` does not imply that every operation ran on the NPU.

**Limits and next steps:**

- Pre-exported ONNX `perf` needs `ml-dtypes` for ONNX dtype handling. Its
  native wheel is included; this does not verify Hugging Face model loading
  or model export. `winml run` is disabled in this CLI version, independently
  of ARM support.
- The full profile from `private/zhenni/arm` selects `pyarrow==25.0.0`;
  its PyPI release has no CPython 3.11 Windows ARM64 wheel. PyTorch's CPU
  index has native torchvision builds, but a `tool.uv.sources` entry is not
  a package index declaration for pip consumers. Do not change the x64
  pandas/Arrow pins or regenerate rule data to unblock runtime installation.
- [PR 1534](https://github.com/microsoft/winml-cli/pull/1534) used matching
  `2.9.31a0` preview packages and split direct Runtime/CGEP paths from the
  export wrappers. This preview uses `windowsml` and `windowsml-llama-core`
  `2.9.36a0` instead. `perf --runtime winml-runtime --backend cgc --device gpu`
  built a pipeline and completed a measured iteration on both the repository
  FP32 fixture and a local FP16 ResNet-50. The latter also ran through
  `--runtime winml-ort --device gpu --ep winmlcg`. This did not reproduce the PR's
  `0x887A0005`, but the model bytes, runtime and driver were not matched
  against the original report.
- Full developer-group `uv sync`, export/eval/build parity, accelerator
  accuracy across models, and a hosted release run are not certified by the ARM64 smoke.
  Validate each separately before promoting the preview.

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
