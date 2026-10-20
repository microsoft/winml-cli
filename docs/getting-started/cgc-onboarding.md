# Preparing your model for the Windows ML Runtime APIs and DX CGC

CGC onboarding has two journeys:

1. **Benchmark the model:** Explore model compatibility and performance through the runtime
   path that your application will use.
2. **Build the model:** Build a reusable `.mlir` model from PyTorch or ONNX.

## Environment setup

Create a virtual environment and install WinML CLI:

```powershell
uv venv --python 3.11
uv pip install winml-cli
```

Install the experimental wheels:

```powershell
uv pip install "windowsml==<windowsml-version>" "onnxruntime-windowsml==<onnxruntime-windowsml-version>"
```
If you want to restore to stable wheel, you can run `uv sync`


Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

## Before you start

Inspect the available devices and execution providers:

```powershell
winml sys --list-device --list-ep
```

Verify that the WinMLCG EP is installed and that the target GPU and its
CGC-capable driver are listed. If they are missing, install or update the
Windows ML components and graphics driver before continuing.

## Benchmark the model

Use `winml perf` to explore model compatibility and performance before changing
your application. It exercises the same paths available to the application and
lets you compare them without first writing integration code. Use
[`winml eval`](../commands/eval.md) similarly to evaluate accuracy; the runtime
and EP options select the same CGC paths shown below.

### MLIR with Runtime API

Use this path when you already have an offline-converted `.mlir` model:

```powershell
winml perf -m .\model.mlir --runtime winml-runtime -o .\model-mlir-perf.json
```

> **Note:** Currently, the graph and weights must be embedded in a single
> `.mlir` file. External weights will be supported in a future release.

The Runtime API loads the model directly and runs it through DXCGC. A
`.mlir` model can only use the Windows ML Runtime API; it cannot be loaded by
ONNX Runtime and does not use an EP.

### ONNX with WinMLCG EP

Use this path when the application uses ONNX Runtime and should keep ONNX as
its model contract:

```powershell
winml perf -m .\model.onnx --runtime winml-ort --ep winmlcg -o .\model-winmlcg-perf.json
```

The WinMLCG Execution Provider converts the model for CGC when the ONNX Runtime
session is created. No reusable CGIR artifact is produced. After this test
succeeds, keep the ONNX model and configure the WinMLCG EP in the application.

### ONNX with Runtime API

Use this path when the application uses the Windows ML Runtime API but should
keep ONNX as its model contract:

```powershell
winml perf -m .\model.onnx --runtime winml-runtime -o .\model-runtime-perf.json
```

The CLI calls the Runtime compiler API to convert ONNX to temporary CGIR,
then loads that artifact and runs it through DXCGC. No EP is selected and no
reusable CGIR artifact is retained. Applications using this path must likewise
invoke the compiler API before pipeline execution.

!!! warning
  With `windowsml==2.7.25.dev0`, the tested CLI's online conversion path
  fails with `Runtime.load_model() got an unexpected keyword argument 'io_counts'`.
  The offline build and MLIR Runtime path below works with this combination.

| Model input | Runtime path | EP | Conversion |
| --- | --- | --- | --- |
| `.onnx` | Windows ML ONNX Runtime | WinMLCG | When the session is created |
| `.onnx` | Windows ML Runtime API + DXCGC | None | When the session is created |
| `.mlir` | Windows ML Runtime API + DXCGC | None | Already converted offline |

## Build the model

Build a reusable `.mlir` model directly. No configuration file is required.
For PyTorch models, build first exports to ONNX. It then applies CGC
compatibility rewrites for supported operator patterns, quantizes the model,
and converts it to MLIR.

> **Note:** By default, the CGC build converts FP32 to FP16. This may improve
> performance, but can reduce accuracy for some models. Add `--no-quant` to
> skip quantization if your model is already quantized or accuracy is a concern.

### PyTorch → ONNX → CGIR

Build from a Hugging Face model:

```powershell
winml build -m microsoft/resnet-50 --backend cgc -o .\model-cgir
```

### ONNX → CGIR

Build from an existing ONNX model:

```powershell
winml build -m .\model.onnx --backend cgc -o .\model-cgir
```

Both paths write `model-cgir\model.onnx` and `model-cgir\model.mlir`.
Benchmark the `.mlir` model using the
[Windows ML Runtime API](#mlir-with-runtime-api).

### Convert without optimization or quantization

Use `winml export` to convert an ONNX model directly to MLIR, skipping the
optimization and quantization stages:

```powershell
winml export -m .\model.onnx --target cgir -o .\model.mlir
```

The model must already be compatible with the CGC converter.

### CGIR to device-targeted CG Output IR

To retain a compiled artifact from existing CGIR, use the explicit compile mode.
This experimental mode remains callable but is not advertised in public command
help or the public command reference.

```powershell
winml compile -m .\model.mlir --compiler cgc --device gpu -o .\model-output.mlir
```

`--compiler cgc` is optional for `.mlir` and `.mlirbc` input. Specify it when
the CGIR file uses another suffix or when you want to make the backend explicit.

The command loads CGIR through Windows ML Runtime, without an ONNX Runtime EP.
Both textual and bytecode CGIR are accepted; invalid or unsupported input is
reported by the Runtime. Supply a complete executable model with its weights,
not a graph-only report projection.

Compilation calls the selected GPU target's Runtime
`ModelCompiler.compile_to_file()` with `CompiledModelForm.DEVICE_TARGETED`.
The saved artifact is **MLIR bytecode**, even when named `.mlir`; text dumping
is not provided by this command. `.mlirbc` is also accepted as an output suffix.
No optimization or quantization pipeline runs.

This first version supports GPU only (the default in explicit IR mode). A single
GPU is selected automatically; multiple GPUs require `--device-luid`. GPU
discovery uses DXCore, not EP discovery or provider installation.

Default `--validate` reloads the compiled artifact, builds a Runtime pipeline and
checks that its I/O schema matches the input. Unlike the EPContext validation
path, this does not execute inference or certify accuracy. `--no-validate` skips
that reload/build check; genuine Input IR parsing and compilation remain required.
Run `winml perf` separately for performance.

The command publishes the artifact and any Runtime-produced resource companions
only after successful compilation and requested validation. It preserves an
existing input's `<stem>_metadata.json` I/O bindings as output metadata and writes
`<output-stem>_compile.json` with input/artifact hashes, output encoding, GPU LUID
and validation status. Metadata and resource collisions also require
`--overwrite`; the input and its metadata are never replaced. Without `-o`, the
default is `<input-stem>_output.mlir` in `--output-dir` or beside the input.

`--ep`, `--ep-options`, `--embed`/`--no-embed`, `--qnn-sdk-root`, `--list`,
`--config` and multiple `-m` inputs are rejected in this mode. Use the existing
ONNX-to-EPContext mode for EP compilation. Omitting the new selectors preserves
the original command behavior.

The preview `windowsml` payload must include Runtime APIs compatible with the
active CLI. If preview wheels override stable project pins, invoke the
environment directly or use `uv run --no-sync` to avoid restoring stable
packages.

## Tutorials and samples

- [ResNet-50 - PyTorch to CGIR (experimental)](../samples/resnet50-cgir.md) walks through
  the complete Hugging Face PyTorch → ONNX → CGIR build, performance, and
  evaluation workflow.
- [YOLO11 - ONNX to CGIR (experimental)](../samples/yolo11-cgir.md) provides an end-to-end
  conversion journey: export a checkpoint to ONNX, build CGIR, collect GPU
  performance results, compare raw output tensors, and try the WinMLCG EP alternative.

## Command references

- [`winml export`](../commands/export.md) documents `--target cgir` and CGIR
  export options.
- [`winml optimize`](../commands/optimize.md) documents ONNX optimization and
  CGC compatibility rewrites.
- [`winml perf`](../commands/perf.md) documents `.mlir`, `winml-runtime`, and
  WinMLCG performance measurement.
- [`winml eval`](../commands/eval.md) documents evaluation of supported CGIR
  models with compatible preprocessing metadata.
- [`winml config`](../commands/config.md) and
  [`winml build`](../commands/build.md) document configuration-driven CGIR
  conversion.