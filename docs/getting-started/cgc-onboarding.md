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

The CLI calls the Runtime compiler API to compile ONNX to temporary CG output IR,
then loads that artifact and runs it through DXCGC. No EP is selected and no
reusable CGIR artifact is retained. Applications using this path must likewise
invoke the compiler API before pipeline execution.

### LLM model with Runtime Task API

The Windows ML Task API handles prompt tokenization and text generation,
streaming generated tokens back to the application. Use this path to benchmark
LLM generation on the GPU with CGC. No ONNX or MLIR export is required.

**Safetensors checkpoint:** Download [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
into `.\Qwen3-0.6B`, keeping the weights, configuration, and tokenizer files together.
Pass the folder, not an individual `.safetensors` file:

```powershell
winml perf -m .\Qwen3-0.6B --backend cgc -o .\checkpoint-perf.json
```

**GGUF:** Download [Qwen3-0.6B-Q4_0.gguf](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q4_0.gguf)
into the current directory, then pass the file:

```powershell
winml perf -m .\Qwen3-0.6B-Q4_0.gguf --backend cgc -o .\gguf-perf.json
```

Both commands select the CGC backend, report time to first token and generation
throughput, and save the results as JSON.

Illustrative JSON excerpt (values vary by model and device):

```json
{
  "benchmark_info": {
    "runtime": "winml-runtime",
    "backend": "cgc",
    "model_format": "gguf"
  },
  "aggregate": {
    "warmup_excluded": true,
    "task_ttft_duration_ms": {
      "mean": 20
    },
    "steady_state_decode_tokens_per_second": {
      "mean": 500
    },
    "request_duration_ms": {
      "mean": 300
    }
  }
}
```

These fields show the average Task time to first token, subsequent generation
speed, and total request time, excluding warmup requests.

## Build the model

Build a reusable `.mlir` model directly. No configuration file is required.
For PyTorch models, build first exports to ONNX. It then quantizes the model,
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

### Fix dynamic input dimensions

CGIR export automatically freezes a symbolic input dimension named `batch_size`
to 1 unless explicitly overridden, and prints a warning. If other named input
dimensions remain dynamic, conversion stops with the input names, axis indices,
missing dimensions, and a `freeze-dims` option template. Substitute your
deployment dimensions; freezing specializes the artifact, not the input images.

Use the same repeatable `--cgc-options KEY=VALUE` syntax for export, config, and build.
For example, for a model whose input symbols are `dynamic_height` and
`dynamic_width`:

```powershell
winml export -m .\model.onnx --target cgir -o .\model.mlir `
  --cgc-options "freeze-dims=dynamic_height=720,dynamic_width=1280"

winml config -m .\model.onnx --backend cgc -o .\build.json `
  --cgc-options "freeze-dims=dynamic_height=720,dynamic_width=1280"
winml build -m .\model.onnx -c .\build.json -o .\model-cgir

winml build -m .\model.onnx --backend cgc -o .\model-cgir-direct `
  --cgc-options "freeze-dims=dynamic_height=720,dynamic_width=1280"
```

Config generation stores these values in `convert.options`, for example:

```json
{
  "convert": {
    "target": "cgir",
    "options": {
      "freeze-dims": "dynamic_height=720,dynamic_width=1280"
    }
  }
}
```

When overriding `freeze-dims` on the command line, include all required dimensions:
the value replaces the entire list in the configuration.

Fixing input dimensions does not guarantee successful conversion; data-dependent
shapes or unsupported operators may still prevent it.


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
- [`winml optimize`](../commands/optimize.md) documents ONNX optimization.
- [`winml perf`](../commands/perf.md) documents `.mlir`, `winml-runtime`, and
  WinMLCG performance measurement.
- [`winml eval`](../commands/eval.md) documents evaluation of supported CGIR
  models with compatible preprocessing metadata.
- [`winml config`](../commands/config.md) and
  [`winml build`](../commands/build.md) document configuration-driven CGIR
  conversion.