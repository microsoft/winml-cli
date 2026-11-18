# Use the CGC backend with Windows ML CLI

> [!IMPORTANT]
> The CGC backend is experimental, **not supported** for use in production environments, and new in Windows ML 2.9 experimental.

CGC is the Windows-native backend that compiles a model into programs for a Direct3D 12 device and runs them on the GPU target your app picked. It runs on the GPU only, with no CPU fallback. Finding a GPU doesn't establish that CGC supports its driver or a given model: a model that CGC can't run fails to load with an error.


This article explains how to convert a model to a format supported by CGC and run it using the Windows ML CLI.

## Environment setup

First, follow the guidance in [The CGC backend in Windows ML](https://review.learn.microsoft.com/en-us/windows/ai/new-windows-ml/runtime/cg-backend?branch=winml%2F2.9-exp) to choose recommended hardware and prepare your environment. Then, follow the instructions below to set up a Python environment for winml-cli.

Create a virtual environment and install WinML CLI version 0.5.0 or above:

```powershell
uv venv --python 3.11
uv pip install winml-cli
```

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```


## Build the model

The `.mlir` format is the model intermediate representation (IR) that can be executed by the CGC backend. For recommended models, see [The CGC backend in Windows ML](https://review.learn.microsoft.com/en-us/windows/ai/new-windows-ml/runtime/cg-backend?branch=winml%2F2.9-exp#recommended-models).

### Export PyTorch model to MLIR

Export a Hugging Face PyTorch model to MLIR:

```powershell
winml export -m microsoft/resnet-50 --target cgir -o .\model.mlir
```

### Export ONNX model to MLIR

Export an existing ONNX model to MLIR:

```powershell
winml export -m .\model.onnx --target cgir -o .\model.mlir
```

You can also use `winml build` to build the model with additional configuration options. For details, see the [`winml build` command reference](../commands/build.md).

## Benchmark the model

Use `winml perf` to explore model compatibility and performance before changing your application. It exercises the same paths available to the application and lets you compare them without first writing integration code.

`winml perf` supports ONNX (`.onnx`), MLIR (`.mlir`), and GGUF (`.gguf`) files, as well as local safetensors checkpoint directories, through the runtime/backend paths described below. For recommended models, see [The CGC backend in Windows ML](https://review.learn.microsoft.com/en-us/windows/ai/new-windows-ml/runtime/cg-backend?branch=winml%2F2.9-exp#recommended-models).

### Benchmark .mlir model

Use this path when you already have an offline-converted `.mlir` model:

```powershell
winml perf -m .\model.mlir --runtime winml-runtime -o .\model-mlir-perf.json
```

A `.mlir` model can only run on the CGC backend through the [Windows ML Runtime API](https://review.learn.microsoft.com/en-us/windows/ai/new-windows-ml/runtime/overview?branch=winml%2F2.9-exp), selected using the `--runtime winml-runtime` CLI option. It cannot be loaded by ONNX Runtime.

### Benchmark .onnx model

The `winml perf` command supports running ONNX models through either the Windows ML Runtime API or ONNX Runtime.

#### Via ONNX Runtime with WinMLCG EP

The WinMLCG Execution Provider (EP) makes CGC functionality available through ONNX Runtime.

Use this path when the application uses ONNX Runtime and should keep ONNX as its model contract:

```powershell
winml perf -m .\model.onnx --runtime winml-ort --ep winmlcg -o .\model-winmlcg-perf.json
```

The WinMLCG Execution Provider converts the model for CGC when the ONNX Runtime session is created. No reusable CGIR artifact is produced. After this test succeeds, keep the ONNX model and configure the WinMLCG EP in the application.

#### Via Windows ML Runtime API

Use this path when the application uses the Windows ML Runtime API but should keep ONNX as its model contract:

```powershell
winml perf -m .\model.onnx --runtime winml-runtime -o .\model-runtime-perf.json
```

The CLI calls the [Runtime compiler API](https://review.learn.microsoft.com/en-us/windows/ai/new-windows-ml/runtime/cg-backend?branch=winml%2F2.9-exp#compile-a-model-for-cg) to compile ONNX to temporary, device-targeted CGC output IR, then loads that artifact and runs it through CGC. No EP is selected and no reusable CGIR artifact is retained. Applications using this path must likewise invoke the compiler API before pipeline execution.

### Benchmark LLM model (gguf and safetensor)

The [Windows ML Task API](https://review.learn.microsoft.com/en-us/windows/ai/new-windows-ml/tasks/overview?branch=winml%2F2.9-exp) handles prompt tokenization and text generation, streaming generated tokens back to the application. Use this path to benchmark LLM generation on the GPU with CGC. No ONNX or MLIR export is required.

**Safetensors checkpoint:** Download [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) into `.\Qwen3-0.6B`, keeping the weights, configuration, and tokenizer files together. Pass the folder, not an individual `.safetensors` file:

```powershell
winml perf -m .\Qwen3-0.6B --backend cgc -o .\checkpoint-perf.json
```

**GGUF:** Download [Qwen3-0.6B-Q4_0.gguf](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q4_0.gguf) into the current directory, then pass the file:

```powershell
winml perf -m .\Qwen3-0.6B-Q4_0.gguf --backend cgc -o .\gguf-perf.json
```

Both commands select the CGC backend, report time to first token and generation throughput, and save the results as JSON.

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

These fields show the average Task time to first token, subsequent generation speed, and total request time, excluding warmup requests.




## Tutorials and samples

- [ResNet-50 - PyTorch to CGIR (experimental)](../samples/resnet50-cgir.md) walks through the complete Hugging Face PyTorch → ONNX → CGIR build, performance, and evaluation workflow.
- [YOLO11 - ONNX to CGIR (experimental)](../samples/yolo11-cgir.md) provides an end-to-end conversion journey: export a checkpoint to ONNX, build CGIR, collect GPU performance results, compare raw output tensors, and try the WinMLCG EP alternative.

## Command references

- [`winml export`](../commands/export.md) documents `--target cgir` and CGIR export options.
- [`winml optimize`](../commands/optimize.md) documents ONNX optimization.
- [`winml perf`](../commands/perf.md) documents `.mlir`, `winml-runtime`, and WinMLCG performance measurement.
- [`winml eval`](../commands/eval.md) documents evaluation of supported CGIR models with compatible preprocessing metadata.
- [`winml config`](../commands/config.md) and [`winml build`](../commands/build.md) document configuration-driven CGIR conversion.