# ResNet-50 - PyTorch to CGIR (experimental)

This tutorial starts with the
[`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50) PyTorch
model on Hugging Face, exports it to ONNX, converts it to a reusable CGIR
`.mlir` artifact, and measures it through the Windows ML Runtime API.

You will use a generated build configuration so the complete PyTorch → ONNX →
CGIR workflow can be reviewed, repeated, and checked into source control.

## Prerequisites

- A Windows device with a GPU.
- An activated Python 3.11 environment with winml-cli and the experimental
    wheels from [CGC environment setup](../getting-started/cgc-onboarding.md#environment-setup).
- A network connection to download the model and evaluation dataset.

## Step 1: Check the CGC environment

Inspect the devices and execution providers available on your machine:

```powershell
winml sys --list-device --list-ep
```

Verify that the WinMLCG EP and target GPU are listed. Discovery alone does
not prove that the driver supports CGC execution; the benchmark below checks
that path on your device.

## Step 2: Generate a CGC build config

Generate a reusable configuration for the Hugging Face model:

```powershell
winml config -m microsoft/resnet-50 --backend cgc -o .\resnet50-cgir\config.json
```

`--backend cgc` configures the pipeline to export the PyTorch model to ONNX,
prepare it for CGC, and convert the final ONNX model to CGIR. The generated
configuration uses FP16 conversion and does not run a separate EP compilation
stage.

Open `resnet50-cgir\config.json` if you want to review the detected
image-classification task, model loader, ONNX export settings, and
`convert.target: "cgir"` before building.

## Step 3: Build ONNX and CGIR artifacts

Run the configured build:

```powershell
winml build -m microsoft/resnet-50 -c .\resnet50-cgir\config.json -o .\resnet50-cgir\build
```

The first run downloads the model from Hugging Face. The build exports the
PyTorch model to ONNX, applies the configured transformations, and writes the
CGIR model. The primary artifacts are:

```text
resnet50-cgir/
├── config.json
└── build/
    ├── model.onnx
    └── model.mlir
```

The build directory can contain additional reports and intermediate files.
Keep any weight sidecars beside `model.mlir` when moving the CGIR model.

### How the pipeline works

`winml config` does not transform the model. It detects the Hugging Face task,
model class, and ONNX export settings, then writes the pipeline configuration
consumed by `winml build`. For `--backend cgc`, that configuration enables FP16
preparation, disables EP compilation, and adds `convert.target: "cgir"`.

`winml build` combines these primitive stages:

1. `winml export` converts the PyTorch model to ONNX.
2. `winml optimize` rewrites ONNX operators to better align with the current
   CGC IR.
3. `winml quantize` converts the model from FP32 to FP16.
4. The CGIR export stage converts the optimized ONNX model to `.mlir`.

FP16 conversion can affect model accuracy. If accuracy drops, rebuild with
`--no-quant` and compare the results.

!!! important
    The `.mlir` artifact cannot be used directly by ONNX Runtime. Load it
    through the Windows ML Runtime API for CGC inference.

## Step 4: Measure CGIR performance

Run the CGIR artifact through the Windows ML Runtime API and save the
performance results:

```powershell
winml perf -m .\resnet50-cgir\build\model.mlir --runtime winml-runtime --output .\resnet50-cgir\perf.json
```

The report includes warm-up and timed iteration counts, latency percentiles,
throughput, and the resolved device. For `.mlir` input, `winml perf`
automatically selects `winml-runtime`; the explicit option documents the
intended runtime path.

## Step 5: Evaluate model accuracy

First measure the source PyTorch model on a fixed sample from mini-ImageNet:

```powershell
winml eval -m microsoft/resnet-50 --runtime pytorch --device cpu --dataset timm/mini-imagenet --split test --samples 100 --no-shuffle -o .\resnet50-cgir\pytorch-eval.json
```

Evaluate the CGIR artifact on the same samples. `--model-id` provides the
Hugging Face image processor and label configuration associated with the local
`.mlir` model:

```powershell
winml eval -m .\resnet50-cgir\build\model.mlir --model-id microsoft/resnet-50 --runtime winml-runtime --dataset timm/mini-imagenet --split test --samples 100 --no-shuffle -o .\resnet50-cgir\cgir-eval.json
```

Compare the accuracy in `pytorch-eval.json` and `cgir-eval.json`. Both commands
use the same ordered 100-sample slice, so a difference reflects the exported
runtime path rather than a different random sample.

## What you built

- `config.json`: the repeatable CGC build configuration.
- `build\model.onnx`: the optimized, FP16-prepared ONNX model used for CGIR conversion.
- `build\model.mlir`: the reusable CGIR model for Windows ML Runtime.
- `perf.json`: GPU latency and throughput results.
- `pytorch-eval.json` and `cgir-eval.json`: source and CGIR accuracy results.

## See also

- [CGC Onboarding (experimental)](../getting-started/cgc-onboarding.md)
- [winml config](../commands/config.md)
- [winml build](../commands/build.md)
- [winml perf](../commands/perf.md)
- [winml eval](../commands/eval.md)