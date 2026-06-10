# ConvNeXt — Primitives Walkthrough

!!! info "Pick the right ConvNeXt page"
    - **This sample** — primitives on CPU, GPU (DirectML), and NPU (QNN) side-by-side. Best when you want to compare devices.
    - **[ConvNeXt on NPU](../tutorials/npu-convnext.md)** — the canonical NPU production tutorial with both QNN and OpenVINO, plus the `winml build` one-shot.
    - **[Quickstart](../getting-started/quickstart.md)** — short Getting Started tour.

ConvNeXt Tiny is a compact convolutional image classifier trained on ImageNet-1k. At roughly 28 million parameters it is small enough to export and quantize in minutes on a developer laptop, yet representative enough that the latency and accuracy numbers you observe reflect real-world deployment trade-offs. Its straightforward architecture — no attention mechanisms, no dynamic control flow — makes it an ideal first model for learning the winml-cli pipeline.

This walkthrough drives the full pipeline using the primitive commands directly: `winml inspect`, `winml config`, `winml export`, `winml quantize`, `winml compile`, `winml perf`, and `winml eval`. Running the steps individually rather than through `winml build` exposes what each command does and how its output feeds the next stage. The walkthrough covers three execution providers: CPU, GPU (DirectML), and NPU (Qualcomm QNN).

## Prerequisites

- winml-cli installed and `winml` available on your PATH — see [Installation](../getting-started/installation.md).
- Internet access so HuggingFace Hub can download the model weights on first run.

## Step 1: Inspect the model

Before touching weights, confirm that winml-cli recognises the model and knows which task, loader class, and exporter to use.

```bash
winml inspect -m facebook/convnext-tiny-224
```

```text
+------------------------- facebook/convnext-tiny-224 --------------------------+
| Task          image-classification                                             |
| Model Class   ConvNextForImageClassification                                   |
| Exporter      OptimumExporter                                                  |
| WinML Class   WinMLImageClassificationModel                                    |
| Status        Supported                                                        |
+-------------------------------------------------------------------------------+
```

!!! note "What we just did"
    `winml inspect` fetched only the model's `config.json` from HuggingFace Hub — no weights — and confirmed that `facebook/convnext-tiny-224` maps to a supported task (`image-classification`), a known model class, and a compatible ONNX exporter.

## Step 2: Generate a config (optional)

```bash
winml config -m facebook/convnext-tiny-224 -o convnext_config.json
```

Generating a config file is optional when running the primitives individually, but it is good practice: the JSON captures the auto-detected loader, export, quantization, and compile settings in one reproducible artifact. You can check it into source control, diff it against future versions of the model, or hand-edit individual fields before passing it to `winml build`. For a full description of every field, see [Config and build](../concepts/config-and-build.md).

## Step 3: Export to ONNX

Download the model weights and convert the PyTorch graph to a portable ONNX file.

```bash
winml export -m facebook/convnext-tiny-224 -o convnext.onnx
```

```text
Model: facebook/convnext-tiny-224
Output: convnext.onnx

Starting HTP export...
  Detected task: image-classification

Success! Model exported to: convnext.onnx
```

!!! note "Hierarchy metadata"
    By default `winml export` embeds `hierarchy_tag` metadata in each ONNX node, recording which PyTorch module the node originated from. This lets downstream tools like `winml perf --module` and `winml analyze` reason about operator groups rather than flat graph positions. To skip the metadata and produce a clean ONNX file, add `--no-hierarchy`. For more detail see [Load and export](../concepts/load-and-export.md).

## Step 4: Quantize

Insert QDQ (Quantize/Dequantize) nodes using 32 calibration samples drawn from the task-default dataset.

```bash
winml quantize -m convnext.onnx -o convnext_int8.onnx --precision int8 --samples 32
```

```text
Calibrating: 32 samples [minmax]
Inserting QDQ nodes...
Saved: convnext_int8.onnx
```

!!! note "Calibration"
    Static quantization needs representative inputs to estimate each tensor's value range before baking scale and zero-point constants into the QDQ nodes. The `--samples` flag controls how many calibration inputs are used; 32 is a reasonable starting point for vision classifiers. If you see accuracy regression after quantization, try increasing `--samples` or switching to `--method entropy`. See [Quantization & QDQ](../concepts/quantization.md) for the full trade-off discussion.

## Step 5: Compile for each EP

Compilation pre-bakes an EP-specific binary cache into the ONNX graph so the runtime can skip per-session JIT compilation. The examples below use the default `ort` compiler backend, which uses ONNX Runtime's built-in compiler.

=== "CPU"

    ```bash
    winml compile -m convnext_int8.onnx --output-dir . --device cpu
    ```

=== "GPU"

    ```bash
    winml compile -m convnext_int8.onnx --output-dir . --device gpu
    ```

=== "NPU (ORT, default)"

    ```bash
    winml compile -m convnext_int8.onnx --output-dir . --device npu
    ```

!!! note "NPU compiler backend"
    The default `--compiler ort` backend uses ONNX Runtime's built-in compilation. For a full explanation of how EPs relate to device targets see [ONNX & Execution Providers](../concepts/eps-and-devices.md).

Only the NPU invocation writes a new compiled artifact — `convnext_int8_npu_ctx.onnx` — which contains an EPContext node embedding the pre-compiled binary. CPU and GPU compile with `enable_ep_context=False` by default: the compile step validates the model against the target EP but does not produce a new file. For CPU and GPU perf benchmarks (Step 6), use the quantized `convnext_int8.onnx` directly.

## Step 6: Benchmark

Measure latency and throughput on each device. Pass the compiled ONNX directly so the benchmark uses the pre-compiled artifact.

```bash
winml perf -m convnext_int8.onnx --device cpu --iterations 200
```

```text
Device:      cpu
Precision:   auto
Task:        image-classification
Iterations:  200 (+ 10 warmup)
Batch Size:  1

Latency (ms)
  Avg    P50    P90    P95    P99    Min    Max    Std
 8.41   8.35   9.02   9.31  10.14   7.88  12.63   0.48

Throughput: 118.91 samples/sec
```

```bash
winml perf -m convnext_int8.onnx --device gpu --iterations 200
winml perf -m convnext_int8_npu_ctx.onnx --device npu --iterations 200
```

The NPU variant typically delivers the lowest latency and highest power efficiency on Qualcomm Snapdragon hardware. Use the JSON output written by `--output` to compare runs programmatically.

## Step 7: Evaluate

Measure top-1 accuracy on 100 samples from the ImageNet-1k validation split. When passing an ONNX file, supply `--model-id` so the command knows which preprocessor and label vocabulary to use.

```bash
winml eval -m convnext_int8.onnx --model-id facebook/convnext-tiny-224 \
    --dataset imagenet-1k --split validation --samples 100 --device cpu
```

```text
Task:     image-classification
Dataset:  imagenet-1k (validation, 100 samples)
Device:   cpu

Accuracy: 81.00%

Results saved to: convnext_int8_eval.json
```

To compare quantized accuracy against the floating-point baseline, run the same command with `convnext.onnx` and compare the two JSON outputs.

## What you learned

- `winml inspect` checks task detection and exporter compatibility from the model's `config.json` alone — no weight download needed.
- `winml config` captures the full pipeline configuration as a reproducible JSON file.
- `winml export` converts the PyTorch model to a portable ONNX graph and embeds hierarchy metadata for downstream analysis.
- `winml quantize` inserts QDQ nodes using calibration data; `--precision int8` and `--samples` control the precision and calibration budget.
- `winml compile` pre-bakes an EP-specific binary cache for NPU (producing `convnext_int8_npu_ctx.onnx`); CPU and GPU compile steps validate EP compatibility but produce no new artifact — use the quantized `convnext_int8.onnx` for those devices.
- `winml perf` and `winml eval` consume the final artifact without modifying it — benchmark first, then validate accuracy before shipping.

## See also

- [BERT — Config + Build + Perf](bert-config-build.md) — the same pipeline driven through `winml build` with a config file
- [How winml-cli Works](../concepts/how-it-works.md) — pipeline overview and stage descriptions
- [Quantization & QDQ](../concepts/quantization.md) — calibration methods and accuracy trade-offs
- [ONNX & Execution Providers](../concepts/eps-and-devices.md) — EP selection and device flags
