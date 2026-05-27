# End-to-End Tour

This page walks the full winml-cli pipeline using `--device auto`. The CLI
resolves to the best available device on your machine — NPU first, then GPU,
then CPU — so the four commands below are identical regardless of whether you
have a Copilot+ PC with a Qualcomm NPU, a DirectML-capable GPU, or a plain
laptop with no accelerator at all. You do not need to think about device flags
after Step 0.

The vehicle for this tour is `facebook/convnext-tiny-224`, a compact image
classifier whose operator mix exercises every stage of the pipeline: export,
optimize, quantize, and compile. Estimated time is 15–25 minutes, most of
which is the Hugging Face model download and the compile stage. At the end you
will have a compiled ONNX artifact targeted at your hardware and a real latency
reading from that device.

## Prerequisites

- Windows 11 24H2 (required for NPU; earlier versions work for CPU/GPU)
- winml-cli installed (see [Installation](installation.md))

!!! note "NPU users only"
    To target the Qualcomm NPU you also need:

    - A Qualcomm Snapdragon X device
    - QAIRT SDK installed; `QNN_SDK_ROOT` env var pointing at it
    - `--extra qnn` installed (Python 3.11+)

    Everything else on this page works without these.

## Step 0: See what your machine has

```bash
uv run winml sys --list-device --list-ep
```

This lists every hardware device detected and the execution providers (EPs)
that can target each one. When you pass `--device auto` in the steps below,
winml-cli resolves that to the highest-priority device shown here: NPU first,
then GPU, then CPU.

=== "Copilot+ PC (NPU available)"

    ```text
    Available Devices (priority order)
      #1  NPU   Qualcomm(R) AI Accelerator
                 Driver: 31.0.0.6978 | Manufacturer: Qualcomm Technologies, Inc.
      #2  GPU   NVIDIA GeForce RTX 4060 Laptop GPU
                 Driver: 31.0.15.5107 | Manufacturer: NVIDIA
      #3  CPU   Snapdragon X Elite - X1E-80-100 - Oryon
                 Cores: 12 | Threads: 12 | Architecture: ARM64

    Available Execution Providers
      QNNExecutionProvider              -> NPU/GPU
      DmlExecutionProvider              -> GPU
      CPUExecutionProvider              -> CPU
    ```

=== "Regular Windows laptop (no NPU)"

    ```text
    Available Devices (priority order)
      #1  GPU   Intel(R) Iris(R) Xe Graphics
                 Driver: 31.0.101.5382 | Manufacturer: Intel Corporation
      #2  CPU   12th Gen Intel(R) Core(TM) i7-1260P
                 Cores: 12 | Threads: 16 | Architecture: x86_64

    Available Execution Providers
      DmlExecutionProvider              -> GPU
      CPUExecutionProvider              -> CPU
    ```

## Step 1: Generate the build config

```bash
uv run winml config -m facebook/convnext-tiny-224 --device auto -o convnext_config.json
```

`winml config` queries Hugging Face, auto-detects the task and model type, and
produces a `WinMLBuildConfig` JSON. Passing `--device auto` tells the config
generator to resolve the target device at generation time — it inspects your
hardware and writes the winning device (NPU, GPU, or CPU) together with
matching precision and compile settings into `convnext_config.json`. You can
open the file to see exactly what was picked before committing to a full build.

For a field-by-field explanation of every section in the generated JSON and how
the `quant` and `compile` blocks interact, see
[Config and build](../concepts/config-and-build.md).

## Step 2: Run the build

```bash
uv run winml build -c convnext_config.json -m facebook/convnext-tiny-224 -o convnext_out/
```

This single command runs all four pipeline stages in sequence — export,
optimize, quantize, and compile — reading the device and precision settings
recorded in `convnext_config.json`. The compile stage targets whichever device
the config captured: it calls the QNN backend and embeds a pre-compiled Hexagon
binary on NPU, or it compiles a DirectML graph on GPU, or it produces a
standard optimized ONNX for CPU. All intermediate artifacts land in
`convnext_out/`, so you can inspect or reuse any stage independently.

You can also pass `--no-quant` or `--no-compile` to stop the pipeline early,
or `--rebuild` to force re-running even when cached artifacts exist. For a
deeper look at how each stage works, see
[Concepts → How winml-cli works](../concepts/how-it-works.md) and
[Config and Build](../concepts/config-and-build.md).

!!! warning "NPU users"
    `winml build` reads `QNN_SDK_ROOT` from the environment. Make sure it
    points at your QAIRT SDK before this step, or the compile stage will fail
    with *"QAIRT SDK path not found"*.

## Step 3: Benchmark on your device

```bash
uv run winml perf -m convnext_out/<artifact>.onnx --device auto --iterations 50 --monitor
```

Replace `<artifact>` with the filename written to `convnext_out/` by the build.
For NPU builds the compiled artifact is named `model.onnx` in the output
directory (the `_npu_ctx.onnx` suffix applies only when the compile stage
produces an EPContext file, which requires `enable_ep_context=True` in the
compile config). You can check the directory listing or read the compiled
artifact path from the build output to get the exact name.

=== "NPU (QNN)"

    ```text
    Device:      npu
    Precision:   auto
    Task:        image-classification
    Iterations:  50 (+ 10 warmup)
    Batch Size:  1

    Latency (ms)
      Avg    P50    P90    P95    P99    Min    Max    Std
     3.87   3.82   4.21   4.38   4.71   3.51   5.04   0.21

    Throughput: 258.14 samples/sec

    Results saved to: model_perf.json
    ```

=== "GPU (DirectML)"

    ```text
    Device:      gpu
    Precision:   auto
    Task:        image-classification
    Iterations:  50 (+ 10 warmup)
    Batch Size:  1

    Latency (ms)
      Avg    P50    P90    P95    P99    Min    Max    Std
    12.43  12.18  13.74  14.11  15.02  11.27  16.55   0.89

    Throughput: 80.45 samples/sec
    ```

=== "CPU"

    ```text
    Device:      cpu
    Precision:   auto
    Task:        image-classification
    Iterations:  50 (+ 10 warmup)
    Batch Size:  1

    Latency (ms)
      Avg    P50    P90    P95    P99    Min    Max    Std
    48.31  47.85  52.14  53.77  57.40  44.62  61.23   2.94

    Throughput: 20.70 samples/sec
    ```

The `--monitor` flag opens a live chart of device utilization while the
benchmark runs — most meaningful on NPU or GPU where it confirms the workload
actually hit the accelerator rather than falling back to CPU. After the run
finishes, a JSON file named `{model_slug}_perf.json` is written to the current
directory; you can load it programmatically to compare results across runs or
across machines.

## Cross-device comparison

Each artifact produced by `winml build` is compiled for the specific device
recorded in the config — a QNN EPContext binary will not execute on DirectML,
and vice versa. If you want to measure NPU vs. GPU vs. CPU latency on the same
model and the same machine you need to generate a separate config and artifact
for each EP. The
[ConvNeXt — Primitives Walkthrough](../samples/convnext-primitives.md) sample
does exactly that: it builds a separate compiled artifact for each execution
provider and benchmarks them side by side so you can compare the numbers
directly.

## Where to go next

- [ConvNeXt on NPU tutorial](../tutorials/npu-convnext.md) — full primitives
  walkthrough plus the `winml build` one-shot wrapper, going deeper than this
  page on NPU-specific tuning
- [ConvNeXt — Primitives Walkthrough sample](../samples/convnext-primitives.md)
  — CPU/GPU/NPU comparison on the same model built with explicit per-device
  configs
- [Concepts → How winml-cli works](../concepts/how-it-works.md) — what each
  stage of the build pipeline does and how they chain together
