# ConvNeXt on NPU

!!! info "Pick the right ConvNeXt page"
    Three pages use ConvNeXt as their vehicle, each with a different teaching purpose:

    - **This tutorial** — the canonical deep-dive: full pipeline with both QNN and OpenVINO NPU backends, plus the `winml build` one-shot. Start here if you want to ship to NPU.
    - **[ConvNeXt — Primitives Walkthrough](../samples/convnext-primitives.md)** — a CPU vs GPU vs NPU comparison using the primitive commands. Start here if you want to compare devices on the same model.
    - **[End-to-End Tour](../getting-started/quickstart.md)** — the short Getting Started introduction. Start here for a 15-minute taste.

This tutorial walks you through the complete journey from a pretrained Hugging Face model — `facebook/convnext-tiny-224` — to a quantized, compiled artifact running on an NPU. By the end you will have benchmarked the model on your device and measured real inference latency. Nothing is skipped, and every command produces a file you can inspect or reuse.

The primary hardware target is a Copilot+PC with a Snapdragon X-class NPU (40+ TOPS). If you do not have an NPU, every step works on CPU or DirectML as a fallback — the only thing that changes is the `--device` and `--ep` flags on the compile and perf commands. Those variations are shown explicitly in the tabbed blocks below.

The tutorial is split into two sections. Section A runs through eight primitive commands — one per pipeline stage — so you understand what each stage does, what artifact it produces, and why it matters. Section B shows you that `winml build` runs the same pipeline in a single command once you have a config file. Most production workflows live in Section B; Section A is how you learn to trust it.

---

## Prerequisites

- **Windows 11 24H2** — required for NPU stack support
- **Copilot+PC with NPU** — 40+ TOPS recommended; CPU and DirectML work as fallback throughout
- **Python 3.11** and **uv** installed (`pip install uv` or follow [astral.sh/uv](https://astral.sh/uv))
- **winml-cli** installed — see [Installation](../getting-started/installation.md)

> No NPU? Set `--device cpu` wherever you see `--device npu` and drop `--monitor` from perf commands. Every other flag stays the same.

---

## Section A — Primitive commands

Working through the primitive commands one at a time is the best way to understand what the `winml build` wrapper does under the hood. Each step accepts the output of the previous step as its input, so the chain is explicit and every intermediate artifact is available for inspection.

### Step 1: Inspect the model

Before downloading any weights, confirm that winml-cli knows how to handle `facebook/convnext-tiny-224`.

```bash
uv run winml inspect -m facebook/convnext-tiny-224
```

You should see output similar to the following:

```text
Model               facebook/convnext-tiny-224
Task                image-classification
Model class         ConvNextForImageClassification
Exporter            optimum/onnx
Input               pixel_values: float32 [1, 3, 224, 224]
Output              logits: float32 [1, 1000]
Support status      supported
```

!!! note "What we just did"
    `winml inspect` queries the Hugging Face model card and winml-cli's internal registry without downloading weights. It confirms three things: the auto-detected task (`image-classification`), the model class that will be used for loading, and the exporter that will handle the ONNX conversion. If this command fails, stop here — something about the model is unsupported and proceeding would waste time. A successful inspect is the green light for every stage that follows.

---

### Step 2: Generate a build config

Generate a `WinMLBuildConfig` JSON file for the model. For the primitive workflow this file is optional — you can drive each stage entirely through CLI flags — but generating it now gives you a versioned record of every auto-detected setting, and it is required for Section B.

```bash
uv run winml config -m facebook/convnext-tiny-224 --device npu --precision int8 -o convnext_config.json
```

Open `convnext_config.json` to see what was auto-detected: the task, I/O tensor shapes, quantization parameters, and the compile target. The `--device npu --precision int8` flags tell the config generator to pre-populate the quantization and compile sections for NPU deployment rather than leaving them at defaults.

!!! note "What we just did"
    `winml config` auto-resolves every setting that would otherwise require you to look up flags manually. The resulting JSON is the single source of truth for a reproducible build. You can commit it to version control, share it with teammates, edit a single field to try a different precision, and replay the exact same build on any machine. See [Concepts → Config and build](../concepts/config-and-build.md) for a deeper look at the config schema and how the stages interact.

---

### Step 3: Export to ONNX

Download the pretrained weights and convert the PyTorch model to ONNX format.

```bash
uv run winml export -m facebook/convnext-tiny-224 -o convnext.onnx
```

This runs an eight-stage export pipeline: model preparation, input generation, hierarchy building, ONNX conversion, node tagging, tag injection, and metadata generation. The result is a standards-compliant ONNX file with winml-cli's Hierarchy-preserving Tags Protocol (HTP) metadata embedded in node `metadata_props`. That metadata is what lets downstream tools make architecture-aware optimization decisions without hardcoded model knowledge.

!!! note "What we just did"
    The default export embeds hierarchy tags — a tree of source module names mapped onto ONNX nodes — so that the optimizer and analyzer can reason about the graph in terms of the original model structure rather than flat node lists. If you need a clean ONNX without that metadata (for compatibility with other tools), add `--no-hierarchy`. See [Concepts → Load and export](../concepts/load-and-export.md) for what hierarchy preservation adds and when it matters.

---

### Step 4: Analyze for EP compatibility

Before spending time on optimization and quantization, check that the model's operators are supported by your target execution provider.

```bash
uv run winml analyze -m convnext.onnx --ep qnn --device npu
```

The analyzer performs static analysis — no runtime required — and classifies every operator in the graph as **supported**, **partial**, or **unsupported** for the target EP. It reports a coverage summary, flags any operators that may fall back to CPU, and exits with code 0 for full support or 1 for partial support.

For CPU fallback, run:

```bash
uv run winml analyze -m convnext.onnx --ep cpu --device cpu
```

!!! note "What we just did"
    Knowing your operator coverage before you quantize or compile saves you from discovering EP incompatibilities at the very last step of a long pipeline. ConvNeXt's operators (Conv, GELU, LayerNorm, Add) have broad support across QNN and OpenVINO, so this command should exit 0. If it exits 1, the output tells you which operators are problematic and includes recommendations for resolving them — typically by enabling a graph rewrite in the optimizer that fuses the unsupported pattern into a supported one. See [Concepts → Analyze and optimize](../concepts/analyze-and-optimize.md) for details on the analyzer's recommendation engine.

---

### Step 5: Optimize the graph

Apply graph-level optimizations: operator fusion, constant folding, shape inference, and EP-specific graph rewrites.

```bash
uv run winml optimize -m convnext.onnx -o convnext_optim.onnx
```

The optimizer reports how many nodes it reduced. A typical ConvNeXt-tiny optimization fuses several element-wise sequences and removes redundant reshape operations, cutting the node count noticeably without changing model semantics. If you want to apply a specific preset suited to the Snapdragon NPU, add `--preset qnn-compatible` to disable fusions that QNN does not benefit from.

!!! note "What we just did"
    Graph optimization is a separate stage from quantization so that you can inspect the intermediate graph, compare node counts, and selectively enable or disable individual fusion passes using the `--enable-*` / `--disable-*` flags. Run `uv run winml optimize --list-capabilities` to see every registered optimization flag and its default state. Optimization always happens on the floating-point graph; quantization is applied after so that calibration statistics are computed on the already-fused topology.

---

### Step 6: Quantize

Insert QDQ (Quantize-Dequantize) nodes into the optimized graph using static calibration. This reduces model size and speeds up inference on hardware with integer execution units, which includes Snapdragon NPUs and Intel NPUs.

```bash
uv run winml quantize -m convnext_optim.onnx -o convnext_int8.onnx --precision int8 --samples 32
```

The quantizer generates 32 random calibration samples, runs them through the model to collect activation statistics, and uses those statistics (with the default `minmax` method) to set the quantization scale and zero-point for each tensor. Thirty-two samples is sufficient for a vision model with fixed-size inputs like ConvNeXt. For models with variable-length inputs or complex activation distributions, increase `--samples` to 64 or 128.

!!! note "What we just did"
    `--precision int8` sets both weights and activations to 8-bit integers, which is the precision most NPU compilers expect. The output model still contains standard `QuantizeLinear` and `DequantizeLinear` ONNX nodes, so it is portable and can run on any ONNX Runtime backend — you do not need special tooling to inspect it. See [Concepts → Quantization and QDQ](../concepts/quantization.md) for a detailed explanation of the QDQ node pattern, calibration methods, and how to choose between per-tensor and per-channel quantization.

---

### Step 7: Compile for the target EP

Compilation converts the portable quantized ONNX into an EP-specific binary format that the execution provider can load directly, skipping JIT compilation at inference time. This is the step that produces a device-locked artifact tied to the selected EP.

The examples below use the default compiler backend (`--compiler ort`), which uses ONNX Runtime's built-in EP context compiler:

=== "Qualcomm NPU"

    ```bash
    uv run winml compile -m convnext_int8.onnx --device npu --ep qnn
    ```

=== "Intel NPU"

    ```bash
    uv run winml compile -m convnext_int8.onnx --device npu --ep openvino
    ```

=== "AMD NPU"

    ```bash
    uv run winml compile -m convnext_int8.onnx --device npu --ep vitisai
    ```

=== "CPU"

    ```bash
    uv run winml compile -m convnext_int8.onnx --device cpu
    ```

The compiled output file appears in the same directory as the input model. The file name follows the pattern `convnext_int8_npu_ctx.onnx` (using the resolved device string `npu`, not the EP name) and an accompanying `.bin` context binary is written alongside it (unless `--embed` is passed, which embeds the binary inside the ONNX file). CPU builds do not produce a new artifact — the compile step validates EP compatibility but writes no output file; use `convnext_int8.onnx` directly for CPU inference.

!!! note "What we just did"
    Compilation embeds EP context — the compiled binary — inside or alongside the ONNX file using the `EPContext` node convention. At inference time the runtime loads the pre-compiled binary directly rather than re-compiling from the ONNX graph, eliminating the 15–60 second JIT penalty on first load. The default `--compiler ort` backend bundles compilation within ONNX Runtime itself. See [Concepts → Compile and EPContext](../concepts/compile-and-epcontext.md) for the full picture of what gets embedded and how the context is consumed at runtime.

---

### Step 8: Benchmark

Measure inference latency and throughput with the `--monitor` flag to see live NPU utilization alongside the timing numbers.

=== "QNN NPU"

    ```bash
    uv run winml perf -m convnext_int8_npu_ctx.onnx --device npu --iterations 50 --monitor
    ```

=== "OpenVINO NPU"

    ```bash
    uv run winml perf -m convnext_int8_npu_ctx.onnx --device npu --ep openvino --iterations 50 --monitor
    ```

=== "CPU"

    ```bash
    uv run winml perf -m convnext_int8.onnx --device cpu --iterations 50
    ```

A representative run on a Snapdragon X Elite NPU produces output like the following:

```text
Device:       npu
Task:         image-classification
Iterations:   50 (+ 10 warmup)
Batch Size:   1

Latency (ms)
  Avg    P50    P90    P95    P99    Min    Max    Std
  2.14   2.11   2.31   2.38   2.59   1.98   2.71   0.14

Throughput:  467.29 samples/sec

Hardware (during benchmark)
  NPU: 72.4% avg, 89.1% peak  |  CPU: 3.2% avg
  Sys Mem: 1842 MB  |  Device Mem: 48/12 MB (local/shared)
```

The CPU fallback (same model, `--device cpu`) will typically show latencies 8–15x higher and near-zero NPU utilization. The contrast between those two runs is the best proof that your NPU path is actually being used.

!!! note "What we just did"
    `winml perf` generates random inputs matching the model's I/O spec, runs the configured number of warmup iterations (excluded from statistics), then the benchmark iterations, and reports full latency percentiles alongside throughput. The `--monitor` flag activates live hardware utilization polling at 200 ms intervals, displaying an in-terminal chart and attaching the hardware metrics to the JSON report saved alongside the console output. See [Concepts → Perf and monitoring](../concepts/perf-and-monitoring.md) for how to interpret the utilization numbers and what `hw_monitor` fields look like in the JSON report.

---

### Step 9 (optional): Evaluate accuracy

After quantization it is good practice to verify that INT8 accuracy is close to the FP32 baseline. The `winml eval` command runs the model against a held-out dataset slice and reports task-relevant metrics.

```bash
uv run winml eval -m convnext_int8.onnx --model-id facebook/convnext-tiny-224 --dataset imagenet-1k --split validation --samples 100 --device npu
```

The `--model-id` flag is required when passing an ONNX file, because the evaluator needs it to locate the preprocessor and label mappings. The command downloads 100 shuffled validation samples, runs inference, and reports top-1 and top-5 accuracy. A well-quantized ConvNeXt-tiny should lose less than 0.5 percentage points of top-1 accuracy compared to the floating-point checkpoint.

!!! note "What we just did"
    Accuracy evaluation gives you a principled stopping criterion for quantization decisions. If the accuracy drop is larger than acceptable, return to Step 6 and try `--precision int16` or per-channel quantization (`--per-channel`) instead of the default per-tensor int8. See [Concepts → Eval and datasets](../concepts/eval-and-datasets.md) for the full list of supported datasets, tasks, and column mapping options.

---

## Section B — One-shot with `winml build`

Once you understand what each primitive stage does (which you now do), you can collapse the entire pipeline into a single command. `winml build` orchestrates export, optimize, quantize, and compile in sequence.

```bash
uv run winml build -m facebook/convnext-tiny-224 -o convnext_out/ --device npu --precision int8
```

!!! tip "Config file is optional"
    The `-c config.json` flag is optional. Without it, `winml build` auto-generates an internal config from the flags you pass (like `--device` and `--precision`). If you need a reusable config, generate one with [`winml config`](../commands/config.md).

The command downloads the pretrained weights, runs all four pipeline stages, and writes every intermediate and final artifact into `convnext_out/`. The stage timing is printed as each stage completes, and the final line tells you the path of the compiled model.

You can selectively skip stages using the override flags:

- `--no-optimize` — skip graph optimization (rarely needed; useful if you have a pre-optimized ONNX)
- `--no-quant` — skip quantization (produces a floating-point compiled model)
- `--no-compile` — skip compilation (produces a quantized but not device-locked ONNX)

For example, to produce an optimized and quantized model without the compile step:

```bash
uv run winml build -m facebook/convnext-tiny-224 -o convnext_out/ --device npu --precision int8 --no-compile
```

!!! note "What we just did"
    `winml build` is the production workflow. It guarantees that stages run in the correct order, passes intermediate artifacts through the pipeline automatically, and records which stages completed or were skipped in the result summary.

Once the build completes, benchmark the final artifact from `convnext_out/`:

```bash
uv run winml perf -m convnext_out/model.onnx --device npu --iterations 50 --monitor
```

The result should match what you saw in Step 8, confirming that the `winml build` pipeline produces bit-identical output to the manual primitive chain.

---

## Where to go next

- [Concepts → How winml-cli works](../concepts/how-it-works.md) — the full mental model for the pipeline
- [Concepts → Compile and EPContext](../concepts/compile-and-epcontext.md) — understanding the compiled artifact format
- [Samples → ConvNeXt primitives walkthrough](../samples/convnext-primitives.md) — a side-by-side CPU vs. GPU vs. NPU device comparison using the same model
- [Commands → Overview](../commands/overview.md) — quick reference for every flag on every command

## See also

- [Concepts → Quantization and QDQ](../concepts/quantization.md)
- [Concepts → Analyze and optimize](../concepts/analyze-and-optimize.md)
- [Concepts → Perf and monitoring](../concepts/perf-and-monitoring.md)
- [Concepts → Eval and datasets](../concepts/eval-and-datasets.md)
