# Bring Your Own ONNX Model

This tutorial walks you through the complete workflow for optimizing, analyzing, and deploying an ONNX model you already have — whether you exported it yourself (`torch.onnx.export`, ONNX Runtime tools), received it from a teammate, or downloaded it from the ONNX Model Zoo.

Unlike the [Hugging Face Model to NPU](npu-convnext.md) tutorial which starts from a HuggingFace model ID, this tutorial assumes you already have a `.onnx` file on disk and want to make it run faster on your target hardware.

The tutorial is split into two sections. Section A walks through the analyze → optimize → re-analyze loop using primitive commands, teaching you how the optimization feedback cycle works. Section B shows how `winml build` automates that same loop in a single command, optionally targeting NPU with quantization.

---

## Prerequisites

- **Windows 11 24H2** — required for NPU stack support
- **Python 3.11** and **uv** installed (`pip install uv` or follow [astral.sh/uv](https://astral.sh/uv))
- **winml-cli** installed — see [Installation](../getting-started/installation.md)
- **An ONNX model file** — this tutorial uses `my_model.onnx` as a placeholder; substitute your own file

> No NPU? Set `--device cpu` wherever you see `--device npu`. Every other flag stays the same.

---

## Section A — Primitive commands

Working through the primitive commands one at a time reveals how the analyze–optimize feedback cycle works. Each command accepts the output of the previous step as input, and every intermediate artifact is available for inspection.

### Step 1: Analyze the original model

Before any optimization, run the static analyzer to understand your model's EP compatibility and get optimization recommendations:

```bash
uv run winml analyze --model my_model.onnx --optim-config optim_config.json
```

The analyzer classifies every operator in the graph as **supported**, **partial**, **unsupported**, or **unknown** for each available EP. It also detects fusible subgraph patterns and writes the recommended optimization flags to `optim_config.json`.

To target a specific EP:

```bash
uv run winml analyze --model my_model.onnx --ep qnn --device npu --optim-config optim_config.json
```

The output shows per-EP compatibility results:

```text
══════════════════════════════════════════════════════════════════════════
 ANALYSIS SUMMARY
══════════════════════════════════════════════════════════════════════════
   QNNExecutionProvider (NPU): 122/0/0/0
      Ready to deploy
```

If the analyzer detects fusible patterns (GeLU, LayerNorm, etc.), they will appear in the output and the `optim_config.json` will contain the recommended fusion settings. If no patterns are detected (as with simple architectures like ResNet), the config will be empty `{}`.

!!! note "What we just did"
    The analyzer performs static analysis — no runtime or hardware required. It tells you two things: (1) can the model run on your target EP at all, and (2) are there graph patterns that the optimizer can fuse to improve performance. The `--optim-config` flag outputs a JSON file with the exact optimization settings the optimizer needs. S/P/U/Unk = Supported/Partial/Unsupported/Unknown.

---

### Step 2: Optimize the graph

Pass the analyzer's output config directly to the optimizer:

```bash
uv run winml optimize -m my_model.onnx -c optim_config.json -o my_model_optimized.onnx
```

The optimizer applies the fusions specified in the config and reports how many nodes it reduced:

```text
Input: my_model.onnx
Output: my_model_optimized.onnx

Success! Model optimized: my_model_optimized.onnx
Nodes: 122 -> 122 (0.0% reduction)
```

!!! tip
    The node reduction depends on your model's architecture. Simple models like ResNet (only Conv, Relu, Add) have no fusible patterns. Transformer-based models (BERT, ViT) typically see 10–30% node reduction from GeLU, LayerNorm, and Attention fusions.

!!! note "What we just did"
    Graph optimization fuses multi-node patterns (like the 5-node GeLU/Erf sequence) into single high-level operators that EPs can execute more efficiently. The optimizer is purely a graph transformation — it doesn't change the model's numerical behavior or require calibration data. Running it before quantization is important: calibration should be performed on the already-fused topology, not the verbose original graph.

---

### Step 3: Re-analyze the optimized model

Run the analyzer again on the optimized output to confirm that the fusions resolved and no new issues appeared:

```bash
uv run winml analyze --model my_model_optimized.onnx --ep qnn --device npu
```

If the original analysis found fusible patterns that were optimized away, this run should show zero detected patterns and the same or better EP compatibility score.

!!! note "What we just did"
    The analyze → optimize → re-analyze cycle is the fundamental feedback loop in winml-cli. In Section B you'll see that `winml build` automates this loop — it calls the analyzer, applies recommendations, re-analyzes, and repeats until convergence (typically 1–3 iterations). Doing it manually here teaches you what the automation is actually doing under the hood.

---

### Step 4 (optional): Quantize

Insert QDQ (Quantize-Dequantize) nodes into the optimized graph using static calibration:

```bash
uv run winml quantize -m my_model_optimized.onnx -o my_model_int8.onnx --precision int8 --samples 32
```

The quantizer generates 32 random calibration samples, runs them through the model to collect activation statistics, and uses those statistics to set the quantization scale and zero-point for each tensor.

!!! note "What we just did"
    `--precision int8` sets both weights and activations to 8-bit integers, which is the precision most NPU compilers expect. The output model still contains standard `QuantizeLinear` and `DequantizeLinear` ONNX nodes, so it is portable and can run on any ONNX Runtime backend. See [Concepts → Quantization and QDQ](../concepts/quantization.md) for calibration methods and per-channel options.

---

### Step 5 (optional): Compile for the target EP

Compilation converts the portable quantized ONNX into an EP-specific binary format that the execution provider can load directly, skipping JIT compilation at inference time:

=== "Qualcomm NPU"

    ```bash
    uv run winml compile -m my_model_int8.onnx --device npu --ep qnn
    ```

=== "Intel NPU"

    ```bash
    uv run winml compile -m my_model_int8.onnx --device npu --ep openvino
    ```

=== "AMD NPU"

    ```bash
    uv run winml compile -m my_model_int8.onnx --device npu --ep vitisai
    ```

=== "CPU"

    ```bash
    uv run winml compile -m my_model_int8.onnx --device cpu
    ```

!!! note "What we just did"
    Compilation embeds EP context — the compiled binary — inside or alongside the ONNX file using the `EPContext` node convention. At inference time the runtime loads the pre-compiled binary directly rather than re-compiling from the ONNX graph. See [Concepts → Compile and EPContext](../concepts/compile-and-epcontext.md) for details.

---

### Step 6: Benchmark

Measure the performance of your model:

=== "Optimized (CPU)"

    ```bash
    uv run winml perf -m my_model_optimized.onnx --device cpu --warmup 5 --iterations 50
    ```

=== "Compiled (NPU)"

    ```bash
    uv run winml perf -m my_model_int8_npu_ctx.onnx --device npu --iterations 50 --monitor
    ```

!!! note "What we just did"
    `winml perf` generates random inputs matching the model's I/O spec, runs warmup iterations (excluded from statistics), then the benchmark iterations, and reports full latency percentiles alongside throughput. The `--monitor` flag activates live hardware utilization polling. See [Concepts → Perf and monitoring](../concepts/perf-and-monitoring.md) for details.

---

## Section B — One-shot with `winml build`

Once you understand the analyze → optimize → re-analyze loop (which you now do), you can let `winml build` handle everything in one command. When you pass a `.onnx` file, winml-cli auto-detects it and skips the export stage — running the optimization loop, quantization, and compilation automatically.

```bash
uv run winml build -m my_model.onnx -o output/ --device npu --precision int8
```

!!! tip "Config file is optional"
    The `-c config.json` flag is optional. Without it, `winml build` auto-generates an internal config from the flags you pass (like `--device` and `--precision`). If you need a reusable config, generate one with [`winml config`](../commands/config.md):

    ```bash
    uv run winml config --onnx my_model.onnx -d npu --precision int8 -o config.json
    uv run winml build -m my_model.onnx -c config.json -o output/
    ```

The pipeline runs: **analyze → optimize → (re-analyze → re-optimize if needed) → quantize → compile → model.onnx**. The output directory looks like:

```text
output/
├── model.onnx                     ← FINAL: deploy this
├── my_model.onnx                  ← Copy of your input
├── my_model_optimized.onnx        ← After optimization loop converged
├── my_model_quantized.onnx        ← After INT8 quantization
├── my_model_compiled.onnx         ← After EP compilation
├── winml_build_config.json        ← Config used (including auto-detected options)
└── analyze_result.json            ← Analysis from optimize stage
```

You can selectively skip stages using the override flags:

- `--no-optimize` — skip graph optimization (rarely needed; useful if you have a pre-optimized ONNX)
- `--no-quant` — skip quantization (produces a floating-point compiled model)
- `--no-compile` — skip compilation (produces a quantized but not device-locked ONNX)

For example, to produce an optimized model without quantization or compilation:

```bash
uv run winml build -m my_model.onnx -o output/ --device cpu
```

!!! note "What we just did"
    `winml build` is the production workflow. It guarantees that stages run in the correct order, passes intermediate artifacts through the pipeline automatically, and records which stages completed or were skipped in the result summary.

Once the build completes, benchmark the final artifact:

```bash
uv run winml perf -m output/model.onnx --device npu --iterations 50 --monitor
```

---

## Using the Python API

```python
from winml.modelkit import WinMLAutoModel

# Load from a pre-built ONNX (skips the build pipeline)
model = WinMLAutoModel.from_onnx(
    "output/model.onnx",
    task="image-classification",  # set your task
    skip_build=True,
)

output = model(pixel_values=your_input_tensor)
```

Or trigger the full build programmatically:

```python
from winml.modelkit.build import build_onnx_model
from winml.modelkit.config import generate_build_config

config = generate_build_config(onnx_path="my_model.onnx", device="npu", precision="int8")
result = build_onnx_model("my_model.onnx", config=config, output_dir="output/")
print(f"Final model: {result.final_onnx_path}")
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ONNX file not found" | Use an absolute path or ensure the file is in the current directory |
| Analyzer reports unsupported ops | Check if an optimization fusion resolves them; if not, the model needs modification for that EP |
| Optimization loop doesn't converge | The default max is 3 iterations; if patterns persist, they may not be fusible — use `--no-quant --no-compile` and inspect |
| Quantization accuracy regression | Try `--precision int16`, `--per-channel`, or increase `--samples` for better calibration |
| EP compilation fails | Check the selected EP, model compatibility, and target device availability |
| Model too large for memory | Use `--no-compile` and compile on the target device |

---

## Where to go next

- [Hugging Face Model to NPU](npu-convnext.md) — the same pipeline starting from HuggingFace (includes export stage)
- [Output Layout](../reference/output-layout.md) — what each output file contains and the `analyze_result.json` schema
- [Concepts → Analyze and optimize](../concepts/analyze-and-optimize.md) — how the convergence loop works internally
- [Build Config Schema](../reference/index.md) — customize quantization, compilation, and optimization settings
