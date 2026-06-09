# Bring Your Own ONNX Model

This tutorial walks you through the complete workflow for optimizing, analyzing, and deploying an ONNX model you already have — whether you exported it yourself (`torch.onnx.export`, ONNX Runtime tools), received it from a teammate, or downloaded it from the ONNX Model Zoo.

Unlike the [ConvNeXt on NPU](npu-convnext.md) tutorial which starts from a HuggingFace model ID, this tutorial assumes you already have a `.onnx` file on disk and want to make it run faster on your target hardware.

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
📊 OP CHECK
══════════════════════════════════════════════════════════════════════════
   📚 Model: my_model.onnx
   🔺 Opset: 17  Producer: pytorch v2.12.0
   📏 Operators: 122 total, 7 unique types
   🏗️ Analysis targets: QNNExecutionProvider (NPU), QNNExecutionProvider (GPU)
────────────────────────────────────────────────────────────────────────
👻 EP 1: QNNExecutionProvider on NPU
────────────────────────────────────────────────────────────────────────
 Op Type                       S/P/U/Unk
 🃓 Conv (53)                  53/0/0/0
 🃓 Relu (49)                  49/0/0/0
 🃓 Add (16)                   16/0/0/0
 🃓 MaxPool (1)                1/0/0/0
 🃓 GlobalAveragePool (1)      1/0/0/0
 🃓 Flatten (1)                1/0/0/0
 🃓 Gemm (1)                   1/0/0/0
 TOTAL (122)                   122/0/0/0
══════════════════════════════════════════════════════════════════════════
📊 ANALYSIS SUMMARY
══════════════════════════════════════════════════════════════════════════
   🃓 QNNExecutionProvider (NPU): 122/0/0/0
      Ready to deploy
```

If the analyzer detects fusible patterns (GeLU, LayerNorm, etc.), they will appear in the output and the `optim_config.json` will contain the recommended fusion settings. If no patterns are detected (as with simple architectures like ResNet), the config will be empty `{}`.

!!! note "What we just did"
    The analyzer performs static analysis — no runtime or hardware required. It tells you two things: (1) can the model run on your target EP at all, and (2) are there graph patterns that the optimizer can fuse to improve performance. The `--optim-config` flag outputs a JSON file with the exact optimization settings the optimizer needs. S/P/U/Unk = Supported/Partial/Unsupported/Unknown.

---

### Step 2: Optimize with the generated config

Pass the analyzer's output config directly to the optimizer:

```bash
uv run winml optimize -m my_model.onnx -c optim_config.json -o my_model_optimized.onnx
```

The optimizer applies the fusions specified in the config. Output:

```text
Input: my_model.onnx
Output: my_model_optimized.onnx
Loading model...
Running optimizer...
Saving optimized model...

Success! Model optimized: my_model_optimized.onnx
Nodes: 122 -> 122 (0.0% reduction)
```

!!! tip
    The node reduction depends on your model's architecture. Simple models like ResNet (only Conv, Relu, Add) have no fusible patterns. Transformer-based models (BERT, ViT) typically see 10–30% node reduction from GeLU, LayerNorm, and Attention fusions.

To see all available optimization capabilities:

```bash
uv run winml optimize --list-capabilities
```

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

### Step 4: Benchmark the optimized model

Measure the performance improvement from optimization:

```bash
uv run winml perf -m my_model_optimized.onnx --device cpu --warmup 5 --iterations 50
```

For NPU (if you have the compiled model from a later step):

```bash
uv run winml perf -m my_model_optimized.onnx --device npu --warmup 5 --iterations 50
```

---

### Step 5 (optional): Quantize and compile for NPU

If your target is NPU deployment, continue the pipeline with quantization and compilation:

```bash
# Quantize (INT8, QDQ format)
uv run winml quantize -m my_model_optimized.onnx -o my_model_int8.onnx --precision int8 --samples 32

# Compile for NPU
uv run winml compile -m my_model_int8.onnx --device npu
```

Then benchmark the final compiled artifact:

```bash
uv run winml perf -m my_model_int8_npu_ctx.onnx --device npu --iterations 50 --monitor
```

---

## Section B — One-shot with `winml build`

Once you understand the analyze → optimize → re-analyze loop (which you now do), you can let `winml build` handle everything in one command. When you pass a `.onnx` file, winml-cli auto-detects it and skips the export stage — running the optimization loop, quantization, and compilation automatically.

### CPU target (optimize only)

```bash
uv run winml build -m my_model.onnx -d cpu -o output/
```

Since `-d cpu` resolves to fp16 precision (no quantization) and compilation is off by default, this just runs the analyze–optimize convergence loop:

```text
output/
├── model.onnx                     ← Deploy this
├── my_model.onnx                  ← Copy of your input
├── my_model_optimized.onnx        ← After graph optimization
├── winml_build_config.json        ← Auto-generated build config
└── analyze_result.json            ← Final analysis output
```

### NPU target (full pipeline)

To get a quantized, compiled model for NPU in one shot, pass `--compile`:

```bash
uv run winml build -m my_model.onnx -d npu --compile -o output/
```

Or generate a config first for more control:

```bash
uv run winml config --onnx my_model.onnx -d npu --precision int8 -o config.json
uv run winml build -m my_model.onnx -c config.json -o output/
```

The pipeline runs: **analyze → optimize → (re-analyze → re-optimize if needed) → quantize → compile → model.onnx**.

The output directory for a full NPU build looks like:

```text
output/
├── model.onnx                     ← FINAL: compiled NPU artifact
├── my_model.onnx                  ← Copy of your input
├── my_model_optimized.onnx        ← After optimization loop converged
├── my_model_quantized.onnx        ← After INT8 quantization
├── my_model_compiled.onnx         ← After EP compilation
├── winml_build_config.json        ← Config used (including auto-detected options)
└── analyze_result.json            ← Analysis from optimize stage
```

!!! note "What we just did"
    `winml build` with an ONNX input runs the same analyze → optimize → re-analyze convergence loop from Section A, but automatically. It reads the analyzer's recommendations, applies them, re-runs the analyzer, and repeats until no new recommendations appear (max 3 iterations by default). The config file specifies device, precision, and EP — so `--device npu --precision int8` in the config causes quantize and compile stages to run automatically.

### Selectively skip stages

By default when auto-generating config (no `-c` flag):

- **Compilation is OFF** — pass `--compile` to enable it
- **Quantization depends on device**:
    - `-d npu` → quantization ON (w8a16 precision by default)
    - `-d gpu` / `-d cpu` → quantization OFF (fp16, no quantization)

Override flags:

- `--no-quant` — force skip quantization (even on NPU)
- `--compile` — force enable compilation
- `--no-compile` — force skip compilation (default when no config file)

```bash
# NPU: optimize + quantize (w8a16), skip compilation
uv run winml build -m my_model.onnx -d npu -o output/

# NPU: full pipeline including compilation
uv run winml build -m my_model.onnx -d npu --compile -o output/

# NPU: optimize only, no quantize, no compile
uv run winml build -m my_model.onnx -d npu --no-quant -o output/

# CPU/GPU: optimize only (quantize and compile are already off)
uv run winml build -m my_model.onnx -d cpu -o output/
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

- [ConvNeXt on NPU](npu-convnext.md) — the same pipeline starting from HuggingFace (includes export stage)
- [Output Layout](../reference/output-layout.md) — what each output file contains and the `analyze_result.json` schema
- [Concepts → Analyze and optimize](../concepts/analyze-and-optimize.md) — how the convergence loop works internally
- [Build Config Schema](../reference/index.md) — customize quantization, compilation, and optimization settings
