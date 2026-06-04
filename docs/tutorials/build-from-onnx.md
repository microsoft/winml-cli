# Build from Your Own ONNX File

This tutorial shows how to use winml-cli to optimize, quantize, and compile a
model you've already exported to ONNX — without going through HuggingFace or
PyTorch export.

Use this workflow when:

- You exported a model yourself (via `torch.onnx.export`, ONNX Runtime tools, etc.)
- You received an `.onnx` file from a teammate or vendor
- You want to optimize a model from the ONNX Model Zoo

---

## Prerequisites

- **winml-cli** installed — see [Installation](../getting-started/installation.md)
- An ONNX model file (e.g., `my_model.onnx`)
- (Optional) QNN SDK for NPU compilation, or OpenVINO for Intel NPU

---

## Step 1: Analyze your ONNX file

Before building, run the static analyzer to understand EP compatibility:

```bash
uv run winml analyze --model my_model.onnx
```

This reports which operators are supported by each EP, whether the model can run
on NPU/GPU without modification, and which patterns (GeLU, LayerNorm, etc.) were
detected.

To save the analysis as JSON:

```bash
uv run winml analyze --model my_model.onnx --output analysis.json
```

---

## Step 2: Build with `winml build`

Pass your `.onnx` file directly as the model argument. winml-cli auto-detects
that it's a local ONNX file and skips the export stage:

```bash
uv run winml build -m my_model.onnx -d cpu -o output/
```

This runs: **optimize → quantize → compile → model.onnx**.

### Target a specific device

=== "CPU (default)"

    ```bash
    uv run winml build -m my_model.onnx -d cpu -o output/
    ```

=== "NPU (QNN)"

    ```bash
    uv run winml build -m my_model.onnx -d npu -o output/
    ```

=== "GPU (DirectML)"

    ```bash
    uv run winml build -m my_model.onnx -d gpu --ep dml -o output/
    ```

### Skip stages

If you only want optimization (no quantization or compilation):

```bash
uv run winml build -m my_model.onnx -d cpu --no-quant --no-compile -o output/
```

---

## Step 3: Inspect the output

After the build, your output directory looks like:

```text
output/
├── model.onnx                     ← Deploy this
├── my_model.onnx                  ← Copy of your input
├── my_model_optimized.onnx        ← After graph optimization
├── my_model_quantized.onnx        ← After quantization (if enabled)
├── my_model_compiled.onnx         ← After EP compilation (if enabled)
├── winml_build_config.json        ← Build config used
└── analyze_result.json            ← EP compatibility analysis
```

!!! tip
    If your model is ≥ 100 MiB, each `.onnx` file will have a companion
    `.onnx.data` file containing the external weights.

---

## Step 4: Benchmark

Run the performance benchmark against the final artifact:

```bash
uv run winml perf -m output/model.onnx -d cpu --warmup 5 --iterations 100
```

For NPU:

```bash
uv run winml perf -m output/model.onnx -d npu --warmup 5 --iterations 100
```

---

## Step 5: Run inference (Python API)

```python
from winml.modelkit import WinMLAutoModel

model = WinMLAutoModel.from_onnx(
    "output/model.onnx",
    task="image-classification",  # set your task
    skip_build=True,              # already built, just load
)

output = model(pixel_values=your_input_tensor)
```

---

## Using a config file

For finer control, generate a config first and customize it:

```bash
uv run winml config --onnx my_model.onnx -d npu -o config.json
```

Edit `config.json` (adjust quantization parameters, compilation options, etc.),
then build with it:

```bash
uv run winml build -m my_model.onnx -c config.json -o output/
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ONNX file not found" | Use an absolute path or ensure the file is in the current directory |
| Quantization fails | Try `--no-quant` first to confirm optimize + compile works, then investigate calibration |
| EP compilation fails | Ensure the target EP SDK is installed (`QNN_SDK_ROOT` for QNN, OpenVINO runtime for Intel) |
| Model too large for memory | Use `--no-compile` and compile on the target device |

---

## See also

- [Output Layout](../reference/output-layout.md) — what each output file contains
- [Build Config Schema](../reference/index.md) — customize the build config
- [ConvNeXt on NPU](npu-convnext.md) — full tutorial starting from HuggingFace
