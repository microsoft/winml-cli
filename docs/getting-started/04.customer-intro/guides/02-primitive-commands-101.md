# ModelKit 101 — Build Your First Model with Primitive Commands

## What Are Primitive Commands?

Primitive commands are the individual building blocks of ModelKit. Each one handles a single stage of the model-building pipeline: inspect, export, analyze, optimize, quantize, or compile. You can run them standalone, reorder them, or compose them into custom workflows.

Think of this as the **coding phase** — you are iterating, experimenting, and debugging your way to a production-ready model.

## The Model: ConvNeXT

In this guide you will build **ConvNeXT** (`facebook/convnext-base-224`) — a family of CNN models inspired by Vision Transformers, introduced by Meta in 2022. ConvNeXT adopts several design choices from Transformers and offers high accuracy while retaining the efficiency of CNNs. It is widely adopted for image classification, detection, and segmentation.

## The BYOM Workflow

The Build Your Own Model (BYOM) workflow takes a source model through a straightforward pipeline. Along the way, three commands serve as **quality gates**:

| Quality Gate | Command | What It Checks |
|---|---|---|
| Portability | `analyze` | Can your model run on the target EP? |
| Performance | `optimize` | Is the graph structured for efficient inference? |
| Fidelity | `quantize` | Does compression preserve acceptable accuracy? |

These three steps define the quality of your output. ModelKit gives you full control over each one.

## Step by Step

### 1. Inspect the Model

Start by inspecting the model metadata. This tells you the task, model class, and I/O shapes — no weights loaded, just metadata.

```bash
wmk inspect -m facebook/convnext-base-224
```

### 2. Export to ONNX

Export the model from PyTorch to ONNX format.

```bash
wmk export -m facebook/convnext-base-224 -o convnext/model.onnx -v
```

### 3. Analyze for EP Compatibility

Run the analyzer to check every operator against the target EPs. It tells you what is supported, what is partial, and what needs fixing. It also generates an optimization config automatically.

```bash
wmk analyze -m convnext/model.onnx --optim-config optim.json
```

### 4. Optimize the Graph

Apply the optimizer using the config the analyzer generated. The analyzer told you what to fix; the optimizer fixes it.

```bash
wmk optimize -m convnext/model.onnx -c optim.json -o convnext/model_opt.onnx
```

### 5. Quantize to INT8

Compress the optimized model to INT8. After this step you have a portable model that can run on any ONNX Runtime backend.

```bash
wmk quantize -m convnext/model_opt.onnx -o convnext/model_opt_int8.onnx
```

### 6. Compile for NPU

Compile the quantized model for QNN — this generates device-specific binaries for the NPU.

```bash
wmk compile -m convnext/model_opt_int8.onnx --ep qnn -o convnext/model_compiled.onnx
```

### 7. Benchmark on NPU

Run the benchmark on NPU. Take note of the latency number.

```bash
wmk perf -m convnext/model_compiled.onnx --ep qnn --iterations 100
```

### 8. Benchmark on CPU for Comparison

Run the same optimized model on CPU. You should see roughly a **25x speedup** — the quantized model on NPU versus the original on CPU. Same model, same accuracy, completely different performance.

```bash
wmk perf -m convnext/model_opt.onnx --ep cpu --iterations 100
```

## Analyzer Deep Dive

The analyzer is the key to building portable ONNX models. It is made of two parts:

**Linter** — Like ESLint, but for ONNX. It checks operator compatibility and classifies each operator: green for supported, gray for partial, red for unsupported.

**AutoConf** — Detects suboptimal patterns in the graph and generates the optimization config automatically.

Together they form the **analyze-optimize loop**: the linter finds the issues, AutoConf writes the fix config, and the optimizer applies it.

## Three-Phase Recap

In this guide you used primitive commands to bring ConvNeXT to Windows ML in three phases:

1. **Inspect** — Understand the model (task, class, I/O shapes)
2. **Build portable ONNX** — Export, analyze, optimize, quantize
3. **Benchmark on device** — Compile, perf on NPU vs CPU

This workflow gives you full control. You can jump into any stage, try different settings, fix errors, or tweak performance.

## Next Steps

You have seen how to build a model step by step with primitive commands. Next, try the config-driven pipeline to achieve the same result with just two commands. See [Two Commands to Production](03-config-build-pipeline.md).
