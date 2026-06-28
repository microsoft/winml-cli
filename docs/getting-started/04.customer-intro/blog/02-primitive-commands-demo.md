# ModelKit 101 — Building Models Step by Step with Primitive Commands

## The Coding Phase

Every model goes through a development lifecycle. Before anything is polished, repeatable, or automated, there is a **coding phase** — where developers explore the model, experiment with settings, debug compatibility issues, and iterate until the output looks right.

ModelKit's **primitive commands** are built for exactly this phase. Each command handles a single stage of the pipeline, and developers can run them in any order, skip stages, or repeat them with different parameters. Full control, no magic.

This walkthrough builds a ConvNeXT model from scratch using primitives, one step at a time.

## The Model: ConvNeXT

ConvNeXT is **a family of CNN models inspired by Vision Transformers**, introduced by Meta (Facebook) in 2022. It borrows several design choices from Transformers — such as larger kernel sizes and modernized training recipes — while retaining the efficiency and simplicity of convolutional architectures. The result is a model that delivers high accuracy at competitive speed, which is why ConvNeXT is widely adopted for tasks such as **image classification, object detection, and segmentation**.

For this walkthrough, the specific variant is `facebook/convnext-base-224`.

## The BYOM Workflow

Before diving into commands, it helps to understand the workflow behind ModelKit. Given a source model, the pipeline will export, analyze, optimize, quantize, and evaluate before shipping.

Three commands in this pipeline serve as **quality gates**:

- **Analyze** — checks **portability**: will this model run on the target execution provider?
- **Optimize** — improves **performance**: restructures the graph for faster inference.
- **Quantize** — controls **fidelity**: compresses the model while preserving accuracy.

These three steps define the quality of the output. ModelKit gives developers full control over each one.

## Step-by-Step Walkthrough

### 1. Inspect the Model

The first step is always to understand what you are working with. `wmk inspect` reads model metadata — task, model class, I/O shapes — without loading weights.

```bash
wmk inspect -m facebook/convnext-base-224
```

This tells you everything about the model before any heavy computation begins.

### 2. Export from PyTorch to ONNX

Next, export the model from its PyTorch source into ONNX format.

```bash
wmk export -m facebook/convnext-base-224 -o convnext/model.onnx -v
```

The `-v` flag enables verbose output so you can see exactly what the exporter is doing.

### 3. Analyze for EP Compatibility

With the ONNX model in hand, run the analyzer. It checks every operator against execution providers — reporting what is supported, what is partial, and what needs fixing. It also generates an optimization config automatically.

```bash
wmk analyze -m convnext/model.onnx --optim-config optim.json
```

The analyzer produces `optim.json`, which captures exactly what the optimizer needs to fix. More on the analyzer below.

### 4. Optimize the Graph

Apply the optimizer using the config the analyzer just generated. The analyzer identified what to fix; the optimizer fixes it.

```bash
wmk optimize -m convnext/model.onnx -c optim.json -o convnext/model_opt.onnx
```

### 5. Quantize to INT8

Compress the optimized model to INT8. After this step, the model is portable — it can run on any ONNX Runtime backend.

```bash
wmk quantize -m convnext/model_opt.onnx -o convnext/model_opt_int8.onnx
```

### 6. Compile for NPU

Generate device-specific binaries for the NPU via the QNN execution provider.

```bash
wmk compile -m convnext/model_opt_int8.onnx --ep qnn -o convnext/model_compiled.onnx
```

### 7. Benchmark on NPU

Run the compiled model on the NPU and record the latency.

```bash
wmk perf -m convnext/model_compiled.onnx --ep qnn --iterations 100
```

Keep this number in mind — the next step puts it in context.

### NPU vs. CPU: The 25x Speedup

Now benchmark the same optimized model on CPU for comparison.

```bash
wmk perf -m convnext/model_opt.onnx --ep cpu --iterations 100
```

The difference is dramatic: the quantized model on NPU runs roughly **25x faster** than the original on CPU. Same model, same accuracy, completely different performance. This is what the full pipeline — analyze, optimize, quantize, compile — delivers.

## Deep Dive: The Analyzer

The analyzer deserves a closer look, because it is the key to building portable ONNX models. It is made of two parts: **Linter** and **AutoConf**.

### Linter — ESLint for ONNX

The Linter checks every operator in the model graph against target execution providers and classifies them with a simple color scheme:

- **Green** — fully supported
- **Gray** — partially supported (runs, but may fall back to CPU for some configurations)
- **Red** — unsupported (will not run on the target EP)

Think of it as **ESLint, but for ONNX models**. It gives developers an immediate, visual read on whether the model is portable.

### AutoConf — Automatic Optimization Config

AutoConf goes one step further. It **detects suboptimal patterns in the graph and generates the configuration for the optimizer**. Instead of manually figuring out which optimization passes to apply, the analyzer does the detective work and writes `optim.json` for you.

Together, Linter and AutoConf form the **analyze-optimize loop**: the analyzer diagnoses, the optimizer treats.

## Recap: Three Phases

The primitive workflow breaks down into three phases:

1. **Inspect** — understand the model (task, architecture, I/O shapes)
2. **Build a Portable Model** — export, analyze, optimize, quantize
3. **On-Device Benchmarking** — compile, perf, eval

Each command is independent. Developers can jump into any stage, swap parameters, retry with different settings, or skip steps entirely. That is the point of primitives — they give you the building blocks and get out of the way.

## When to Use Primitives

Primitive commands are the right tool when:

- You are **exploring a new model** and want to understand its structure
- You need to **debug a compatibility issue** at a specific pipeline stage
- You want to **experiment with different optimization or quantization settings**
- You are building a **custom workflow** that does not follow the standard pipeline order

In short, primitives are for the coding phase — flexible, transparent, and fully under your control. When the workflow is nailed down and it is time to automate, ModelKit offers a config-driven pipeline that collapses these steps into two commands. But that is a story for the next article.
