# From Nine Commands to Two — ModelKit's Config-Driven Pipeline

## Same Model, Different Approach

The previous article walked through building ConvNeXT with primitive commands — seven individual steps from inspect to benchmark. That workflow is great for exploration and debugging, but once the recipe is known, running each command by hand gets repetitive.

ModelKit's **config-driven pipeline** offers a different approach: automation over manual steps. Same model (`facebook/convnext-base-224`), same output quality, but collapsed into two commands.

## The CMake Analogy

If the concept sounds familiar, it should. The config-driven pipeline follows the same pattern as CMake:

- **`wmk config`** is like **`cmake configure`** — it inspects the source, detects settings, and writes a build configuration.
- **`wmk build`** is like **`cmake --build`** — it reads that configuration and executes the full pipeline.

Separate the *what* from the *how*. Configure once, build repeatably.

## Step 1: Generate the Config

```bash
wmk config -m facebook/convnext-base-224 -o convnext_config.json
```

This single command auto-detects everything the pipeline needs:

- **Task** — image classification
- **I/O shapes** — input dimensions, output labels
- **Optimization flags** — which passes to apply for the target EP
- **Quantization parameters** — compression settings for INT8

The result is a JSON file that captures the entire build recipe. It is **reviewable** — open it in any editor to see exactly what will happen. It is **editable** — override any setting before building. And it is **version-controllable** — check it into source control alongside the model, so every build is traceable and reproducible.

## Step 2: Build the Model

```bash
wmk build -c convnext_config.json -m facebook/convnext-base-224 -o convnext_build/
```

`wmk build` takes the config and runs the full pipeline in one go: **export, analyze, optimize, quantize, compile**. Every stage that was a separate command in the primitive workflow now executes automatically, in the right order, with the right parameters.

## Benchmark the Result

```bash
wmk perf -m convnext_build/model.onnx --ep qnn --iterations 100
```

Same model, same quality — **two commands instead of eight**. The output is identical to what the primitive workflow produces, because the underlying pipeline stages are the same. The config-driven approach simply orchestrates them.

## Primitive Commands vs. Config-Driven Pipeline

| | **Primitive Commands** | **Config-Driven Pipeline** |
|---|---|---|
| **Lifecycle analogy** | Coding phase | Polish and delivery |
| **Best for** | Exploring, debugging, experimenting | Repeatable builds, team handoff, CI/CD |
| **Control** | Full — run any stage independently | Guided — configure once, build automatically |
| **Steps** | 7+ individual commands | 2 commands (config + build) |
| **When to use** | You need to iterate on a specific stage | The recipe is known and needs to be reproducible |

Both approaches produce the same portable ONNX model. The difference is where you are in the development lifecycle and how much manual control you need.

## One-Command Benchmark

There is an even simpler path. When someone hands over a model and the only question is *"does it run, and how fast?"*, ModelKit can handle everything in a single command:

```bash
wmk perf -m facebook/convnext-base-224 --ep qnn --iterations 100 --monitor
```

`wmk perf` with a model ID loads the model, exports it, optimizes it, and benchmarks it — all behind the scenes, with live hardware monitoring included. Think of it as a **sanity check for QA**: no config files, no build steps, just a quick smoke test.

## Recap: Three Ways to Build

ModelKit offers three approaches to building models, each tuned to a different stage of the development lifecycle:

1. **Primitive commands** for development — iterate, debug, experiment. Full control over every stage.
2. **Config + Build** for delivery — repeatable, scriptable, easy to share with the team. Two commands.
3. **One command** for QA — validate, benchmark, and deliver. A quick smoke test when the model is ready.

Same ConvNeXT, three different workflows. Pick what fits.
