# Two Commands to Production — ModelKit's Config-Driven Pipeline

## From Coding to Polish

In the previous guide you built ConvNeXT step by step with primitive commands — the coding phase. Now you will build the same model with just two commands using the config-driven pipeline. Think of this as the **polish phase**: repeatable, scriptable, and ready to hand off to your team.

## Step by Step

### 1. Generate the Build Config

Run `wmk config` to generate a JSON config file. This config contains all settings for every pipeline step — task, I/O shapes, optimization flags, quantization parameters — all auto-detected from the model.

```bash
wmk config -m facebook/convnext-base-224 -o convnext_config.json
```

You can review it, edit it, or pass it directly to the build command. The config is a plain JSON file — open it in any editor and you will see every knob ModelKit exposes.

### 2. Build the Model

Run `wmk build` with that config. It orchestrates the full pipeline — export, analyze, optimize, quantize, compile — all in one go.

```bash
wmk build -c convnext_config.json -m facebook/convnext-base-224 -o convnext_build/
```

### 3. Benchmark the Result

Run the benchmark to confirm. Same model, same quality — but two commands instead of eight.

```bash
wmk perf -m convnext_build/model.onnx --ep qnn --iterations 100
```

## The CMake Analogy

If you have used CMake, the pattern will feel familiar. `wmk config` is like `cmake -B build` — it inspects the project and generates a build specification. `wmk build` is like `cmake --build build` — it executes that specification. The config file sits in the middle: reviewable, editable, and version-controllable.

## Config Is Your Build Specification

The generated config file is the single source of truth for your build. You can:

- **Review** it before building to verify auto-detected settings
- **Edit** it to override optimization flags, quantization parameters, or target EPs
- **Version-control** it alongside your application code
- **Share** it with teammates so everyone produces identical builds
- **Replay** builds deterministically on any machine

## Primitive Commands vs. Config-Driven Pipeline

| | Primitive Commands | Config-Driven Pipeline |
|---|---|---|
| **Lifecycle analogy** | Coding phase | Polish phase |
| **Best for** | Exploring, debugging, experimenting | Delivery, handoff, CI/CD |
| **Control** | Full — enter at any stage, tweak any setting | Config-level — edit the JSON, then build |
| **Steps** | One command per stage (~8 commands) | Two commands (`config` + `build`) |
| **When to use** | You need to iterate on a specific stage or diagnose an issue | You want repeatable, scriptable builds |

Both approaches produce the same portable ONNX model. The difference is where you are in the development lifecycle and how much manual control you need.

## One-Command Benchmark

There is an even simpler option. If you have a model and just want a quick smoke test — does it run, how fast is it — use `wmk perf` with a model ID directly:

```bash
wmk perf -m facebook/convnext-base-224 --ep qnn --iterations 100 --monitor
```

This handles everything behind the scenes: load, export, optimize, and benchmark. Live hardware monitoring is included with the `--monitor` flag. Think of it as a sanity check in the QA process.

## Recap: Three Ways to Build

You have now seen all three ways to build a model with ModelKit:

1. **Primitive commands** for development — iterate, debug, experiment
2. **Config-driven pipeline** for polish and handoff — repeatable, scriptable, two commands
3. **One command** for QA — validate, benchmark, deliver

Same ConvNeXT, three different approaches. Pick what fits your workflow.
