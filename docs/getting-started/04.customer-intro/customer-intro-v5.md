# ModelKit Customer Introduction Deck — V5

**Session**: 20-minute introduction for software vendors
**Audience**: Developers and engineering leads new to ModelKit, interested in bringing models to WinML
**Author**: Zheng Te
**Date**: March 2026
**Classification**: MVP Summit

---

## Flow Overview

```
Opening → Slide 1-2 (Intro) → Slide 3-4 (Demo Setup) → Live Demos → Slide 5-9 (Recap) → Slide 10 (Close)
```

| Phase | Content | Time |
|-------|---------|------|
| Opening + Intro | Opening → Slide 1 → Slide 2 | ~4 min |
| Demo Setup | Slide 3 (Workflow) → Slide 4 (Three Demos) | ~2 min |
| Live Demos | Demo 1-3 in terminal | ~8 min |
| Recap | Slides 5-9 | ~4 min |
| Close | Slide 10: Why ModelKit? | ~2 min |
| **Total** | | **~20 min** |

---

## Opening

> Hello everyone, I'm Zheng from the WinPD team, welcome to this session. In the next 20 minutes, I'll introduce ModelKit — a new toolkit we've built over the past few months.

---

## Part 1: Intro

---

### Slide 1: What is ModelKit?

#### Definition

_ModelKit is a CLI toolkit to build portable, performant and high-quality models for Windows ML._

#### Goals

1. **Portable Models** — Build once, run anywhere
2. **Flexible Pipeline** — Compose pipeline to build any model
3. **Human-in-the-Loop** — Step into any stage for model refinement and error debugging
4. **AI Agent Ready** — Skills for AI-augmented model building workflows

#### Promises

1. **Out-of-box Experience** — supported models work with minimal setup
2. **One Toolkit Covers All EPs** — no separate toolchain per vendor
3. **Repeatability and Traceability** — config-driven, predictable builds
4. **Build-Time Quality Gates** — catch issues ahead of time, with auto-fix

#### Speaker Transcript

> So, what is ModelKit?
>
> ModelKit is a CLI toolkit to build portable, performant and high-quality models for Windows ML.
>
> With ModelKit, you can build a model once and run anywhere — using built-in pipelines or composing your own from primitive commands. You can drill down into model details, pinpoint errors or performance bottlenecks at any stage. And ModelKit is AI-ready — we provide built-in skills that work with your favorite coding agent.
>
> ModelKit promises you an out-of-box experience — one toolkit covers all EPs, as well as full repeatability and traceability throughout both commands and pipelines. And we build quality gates into ModelKit to catch compatibility issues and suggest fixes automatically.

---

### Slide 2: ModelKit — Command List

#### Command List

| Primitives | Pipeline | Insights | Utilities |
|-----------|----------|----------|-----------|
| inspect | config | analyze | cache |
| export | build | debug* | doctor |
| optimize | perf | | setting |
| quantize | eval | | sys |
| compile | run* | | |

*\* coming soon*

#### Speaker Transcript

> OK, let's quickly go through the commands. We bucketize the commands into four categories.
>
> Primitive commands — you can use them individually or compose them into workflows. Pipeline commands that help you build and benchmark models end-to-end. Insight commands that enable model analysis and debugging. And a few utilities that support daily usage.

---

## Part 2: ModelKit in Practice

---

### Slide 3: Background — BYOM Workflow

#### One-liner

_Enhanced workflow with three commands as quality gates_

#### Three Pillar Steps

| Command | Pillar | What it guards |
|---------|--------|----------------|
| **Analyze** | Portability | Does it run on the target EP? |
| **Optimize** | Performance | Is the graph efficient enough? |
| **Quantize** | Fidelity | Is the model still accurate? |

#### Speaker Transcript

> Before we jump into the demos, let me show you the workflow behind ModelKit.
>
> This is the pipeline your model goes through. Given a source model, we export to ONNX, analyze, optimize, quantize if needed, and evaluate before shipping.
>
> Here I want to highlight three commands that serve as quality gates. Analyze for portability — checking whether your model runs on the target EP. Optimize for performance — making sure the graph is efficient enough. And quantize for fidelity — understanding the impact on model accuracy after compression.
>
> These three steps directly impact the quality of your output. And ModelKit gives you full control over each one.

---

### Slide 4: Three Ways to Build ConvNeXT with ModelKit

#### Demo Preview

All three demos use the **same model — ConvNeXT** — but in different ways:

1. **Build with Primitive Commands**
   - Full control, step by step
   - Iterate, debug, experiment

2. **Build with Config-Driven Pipeline**
   - Two commands, automated
   - Reproduce, polish, hand over

3. **Benchmark in One Command**
   - Zero setup, instant results
   - Validate, benchmark, deliver

#### Speaker Transcript

> I'm going to show you three ways to build models with ModelKit. All with the same model — ConvNeXT — for easy comparison.
>
> First, I'll go with primitive commands. You'll see how to craft a model step by step with ModelKit. Then, I'll build ConvNeXT again with the config-driven pipeline — only two commands. And last, I'll show you how to quick-bench a model with ModelKit — in one command.
>
> Let's start.

---

## Part 3: Recap Slides

*After demos, switch back to slides.*

---

### Slide 5: Build ConvNeXT with Primitive Commands

#### Two Phases

**📦 Build Portable Model**
- `export` — PyTorch to ONNX conversion
- `analyze` — EP compatibility and performance gap detection
- `optimize` — graph optimizations (shape inference, fusion, rewrite)
- `quantize` — low-bit model compression for fast inference

**⚡ On-Device Benchmarking**
- `compile` — target a specific EP
- `perf` & `eval` — measure latency, throughput, and accuracy on device

#### Speaker Transcript

> OK, let me recap. In demo one, we used primitive commands to bring ConvNeXT to Windows ML — step by step.
>
> Two phases. Build a portable ONNX through export, analyze, optimize, quantize. Then benchmark on device with compile, perf, and eval.
>
> This gives you full control — you can jump into any stage, try different settings to fix errors or tweak performance.

---

### Slide 6: Analyze ConvNeXT for EP Compatibility

#### Key Points

- Tagline: "Analyzer is the key to building portable ONNX models"
- **🚦 Linter**: ESLint for ONNX — Supported / Partial / Unsupported
- **⚙️ AutoConf**: GNU AutoConf for ONNX — tests capabilities, detects patterns, suggests fixes

#### Speaker Transcript

> Let me go deeper on the analyzer — it's the key to building portable ONNX models.
>
> The analyzer is made of two parts — Linter and AutoConf.
>
> The linter is like ESLint, but for ONNX. As you saw, it checks operator compatibility and classifies — green for supported, gray for partial, red for unsupported.
>
> AutoConf detects suboptimal patterns and generates the config for the optimizer. Together they form the analyze-optimize loop — which is what makes the models portable.

---

### Slide 7: Build ConvNeXT with Config-Driven Pipeline

#### Key Points

- Two steps: `wmk config` → `wmk build` — same pattern as CMake
- Config is reviewable, editable, version-controllable

#### Speaker Transcript

> In demo two, we used `config` and `build`. Two commands instead of eight.
>
> `wmk config` generates the build config — auto-detects everything. `wmk build` orchestrates the full pipeline. Same result, repeatable and scriptable. Think CMake for models.

---

### Slide 8: Primitive Commands vs. Config-Driven Pipeline

#### Comparison

| | Primitive Commands | Config-Driven Pipeline |
|---|---|---|
| **Lifecycle Analogy** | Coding | Polish |
| **Best for** | Flexible workflow | Production-ready delivery |
| **Control** | Start from any stage, try different settings to fix errors or tweak performance | Repeatable, scriptable, version-controllable |
| **Steps** | One command per stage | Two commands: config + build |
| **When to use** | Exploring, debugging, prototyping | CI/CD, batch builds, team workflows |

#### Speaker Transcript

> So when do you use which? Think of it like a development lifecycle.
>
> Primitive commands are the coding phase — you explore, debug, try different settings until the model works the way you want. Full flexibility.
>
> Config-driven pipeline is the polish phase — you've figured out what works, now you make it repeatable, scriptable, and easy to hand over to your team.
>
> Both produce the same portable ONNX. It's about where you are in your workflow and how much control you need.

---

### Slide 9: Benchmark ConvNeXT in One Command

#### Key Points

- `wmk perf -m facebook/convnext-base-224 --ep qnn --iterations 100 --monitor`
- One command: load, export, optimize, benchmark
- Live NPU/CPU monitoring, latency percentiles, throughput

#### Speaker Transcript

> And the third way — the easiest. Say someone hands you a production-ready model. You just want a quick smoke test — does it run, how fast is it? One command. `wmk perf` with a model ID. Load, export, optimize, benchmark — all behind the scenes. Think of it as a sanity check before you commit to a full integration.
>
> That's three ways to build. Primitive commands for development — iterate, debug, experiment. Config-driven pipeline for release — reproduce, polish, hand over. And one command for QA — validate, benchmark, deliver.

---

## Part 4: Close

---

### Slide 10: Why ModelKit?

#### 🎯 ModelKit is Right for You If

1. You want to build models that run on **any Windows device**
2. You want to benchmark a model with **one command**
3. You want to catch compatibility issues **ahead of time**
4. You want **deep insights** into your model
5. You want a **repeatable and traceable** model building process
6. You want **AI agents** to build and profile models for you

#### 🗺️ Roadmap

- **Q4 2025**: Project Kickoff
- **Q1 2026**: Early Access & Feedback
- **Q2 2026**: Public Beta
  - Open Source
  - Coding Agent Skills
  - AITK Integration
- **Q3–Q4 2026**: Release Candidate
  - LLM Support (with LoRA)
  - More Devices — GPU & NPU
  - MLIR

#### Speaker Transcript

> OK, that's all for the demos. Now — if you want to build models for Windows ML, quickly benchmark a model, catch compatibility issues ahead of time, troubleshoot errors or performance bottlenecks, or you just want AI to do the heavy lifting — please reach out to us for early access. Your feedback is the most valuable thing to us.
>
> And the roadmap — ModelKit is ready for early access now. We'll release the public beta in Q2, with coding agent skills available and AITK integration. After that, we'll continue bringing more into ModelKit — LLM support, MLIR, and broader device coverage.
>
> Thank you. Happy to take any questions.

---

## Appendix

### Session Flow

| Phase | Content | Time |
|-------|---------|------|
| Opening + Intro | Opening → Slide 1 → Slide 2 | ~4 min |
| Demo Setup | Slide 3 (Workflow) → Slide 4 (Three Demos) | ~2 min |
| Live Demos | Demo 1: ConvNeXT primitives | ~4 min |
| | Demo 2: ConvNeXT config+build | ~2.5 min |
| | Demo 3: ConvNeXT one command | ~1.5 min |
| Recap | Slides 5-9 | ~4 min |
| Close | Slide 10: Why ModelKit? | ~2 min |
| **Total** | | **~20 min** |

### EP Coverage Reference

| EP | Hardware | Priority |
|----|----------|----------|
| QNN | Qualcomm NPU | P1 |
| OpenVINO | Intel NPU/GPU/CPU | P1 |
| VitisAI | AMD NPU | P1 |
| QNN | Qualcomm GPU | P2 |
| MIGraphX | AMD GPU | P2 |
| TensorRT | NVIDIA GPU | P2 |
| DML | DirectML (any GPU) | P2 |
| CPU | Fallback | P3 |
