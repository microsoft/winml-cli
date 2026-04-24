# ModelKit — Swiss Knife for Windows ML Model

## What Is ModelKit?

ModelKit is a CLI toolkit to build portable, performant, and high-quality models for Windows ML — bridging the gap between pretrained models and on-device inference.

One toolkit covers everything — export, optimization, quantization, compilation, and benchmarking — across all execution providers, regardless of silicon.

## Goals

ModelKit is built around four guiding principles:

- **Portable Models** — **Build once, run anywhere**. A model prepared through ModelKit targets the Windows ML runtime and runs on **any supported execution provider** without per-device rework.
- **Flexible Pipeline** — **Compose your own pipeline** from independent primitives. Each stage (export, optimize, quantize, compile) can be mixed, matched, and reordered to build any model.
- **Human-in-the-Loop** — **Nothing is a black box**. Step into any stage for refinement or debugging. Every intermediate artifact is inspectable, editable, and reproducible.
- **AI Agent Ready** — ModelKit exposes built-in skills that AI-augmented workflows can consume. **Coding agents can drive the entire build pipeline** programmatically.

## ModelKit Is Right for You If

- You want to build models that run on **any Windows device**
- You want to benchmark a model with **one command**
- You want to catch compatibility issues **ahead of time**
- You want **deep insights** into your model
- You want a **repeatable and traceable** model building process
- You want **AI agents** to build and profile models for you

## Roadmap

- **Q4 2025** — Project kickoff
- **Q1 2026** — Early access and feedback
- **Q2 2026** — Public beta: open source, coding agent skills, AI Toolkit integration
- **Q3–Q4 2026** — Release candidate: LLM support with LoRA, more devices (GPU & NPU), MLIR
