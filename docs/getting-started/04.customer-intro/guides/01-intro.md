# Getting Started with ModelKit — Swiss Knife for Windows ML Model

## What is ModelKit?

ModelKit is a CLI toolkit that builds **portable, performant, and high-quality** models for Windows ML. It bridges the gap between pretrained models and on-device inference — you bring a model from Hugging Face (or your own checkpoint), and ModelKit takes it all the way to optimized, device-ready ONNX.

## Goals

**Portable Models** — Build a model once and run it anywhere. ModelKit produces ONNX models that work across every ONNX Runtime backend.

**Flexible Pipeline** — Use built-in pipelines for end-to-end builds, or compose your own workflows from primitive commands. You decide how much control you need.

**Human-in-the-Loop** — Drill down into model details, pinpoint errors, and identify performance bottlenecks. ModelKit keeps you in the driver's seat at every stage.

**AI Agent Ready** — ModelKit provides built-in skills that work with all mainstream coding agents, so you can automate model-building workflows with AI assistance.

## Promises

**Out-of-Box Experience** — Install ModelKit and start building immediately. No boilerplate, no scaffolding, no manual dependency wrangling.

**One Toolkit Covers All EPs** — A single CLI handles every supported execution provider. You do not need separate tools for QNN, OpenVINO, VitisAI, or any other backend.

**Repeatability and Traceability** — Every command and pipeline produces deterministic, reproducible results. Configs capture the full build specification so you can replay, share, and audit builds.

**Build-Time Quality Gates** — ModelKit catches compatibility issues and suggests fixes automatically. The analyzer checks operator support before you deploy, not after.

## Command Overview

ModelKit organizes its commands into four categories:

| Category | Commands | Purpose |
|---|---|---|
| **Primitives** | `inspect`, `export`, `analyze`, `optimize`, `quantize`, `compile` | Individual pipeline stages you can run standalone or compose into custom workflows |
| **Pipeline** | `config`, `build` | Config-driven end-to-end builds that orchestrate primitives automatically |
| **Insights** | `perf`, `eval` | Benchmarking, evaluation, and hardware monitoring |
| **Utilities** | `env`, `cache` | Environment setup and cache management |

## ModelKit Is Right for You If

- You want to build optimized models for Windows ML without stitching together separate tools
- You want to quick-bench a model on NPU, GPU, or CPU with a single command
- You need to catch EP compatibility issues before deployment, not after
- You want repeatable, config-driven builds you can share with your team and check into source control
- You need to troubleshoot errors or performance bottlenecks in your ONNX pipeline
- You want AI agents to handle model optimization while you focus on your application

## Execution Provider Coverage

ModelKit supports a broad set of execution providers across priority tiers:

| Priority | Execution Provider | Status |
|---|---|---|
| P1 | QNN (NPU) | Supported |
| P1 | OpenVINO | Supported |
| P1 | VitisAI | Supported |
| P2 | QNN GPU | Planned |
| P2 | MIGraphX | Planned |
| P2 | TensorRT | Planned |
| P2 | DirectML | Planned |
| P3 | CPU | Planned |
