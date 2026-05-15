# ModelKit — Core Promises & Command Reference

## Promises

- **Out-of-Box Experience** — **Build models for Windows ML with minimal setup**. Every command auto-detects optimal settings for the given model and **executes accordingly** — no manual configuration needed.
- **One Toolkit Covers All EPs** — **No separate vendor toolchain** per silicon. QNN, OpenVINO, VitisAI, DirectML, TensorRT, MIGraphX, and CPU — **seven EPs, one roof**.
- **Full Control** — **Reproducible and traceable** — whether through built-in pipelines or individual commands. **Step into any stage** for refinement or debugging. Every intermediate artifact is **inspectable and editable**.
- **Build-Time Quality Gates** — Catch compatibility problems, suboptimal operators, and quantization regressions — and **suggest fixes automatically**, before the model ever reaches a device. **Three quality pillars**:
  - **Analyze** ensures portability across target EPs
  - **Optimize** improves graph performance
  - **Evaluate** guards accuracy after quantization

## Command Overview

ModelKit organizes its commands into four categories:

**Primitives** — The building blocks. Each command handles one stage of the model preparation pipeline.

- `inspect` — Discover model metadata, task, I/O shapes, and EP support
- `export` — Convert source model to ONNX with hierarchy-preserving metadata
- `optimize` — Fuse operators, simplify graphs, prepare for target EP
- `quantize` — Compress model to low-bit precision for smaller footprint and faster inference
- `compile` — Generate device-specific binaries (e.g., QNN context binaries)

**Pipeline** — Orchestration commands that chain primitives together.

- `config` — Auto-detect task, I/O shapes, and optimal settings into a JSON config
- `build` — Orchestrate the full pipeline — export, analyze, optimize, quantize, compile
- `perf` — Benchmark latency, throughput, and hardware utilization
- `eval` — Evaluate model accuracy against reference datasets
- `run` — End-to-end inference with pre/post processing *(coming soon)*

**Insights** — Diagnostic tools for understanding what is happening inside a model.

- `analyze` — Lint operators, check EP compatibility, generate optimization config
- `debug` — Interactive model debugging and layer-by-layer inspection *(coming soon)*

**Utilities** — Housekeeping and environment management.

- `cache` — Manage built model artifacts and pipeline outputs
- `doctor` — Diagnose environment issues (runtimes, providers, dependencies)
- `setting` — Configure ModelKit preferences
- `sys` — System information and capability reporting

## Execution Provider Coverage

ModelKit supports the major execution providers in the Windows ML ecosystem:

| Provider | Hardware | Status |
|----------|----------|--------|
| **QNN** | Qualcomm GPU and NPU | 🟢 Ready |
| **OpenVINO** | Intel CPU, iGPU, dGPU, and NPU | 🟢 Ready |
| **VitisAI** | AMD NPU | 🟢 Ready |
| **TensorRT** | NVIDIA discrete GPUs | 🔶 Planned |
| **MIGraphX** | AMD discrete GPUs | 🔶 Planned |
| **DirectML** | Hardware-agnostic GPU backend | 🔶 Planned |
| **CPU** | Cross-platform fallback | ⚪ Always |
