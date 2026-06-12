# Tutorials

Tutorials are linear, prescriptive, end-to-end walkthroughs that guide you through building something concrete with `winml-cli`. Each tutorial moves in one direction—start to finish—so you can follow along without making decisions. If you need to understand the reasoning behind a feature, see the Concepts section (the why and when). If you need a quick reference for a specific command, see Commands (the what). Tutorials sit alongside Samples, which are reference-style demos that compare multiple approaches side by side rather than walking through a single path.

## Available tutorials

| Tutorial | What you'll build | Hardware |
|---|---|---|
| [Hugging Face Model to NPU](npu-convnext.md) | A quantized ConvNeXt image classifier compiled for Snapdragon NPU (with CPU/DirectML fallback) | Copilot+PC NPU primary; CPU works as fallback |
| [Bring Your Own ONNX Model](build-from-onnx.md) | Optimize and deploy an ONNX file you already have, using the analyze → optimize → re-analyze feedback loop | Any (CPU, NPU, GPU) |

More tutorials are coming, covering additional model families, execution providers, and deployment scenarios. Check back as the `winml-cli` documentation expands.
