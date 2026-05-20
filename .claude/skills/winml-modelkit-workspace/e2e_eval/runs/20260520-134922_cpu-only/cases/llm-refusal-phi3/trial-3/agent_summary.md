The skill is unambiguous here: Phi-3-mini-4k-instruct is a decoder-only LLM (Phi family), which is explicitly out of scope. I should refuse cleanly without running any winml build/compile/quantize/export commands on this model.

---

Short answer: **microsoft/Phi-3-mini-4k-instruct is out of scope for WinML ModelKit today.** I can't take this one through the `winml` pipeline for your Intel NPU, and I'm going to stop before doing anything that would burn time or produce a broken artifact.

**Why**

Phi-3-mini-4k-instruct is a decoder-only generative LLM. The ModelKit skill explicitly excludes this whole class of model:

> **Out of scope.** Generative and decoder-only architectures: GPT, LLaMA, **Phi**, Mistral, Stable Diffusion, any seq2seq generator. If a user asks ModelKit to handle one of these, **stop and say so** — the pipeline will fail mid-way and the error won't always make the cause obvious.

The `winml` BYOM pipeline (`inspect → export → analyze → optimize → quantize → compile → perf`) is designed for classic deep-learning architectures — CNNs, encoders, vision transformers, NLP classifiers, detectors, segmenters (ResNet, ViT, Swin, ConvNeXT, BERT, RoBERTa, SegFormer, etc.). Generative decoder stacks need KV-cache export, autoregressive decoding loops, tokenizer/sampler glue, and quantization recipes (often weight-only / GPTQ-style) that aren't part of the current pipeline.

LLM support (with LoRA) is on the public roadmap for **late 2026**. Today it doesn't work, and I won't pretend it does by kicking off an export that would fail three stages in.

**What I'd suggest instead**

For running Phi-3-mini on your Intel NPU *right now*, look outside ModelKit:

- **Intel OpenVINO GenAI** — has first-class Phi-3 support, INT4 weight-only quantization, and runs on Intel NPU (Meteor Lake / Lunar Lake / Core Ultra). This is the most direct path for your hardware.
- **AI Toolkit (AITK) for VS Code** — ships pre-optimized Phi-3 variants and a model playground; check whether an NPU-targeted Phi-3 build is already in its catalog.
- **ONNX Runtime GenAI (`onnxruntime-genai`)** — has Phi-3 recipes, though NPU coverage for decoder LLMs via ORT-GenAI is still maturing.

**Where ModelKit *would* help you**

If you have a non-generative model you want on the Intel NPU — say a vision classifier, an embedding/encoder model, BERT-style NLU, an object detector, or a segmenter — that's exactly the BYOM sweet spot. The flow would be `winml inspect -m <hf-id>` first to confirm support, then `winml config` + `winml build` targeting the OpenVINO EP for your NPU. Happy to walk through that anytime you have one of those in mind.

Relevant file: `C:/repo/WinML-ModelKit/.claude/skills/winml-modelkit/SKILL.md` (scope rules at lines 144–150).
