# Phi-3-mini on Intel NPU via WinML ModelKit — not today

Short version: **WinML ModelKit can't do this yet.** Phi-3 is a generative decoder-only LLM, and that whole family (GPT, LLaMA, Phi, Mistral, Qwen, Stable Diffusion, seq2seq generators) is explicitly out of scope for the current pipeline. If we push `microsoft/Phi-3-mini-4k-instruct` through the toolkit, it will likely fail somewhere mid-pipeline (export, optimize, or quantize), and the error usually won't make the root cause obvious — so I'd rather flag it now than burn an hour of your time.

LLM support (with LoRA) is on the public roadmap for **late 2026**, but it isn't shipping today.

## What the toolkit is for

WinML ModelKit's `winml` CLI is built around classic deep-learning models — the kind of thing you'd target with ONNX Runtime on a Windows AI PC for low-latency, on-device inference. Concretely, that means:

- **Vision:** ResNet, ViT, Swin, ConvNeXT, SegFormer, Table Transformer, object detection
- **NLP (encoder-style):** BERT, RoBERTa, NER, classifiers
- **Speech, segmentation, etc.** in the same encoder/CNN/ViT shape

For those, the pipeline (`inspect → export → analyze → optimize → quantize → compile → perf`, or just `config` + `build` end-to-end) handles Intel NPU via the **OpenVINO** execution provider, Qualcomm via **QNN**, AMD via **VitisAI**, plus GPU/CPU fallbacks.

Phi-3 just isn't in that shape — it's a 3.8B-parameter autoregressive transformer with KV cache, dynamic sequence lengths, and quantization needs (per-channel weight-only INT4, group-wise) that the current ModelKit optimize/quantize stages aren't designed for. The compile stage for Intel NPU in particular wouldn't produce a usable artifact.

If you want, I can run `winml inspect -m microsoft/Phi-3-mini-4k-instruct` to show you what the toolkit itself says — but I'd expect a `Default` verdict (it has a known model type with TasksManager defaults), which is *not* a green light. The scope rule overrides the inspect verdict for generative LLMs, and that's by design: inspect only reads config, not the full graph.

## What actually works for Phi-3 on Intel NPU today

Since you're on Intel hardware, the realistic options for running Phi-3-mini on the NPU right now are:

1. **Intel's OpenVINO GenAI** — `optimum-intel` has direct Phi-3 support and can export to OpenVINO IR with INT4 weight compression targeting NPU. This is the path Intel themselves recommend for Core Ultra NPUs. Search for `OVModelForCausalLM` and the OpenVINO NPU plugin docs.
2. **ONNX Runtime GenAI** (`onnxruntime-genai`) — Microsoft's runtime specifically for generative models. There are pre-built Phi-3-mini ONNX variants on Hugging Face (`microsoft/Phi-3-mini-4k-instruct-onnx`) with DirectML and CPU configurations. NPU support via OpenVINO EP is improving but less mature.
3. **DirectML via ONNX Runtime** — runs on the iGPU rather than the NPU, but on Core Ultra the Arc iGPU is often the strongest target for LLMs anyway.

None of those go through `winml`. They're separate toolchains.

## If you want me to do something concrete

A few directions I can help with:

- **Pick a non-LLM model that *does* work on your Intel NPU through ModelKit** — e.g. a vision or BERT-family model — and walk through the full inspect → config → build → perf flow end to end. Good way to learn the toolkit without hitting the scope wall.
- **Get you started on the Intel OpenVINO GenAI path for Phi-3 outside of ModelKit** — that's the right tool for this job today, even though it's a different stack.
- **Confirm what hardware/EPs are registered** on your machine with `winml sys --list-device --list-ep` so we know what your NPU actually advertises.

Which of those would be most useful? My default would be option 2 — get you onto a path that will actually run Phi-3 on your NPU — unless you specifically need the ModelKit workflow, in which case option 1 with a different model is the way.
