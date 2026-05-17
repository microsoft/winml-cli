Short answer: Phi-3-mini-4k-instruct isn't supported by WinML ModelKit today, so I can't put together a working winml pipeline for it — and I want to flag that before you spend time on it.

Why

ModelKit's BYOM pipeline (inspect -> export -> analyze -> optimize -> quantize -> compile -> perf) is built for classic deep-learning architectures: CNNs, vision transformers, encoder-style NLP models, detectors, segmenters. The skill that governs this toolkit explicitly calls out generative / decoder-only LLMs as out of scope, and names Phi by family:

> Out of scope. Generative and decoder-only architectures: GPT, LLaMA, Phi, Mistral, Qwen, Stable Diffusion, any seq2seq generator.

winml inspect would probably print Default for it (it has a known model type), but the pipeline is known to fail at a later stage — usually optimize/quantize/compile — with an error that doesn't make the root cause obvious. LLM support (with LoRA) is on the public roadmap for late 2026, but it doesn't work today.

What I'd suggest instead

For Phi-3 on an Intel NPU specifically, the right tools are not winml:
1. Intel OpenVINO GenAI — Intel's own stack for running Phi-3 / LLaMA / Qwen on the Core Ultra NPU.
2. ONNX Runtime GenAI — has Phi-3 example notebooks; pair with the OpenVINO EP.
3. DirectML path — works on the Intel GPU (not NPU).
