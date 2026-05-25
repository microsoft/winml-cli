Phi-3-mini-4k-instruct is explicitly out of scope. The skill is clear: Phi is listed in the out-of-scope examples ("Generative and decoder-only architectures: GPT, LLaMA, **Phi**, Mistral, ..."). I should refuse cleanly and not run any winml build/compile/quantize/export.

---

Short answer: **this isn't in scope for the WinML ModelKit toolkit today** — I can't convert `microsoft/Phi-3-mini-4k-instruct` for your Intel NPU with `winml`, and I shouldn't try, because the pipeline will fail partway through and the failure mode won't be obvious.

## Why Phi-3 is out of scope

`winml` (the Windows ML ModelKit CLI) targets **classic deep-learning architectures** — CNNs, encoders, vision transformers, NLP classifiers, NER, detection, segmentation. Think ResNet, ViT, Swin, ConvNeXT, BERT, RoBERTa, SegFormer.

It explicitly **does not** support generative / decoder-only architectures today. From the skill's scope section:

> Out of scope. Generative and decoder-only architectures: GPT, LLaMA, **Phi**, Mistral, Stable Diffusion, any seq2seq generator. If a user asks ModelKit to handle one of these, stop and say so — the pipeline will fail mid-way and the error won't always make the cause obvious.

Phi-3-mini-4k-instruct is a decoder-only causal LM, so it lands squarely in that excluded set. The KV-cache, rotary embeddings, autoregressive decoding loop, and tokenizer-coupled generation that an LLM needs aren't part of the BYOM `inspect → export → analyze → optimize → quantize → compile → perf` pipeline yet. LLM support (with LoRA) is on the public roadmap for **late 2026** — it isn't ready today, and I won't pretend otherwise by kicking off `winml export` / `quantize` / `compile` against it.

(For the same reason I'm intentionally not running `winml inspect` on it either — inspect can sometimes accept an architecture via TasksManager defaults that's still out of scope, and the scope rule overrides what inspect prints.)

## What you can do instead for Phi-3 on Intel NPU

These are outside this skill, but they're the right pointers:

- **OpenVINO GenAI** (`optimum-intel` + `openvino-genai`) — Intel's first-party path for running Phi-3-class LLMs on Intel NPU / GPU. It handles INT4 weight-only quantization, KV-cache, and the decoding loop, which are the parts `winml` doesn't yet wrap for LLMs.
- **ONNX Runtime GenAI** (`onnxruntime-genai`) with the DirectML EP — runs Phi-3 on a wide range of Windows GPUs; NPU support for LLMs through ORT is still maturing.
- **AI Toolkit for VS Code / Foundry Local** — Microsoft's user-facing path for running Phi-3 locally on Windows; under the hood it uses the runtimes above.

## What `winml` *is* a great fit for

If you have a non-generative workload on the same Intel NPU — for example image classification (ResNet, ViT), segmentation (SegFormer), detection, BERT-style text classification, or embeddings — `winml` will happily build, quantize, and compile that to OpenVINO and benchmark it on your NPU. Bring me a model in one of those families and I'll walk you through `inspect → config → build → perf` end to end.

Happy to come back to Phi-3 once the LLM path lands on the roadmap, or to help right now if you want to pivot to a classic-DL model on your Intel NPU.
