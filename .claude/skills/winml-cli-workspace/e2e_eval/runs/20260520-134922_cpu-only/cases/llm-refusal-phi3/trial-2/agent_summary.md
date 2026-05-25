The skill is explicit about this. Phi-3 is a decoder-only LLM, which is explicitly out of scope per the skill description and the Scope section ("Out of scope. Generative and decoder-only architectures: GPT, LLaMA, **Phi**, Mistral..."). I should refuse cleanly without running any pipeline commands.

---

Short answer: **no, I can't help you put `microsoft/Phi-3-mini-4k-instruct` through WinML ModelKit today.** Phi-3 is out of scope for this toolkit, and running `winml build/export/compile/quantize` against it would fail somewhere mid-pipeline with an error that won't necessarily point at the real cause. I'd rather tell you up front than burn an hour of your time.

**Why it's out of scope**

ModelKit's BYOM pipeline (`inspect → export → analyze → optimize → quantize → compile → perf`) is built for *classic* deep-learning architectures — CNNs, vision transformers, encoder-style NLP models (ResNet, ViT, Swin, ConvNeXT, BERT/RoBERTa, SegFormer, Table Transformer, object-detection/segmentation heads, etc.).

Generative / decoder-only architectures are explicitly excluded. From the skill's Scope section:

> **Out of scope.** Generative and decoder-only architectures: GPT, LLaMA, **Phi**, Mistral, Stable Diffusion, any seq2seq generator.

Phi-3 is a decoder-only causal LM with KV-cache, RoPE, and a generation loop — the export, quantization, and EP-compile stages aren't designed for that shape. LLM support (with LoRA) is on the public roadmap for **late 2026**; it isn't a thing you can flip on today, and I shouldn't pretend otherwise.

**What you can do on your Intel NPU instead**

Your machine has `OpenVINOExecutionProvider` registered, so the NPU path itself is healthy — the blocker is purely the model class. A few real options:

1. **Run Phi-3 on Intel NPU via a tool actually designed for LLMs.** Intel's OpenVINO GenAI stack (`optimum-intel` + OpenVINO, or the OpenVINO GenAI runtime) targets exactly this: Phi-3-mini on Meteor Lake / Lunar Lake NPU, INT4/INT8 weight-only quantization, KV-cache handling, the works. That's the path Microsoft and Intel both point at for Phi-3-on-NPU today.
2. **Or use AI Toolkit (AITK) / Foundry Local.** If you want a Microsoft-side flow, AITK's model catalog ships Phi-3 variants pre-packaged for on-device inference, including NPU-targeted builds where available. That bypasses the BYOM pipeline entirely.
3. **If you specifically want to exercise WinML ModelKit on your Intel NPU**, pick an in-scope model — e.g. `microsoft/resnet-50`, a BERT/RoBERTa classifier, `google/vit-base-patch16-224`, or a SegFormer/Table-Transformer checkpoint. I can walk you through `winml inspect` → config → build → perf end-to-end on one of those whenever you want.

Happy to set up option 3 right now if you give me a non-generative model you care about, or point you at concrete OpenVINO GenAI / AITK steps for Phi-3 if option 1 or 2 is what you actually need.
