# codet5p-220m on Snapdragon X Elite NPU — not a fit for ModelKit today

Short answer: I can't walk you through a winml pipeline for `salesforce/codet5p-220m`, because that model is **out of scope** for ModelKit right now. I want to flag that up front rather than have you invest an hour in a build that fails three stages in.

## Why it's out of scope

`codet5p-220m` is a **T5-style encoder-decoder (seq2seq) generative model**. It takes a code snippet, runs an encoder, then autoregressively decodes a natural-language summary token by token. ModelKit's BYOM pipeline today targets *classic* deep learning architectures — CNNs, encoders, vision transformers, NLP classifiers, NER, detection, segmentation (ResNet, ViT, Swin, ConvNeXT, BERT, RoBERTa, SegFormer, Table Transformer, etc.).

Generative and decoder-only / seq2seq architectures are explicitly **excluded** from the current pipeline:

- LLMs (GPT, LLaMA, Phi, Mistral)
- Stable Diffusion
- Whisper
- **Any seq2seq generator** — which is exactly what CodeT5+ is

If you push CodeT5+ through `winml export`/`optimize`/`compile`, you'll either hit a hard stop or get a partial artifact that doesn't actually run end-to-end on QNN, because the pipeline doesn't yet handle the decoder loop, KV-cache, past-key-values inputs, or beam-search plumbing that seq2seq generation needs.

LLM and generative support (including LoRA) is on the public roadmap for **late 2026**. Until then, I'd be lying if I said "yes, here's the seven-stage pipeline."

## How to confirm this yourself

If you want to see ModelKit's own verdict before taking my word for it, install the CLI and run `inspect` — it reads the HF config without downloading weights:

```powershell
# 1. Create a Python 3.10 env (ModelKit pins 3.10 exactly)
uv venv --python 3.10
.venv\Scripts\activate

# 2. Install the CLI from the AITK cache
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"

# 3. Sanity check
winml --help
winml sys --list-ep   # confirms QNN is registered on your Snapdragon X Elite

# 4. The golden rule — inspect first
winml inspect -m salesforce/codet5p-220m
```

`inspect` will surface the loader / exporter / WinML inference class it would use. Even if it produces *something*, treat the scope rule above as the override: seq2seq is out, regardless of what `inspect` is willing to attempt.

## What you can actually do for offline code summarization in VS Code today

A few options, roughly in order of "closest to what you asked for":

1. **Wait for the generative roadmap.** If on-device CodeT5+ on QNN specifically is the goal, ModelKit's late-2026 LLM track is the path. Not useful this quarter.

2. **Use a runtime that already handles seq2seq generation on Snapdragon.** ONNX Runtime GenAI has scaffolding for encoder-decoder / KV-cached generation; you'd export CodeT5+ to ONNX with `optimum-cli` (encoder + decoder + decoder-with-past), then drive the generation loop yourself or via ORT GenAI. This sits *outside* ModelKit but is the realistic on-device path for a seq2seq today. Whether QNN will accept every op in the decoder is a separate question — you may end up on CPU EP for parts of it.

3. **Reframe the task as a classification / retrieval problem.** If your VS Code extension can live with templated summaries, a CodeBERT-style **encoder-only** classifier (e.g. predicting summary category, intent, or retrieving a nearest-neighbor summary from an index) *is* in scope for ModelKit. That pipeline would look like `inspect → export → analyze → optimize → quantize → compile --ep QNN → perf`, and would actually run on your NPU. Different product, but a viable one.

4. **Run CodeT5+ on CPU via ORT for now, NPU later.** Ship the extension against CPU ORT with the seq2seq model as-is, and swap to an NPU path once either ModelKit's generative track lands or you've validated an ORT GenAI + QNN flow. Latency will be the obvious tradeoff on a 220M seq2seq decoded token-by-token on CPU.

## If you want me to help with an in-scope model

If option 3 sounds interesting — or if you have a different in-scope model in mind (a code-understanding encoder, a classifier, an embedding model) — point me at the HF ID and I'll walk you through the real winml pipeline end to end: `inspect`, then either the primitives (`export → analyze → optimize → quantize → compile --ep QNN → perf`) or the shortcut pair (`config` then `build`) targeting your Snapdragon X Elite NPU.

But for `salesforce/codet5p-220m` specifically, the honest answer is: not yet, and I'd rather tell you that now than after a failed build.
