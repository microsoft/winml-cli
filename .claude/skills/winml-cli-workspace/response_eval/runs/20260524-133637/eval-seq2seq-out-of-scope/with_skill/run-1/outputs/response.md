# Heads up: CodeT5+ 220M is out of scope for WinML ModelKit today

Before we map out a pipeline, I have to flag a blocker: **`Salesforce/codet5p-220m` is a T5-family encoder-decoder (seq2seq) model**, and seq2seq / decoder-only / generative architectures are explicitly out of scope for the current `winml` toolkit. The skill's scope rule is clear:

> Out of scope. Generative and decoder-only architectures: GPT, LLaMA, Phi, Mistral, Stable Diffusion, **any seq2seq generator**. If a user asks ModelKit to handle one of these, stop and say so — the pipeline will fail mid-way and the error won't always make the cause obvious.

Code summarization with CodeT5+ requires autoregressive decoding (generate token, feed back, repeat with KV cache), which the BYOM pipeline isn't designed for yet. Generative model support (with LoRA) is on the public roadmap for **late 2026**, so it's coming — just not via `winml` today.

I don't want to walk you through `inspect → export → analyze → optimize → quantize → compile → perf` and have you hit a wall at `export` or, worse, at `compile` after you've already invested time. Better to set expectations now.

## What you can still do with `winml`

If you want to confirm the verdict yourself rather than take my word for it, that's exactly what `winml inspect` is for — it reads the model config without downloading weights and tells you whether the toolkit recognizes the architecture and has a loader/exporter path:

```bash
# Prereq: install winml (Python 3.10, then install the winml-cli wheel from AITK cache)
uv venv --python 3.10
.venv\Scripts\activate
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"

# Sanity-check the install
winml --help
winml sys --list-ep

# Ask the toolkit directly
winml inspect -m Salesforce/codet5p-220m
```

Trust whatever `inspect` says — even if it surfaces a TasksManager default loader, the scope rule still overrides. The skill body is explicit on this:

> Inspect tells you whether the toolkit knows how to handle the architecture. But always cross-check against the scope section — a model that inspect accepts can still be out of scope. The scope rule overrides anything inspect prints.

## Realistic paths to your goal (offline code summarization in a VSCode extension on Snapdragon X Elite)

Since the deployment target is fixed (Snapdragon X Elite NPU, offline, VSCode extension) and you specifically want code summarization, here are the practical options:

1. **Pick a classifier/encoder model instead of a generator, if your UX can tolerate it.** `winml` is in scope for encoder-style transformers — BERT, RoBERTa, ViT, etc. If "summary" can be reframed as "predict one of N tags / categories / topics" or "extract key spans," an encoder-only model is in scope today and runs great on QNN. This is a UX compromise, not a true summary, so flag the tradeoff.

2. **Use a runtime outside `winml` for the seq2seq part.** For genuine free-form code summarization on Snapdragon X Elite today, you'd reach for the QNN SDK directly or AI Toolkit's generative-model path (which uses different tooling than the `winml` BYOM CLI). That's outside what this skill covers — I can't walk you through that pipeline as a `winml` workflow because it isn't one.

3. **Wait for late-2026 generative support.** If your timeline allows, the planned LLM/LoRA support in ModelKit is the eventual right answer for this exact use case. I don't have a firm date beyond "late 2026" from the public roadmap.

4. **Swap to a smaller encoder + extractive summarization.** Combine a CodeBERT-style encoder (in scope) with extractive logic in your extension — identify the most "summary-worthy" lines/comments. Less satisfying than CodeT5+'s natural-language generation, but ships today.

## If you change models, here's the shape of the pipeline you'd use

So you're not flying blind on the next attempt, the QNN-on-Snapdragon pipeline for an in-scope model looks like this:

```bash
# Confirm QNN is registered on this machine
winml sys --list-ep

# Always inspect first
winml inspect -m <hf-id-or-onnx-path>

# Two paths from here — default to config + build
winml config -m <hf-id-or-onnx-path> -o codet5-replacement.config.json
# (edit the JSON to pin EP=QNN and any quant/optim choices you want)
winml build -c codet5-replacement.config.json -o ./dist

# Then benchmark on-device
winml perf -m ./dist/<artifact>.onnx --ep QNN --monitor
```

I'm intentionally not quoting exact flag spellings for `--ep`, device selection, or quantization mode — those evolve and you should read them live:

```bash
winml config --help
winml build --help
winml perf --help
```

But that's all academic for `codet5p-220m`. **The honest answer is: don't run that model through `winml` today.** Tell me which of the four paths above fits your constraints (UX tolerance for non-generative output, timeline, willingness to use a non-`winml` runtime) and I'll go deeper on whichever one you pick.
