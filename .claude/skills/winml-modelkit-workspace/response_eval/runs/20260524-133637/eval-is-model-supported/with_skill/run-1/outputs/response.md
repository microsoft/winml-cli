Yes — and the half-day burn you described is exactly the failure mode ModelKit's early-stage commands are designed to prevent. The trick is to spend ~30 seconds on two cheap, read-only checks *before* you touch `export`, `optimize`, or `build`. They don't guarantee success, but they catch the vast majority of "exports fine, dies later" cases.

## First, the scope check (free, no command needed)

`nvidia/segformer-b1-finetuned-ade-512-512` is a SegFormer-family semantic segmentation model. SegFormer is explicitly called out as in-scope for ModelKit (it's in the same bucket as ResNet / ViT / Swin / ConvNeXT / BERT / Table Transformer). Encoder-only vision transformer with a lightweight decode head — no generation, no KV cache, no seq2seq. So architecturally you're on a supported path. That alone puts you in much better shape than whatever bit you last time.

Intel NPU goes through the **OpenVINO** EP, which is currently Ready, so the hardware target is fine too.

## Then, the two commands that actually de-risk the build

These are the cheapest things in the pipeline. Run them in order; if either one is unhappy, don't proceed to export.

**1. `inspect` — does ModelKit know how to handle this architecture?**

It reads the Hugging Face config *without downloading weights*, and tells you the loader, exporter, the WinML inference class it'll use, I/O specs, and the build resolution the pipeline plans to apply. If `inspect` can't resolve a loader/exporter for the model, no later stage will either.

```bash
winml inspect -m nvidia/segformer-b1-finetuned-ade-512-512
```

Add `-f json` if you want to grep through it programmatically. Note the flag is `-m` — model IDs are not positional.

**2. `analyze` — will the operators survive optimize/quantize/compile on the OpenVINO EP?**

This is the one that specifically addresses your "died at optimize" scar. The analyzer's linter walks every operator in the graph and classifies it as supported / partial / unsupported *for the EP you're targeting*. That's where optimize and quantize failures usually originate — an op pattern that exported cleanly but has no EP-compatible rewrite.

```bash
winml analyze -m nvidia/segformer-b1-finetuned-ade-512-512
```

Check `winml analyze --help` for the exact flag to pin the target EP to OpenVINO — that's the spelling I'd read live rather than guess. If analyze flags ops as unsupported on OpenVINO, you've found out in seconds instead of half a day, and you have a concrete op name to either work around (different optim/quant config) or use as a "this model isn't a fit for this EP" call.

## The order matters

Always inspect first, then analyze. `inspect` is the cheapest possible signal (no weights downloaded). If it's a no, you stop right there. Only if inspect passes do you spend the export to get an ONNX that analyze can lint against the EP.

## Prereqs (in case you haven't installed `winml` yet)

ModelKit pins **Python 3.10 exactly**. Use `uv` to keep it isolated:

```bash
uv venv --python 3.10
.venv\Scripts\activate
uv pip install "$env:USERPROFILE\.aitk\bin\model_lab_runtime\cache\winml_cli-0.0.3-py3-none-any.whl"
winml --help
winml sys --list-ep
```

The `sys --list-ep` line is worth a glance — confirm `OpenVINOExecutionProvider` actually shows up on your machine. If it doesn't, no amount of analyzer green lights will save the compile stage; you'd be missing the EP runtime itself.

## What I'd actually do, start to finish

1. `winml sys --list-ep` — confirm OpenVINO EP is registered.
2. `winml inspect -m nvidia/segformer-b1-finetuned-ade-512-512` — confirm the architecture is resolvable.
3. `winml analyze -m nvidia/segformer-b1-finetuned-ade-512-512` (with the OpenVINO EP flag) — confirm operators are buildable on the Intel NPU.
4. Only now: `winml config` → `winml build -o <dir>` for the full end-to-end.

If step 2 or 3 surfaces anything unsupported, you've saved the half day. If they're both clean, your odds of a successful end-to-end run are dramatically higher — not 100%, but the residual risk is mostly in shape/quantization edge cases rather than "this model fundamentally doesn't fit," which is the expensive failure to avoid.
