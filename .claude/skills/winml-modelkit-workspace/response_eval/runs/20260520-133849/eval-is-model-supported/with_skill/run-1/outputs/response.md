# Pre-flight checks before committing to a full SegFormer build on Intel NPU

Short answer: yes — there are two cheap commands that, together, will tell you whether `nvidia/segformer-b1-finetuned-ade-512-512` is going to survive the full pipeline on your Intel NPU. Run them in this order and you'll know inside a couple of minutes, not half a day.

## Good news on scope

SegFormer is explicitly an in-scope architecture for ModelKit (classic vision transformer for segmentation — same family as ResNet, ViT, Swin, ConvNeXT, BERT, etc.). It is not a generative / decoder-only model, so it's not in the LLM/diffusion exclusion list. That's the first hurdle cleared before you spend a CLI cycle.

Your Intel NPU target maps to the **OpenVINO** execution provider, which is in 🟢 Ready status today.

## Step 1 — `winml inspect` (no weights downloaded)

This is the golden rule of the toolkit: always inspect first. It reads the model config without pulling weights, and prints the loader, exporter, WinML inference class, I/O specs, and the build resolution the pipeline would use.

```powershell
winml inspect -m nvidia/segformer-b1-finetuned-ade-512-512
```

What you're looking for:
- A resolved loader / exporter / inference class (not "unknown" or an error).
- I/O specs that match what you expect (image input, segmentation map output).
- No scope warning.

If `inspect` is happy, the toolkit knows how to handle the architecture. That kills the most common "export worked, everything else exploded" scenarios up front — the ones rooted in *the model architecture itself* not being recognized.

Add `-f json` if you want to grep the output programmatically.

## Step 2 — `winml analyze` against the exported ONNX (the part that actually catches optimize-stage blow-ups)

`inspect` confirms the architecture is known. But your last bad day was at the **optimize** stage, and that's almost always an operator-pattern problem, not an architecture problem. The cure for that is `winml analyze`, whose linter classifies every operator in the graph as supported / partial / unsupported for a specific EP+device. From the skill's own troubleshooting note: "Most operator-pattern failures land at [optimize/quantize] even when export succeeded. Re-run analyze against the exported ONNX; the linter will usually name the offending op pattern."

The catch: `analyze` runs against an ONNX file, so you do need to export first. But export is the cheap stage — it's the steps *after* export (optimize / quantize / compile) that burned your half day. Exporting just to feed the linter is the right tradeoff.

```powershell
# 1. Export (this is the cheap part)
winml export -m nvidia/segformer-b1-finetuned-ade-512-512 -o segformer.onnx

# 2. Lint it against your Intel NPU target
winml analyze -m segformer.onnx --ep openvino --device NPU
```

Exit codes are designed exactly for your use case:
- `0` — model fully supported. Green light to build end-to-end.
- `1` — partial support, some unsupported operators. Read the per-op output; the linter names the offending pattern. You can often dodge it by tweaking the optimize/quantize config rather than abandoning the model.
- `2` — error / analysis failure. Stop and read the message before going further.

If you want a saved report to diff against later, add `--output analyze.json`. If you want the auto-discovered optimization config that would dodge any partial ops, add `--optim-config optim.json` — you can feed that back into the later stages.

## Step 3 (optional, one-time) — confirm OpenVINO is actually registered

Cheapest of all and worth doing once on a new machine:

```powershell
winml sys --list-ep
```

If `OpenVINOExecutionProvider` (or its `openvino` / `ov` alias) isn't in the list, `compile` will silently produce an artifact you can't actually use. This is a sanity check, not a per-model check — once it passes on this box, you don't need to re-run it.

## Suggested pre-flight, end to end

```powershell
# One-time sanity check
winml sys --list-ep

# Per-model pre-flight (the two cheap commands)
winml inspect -m nvidia/segformer-b1-finetuned-ade-512-512
winml export  -m nvidia/segformer-b1-finetuned-ade-512-512 -o segformer.onnx
winml analyze -m segformer.onnx --ep openvino --device NPU
```

If all three are clean, then commit to the full build — either the primitive chain (`optimize` → `quantize` → `compile` → `perf`) or, more likely, the shortcut pair (`winml config` then `winml build`). If `analyze` flags unsupported ops, you've spent ~5 minutes finding that out instead of half a day, and you have a named operator pattern to plan around.

## A note on what `inspect` alone can and can't tell you

`inspect` is necessary but not sufficient as a "will it survive the whole pipeline?" check. It tells you the toolkit recognizes the architecture and has a loader/exporter/inference class for it. It does **not** tell you that every operator that ends up in the exported graph will be supported by your specific EP+device combo — that's what `analyze` is for. Pair them. That's exactly the gap that bit you last time.
