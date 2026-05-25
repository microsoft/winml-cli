# eval-is-model-supported

## Prompt

> Before I commit to building `nvidia/segformer-b1-finetuned-ade-512-512` end-to-end on my Intel NPU, is there a way to know if it's going to make it through the full pipeline? Burned half a day last time on a model that exported fine but then died at the optimize stage.

## with_skill — 6/6

### Response

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

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✓ | Recommends `winml inspect` as the first, cheapest check (config-only, no weights download). | PASS | Inspect framed as the cheapest preflight ('reads HF config without downloading weights'). |
| ✓ | Recommends `winml analyze` after inspect -- explicitly to find op patterns that won't survive optimize/quantize for the target EP. | PASS | Analyze step explicitly framed as 'the step that would have saved you half a day' — per-EP op-coverage check. |
| ✓ | Identifies OpenVINO as the EP for Intel NPU. | PASS | Uses --ep openvino --device NPU; mentions OpenVINOExecutionProvider in the EP-registration check. |
| ✓ | Explains analyze's role: a linter that classifies operators as supported / partial / unsupported per EP (so the user knows WHY this answers their question). | PASS | Linter behavior explained (classifies operators per EP) plus exit code 0/1/2 mapping. |
| ✓ | Does NOT just say "build the full pipeline and see what happens" -- that's exactly the workflow the user already said burned them. | PASS | Whole response framed as pre-flight before committing to the build. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | All ? winml commands verified runnable against --help. |

---

## without_skill (baseline) — 2/6 ⚠

### Response

# Pre-flight checking `nvidia/segformer-b1-finetuned-ade-512-512` for Intel NPU

Totally fair concern — "exports fine, dies at optimize" is the classic ONNX-on-NPU failure mode, and burning half a day on it stings. Here's how I'd think about de-risking this *before* you commit to the full pipeline.

## TL;DR on this specific model

SegFormer-B1 is a semantic segmentation transformer (hierarchical MiT encoder + lightweight all-MLP decoder). On an **Intel NPU** going through OpenVINO, it's *plausible but not guaranteed* — the encoder is mostly standard ops, but a few things historically trip up NPU compilers:

- **LayerNorm** over non-trailing axes (SegFormer normalizes spatial-flattened tokens — sometimes fine, sometimes folded into a pattern the compiler doesn't recognize).
- **Overlapping patch embeddings** — these are just `Conv2d` with stride < kernel, which is fine.
- **Efficient self-attention** — SegFormer reduces K/V spatially via a strided conv before attention. The `Reshape` + `Transpose` + `MatMul` + `Softmax` chain is usually OK but can blow up if shapes go dynamic.
- **Bilinear `Resize`** in the decoder head — Intel NPU/OpenVINO generally supports this, but `align_corners` / `half_pixel` modes vary by NPU generation.
- **`GELU`** — supported on modern Intel NPUs (Meteor Lake / Lunar Lake / Arrow Lake). On older NPUs it falls back to CPU.
- **Dynamic shapes** — segmentation models often export with dynamic H/W. NPUs almost always need static shapes; if you don't lock the input to 512x512 at export time, the optimize/compile stage *will* fail.

So: it's a model that *should* work, but only if the export is clean and shapes are pinned.

## A "will it make it through?" checklist you can run in ~15 minutes

Rather than committing to the full pipeline, do this triage first:

### 1. Export with static shapes and inspect

Export to ONNX with a fixed input (`1x3x512x512`), opset 17 or higher. Then *before* doing anything else:

- Open it in **Netron** and eyeball the op list. Look for: `Trilu`, `ScatterND`, `GridSample`, `NonZero`, `Unique`, custom-domain ops, anything with dynamic dim symbols still present.
- Run `onnx.checker.check_model` and `onnx.shape_inference.infer_shapes` — if shape inference fails or leaves dims unresolved, the NPU compiler will fail too.

### 2. Get the op coverage list for your target

Intel publishes an **OpenVINO supported ops** matrix per device (CPU/GPU/NPU are different). The NPU list is the strictest. Cross-reference your ONNX op set against that list. Any op not on it = fallback to CPU at best, hard failure at worst.

For Qualcomm QNN and AMD VitisAI there are equivalent op-support tables — but you said Intel, so OpenVINO NPU plugin docs are the source of truth.

### 3. Dry-run the optimize stage in isolation

Most ONNX optimization pipelines (Olive, ORT graph optimizers, OpenVINO `mo`/`ovc`) can be run *without* the full quantize+compile cycle. Run just the optimizer with verbose logging:

- If it crashes during constant folding → usually a shape/dtype mismatch from the export.
- If it crashes during operator fusion → an op pattern the optimizer doesn't recognize (often LayerNorm or attention variants).
- If it crashes during EP partitioning → that op isn't supported on the NPU; you'll see which one.

This is the stage that bit you last time, so isolate it.

### 4. Try CPU-EP first, NPU second

Run the optimized ONNX through ONNX Runtime on **CPU** first with `ort.InferenceSession`. If that fails, the model is broken regardless of NPU. If it passes, *then* switch the EP to OpenVINO NPU. The error messages from the NPU EP at session-creation time will tell you exactly which subgraphs got partitioned to CPU vs NPU — if too much falls back to CPU, the model is "technically working" but you've lost the acceleration.

### 5. Quantization sanity check (only if you got past step 4)

INT8 static quantization needs a calibration dataset. SegFormer's outputs are per-pixel logits, so calibration quality matters. But more importantly: check whether your NPU requires **per-tensor** vs **per-channel** quant, and whether it supports **QDQ** vs **QOperator** format. Intel NPU prefers QDQ. A model that runs FP16 on the NPU may still fail to quantize if a `MatMul` has an unusual rank.

## Heuristics for "is this model a good candidate?"

When deciding whether to even try, I rank models like this:

| Signal | Good | Bad |
|---|---|---|
| Architecture family | CNN, ViT, SegFormer, DETR-lite | LLMs, Stable Diffusion, Whisper |
| Input shapes | Fixed, NCHW | Dynamic batch + dynamic H/W |
| Op count | < 500 nodes | > 2000 nodes |
| Exotic ops | None | `GridSample`, `ScatterND`, custom |
| Reference deployments | Someone already ran it on this NPU | Nobody has |

SegFormer-B1 scores: encoder-decoder transformer (medium risk), fixed shape if you export it that way (good), ~400 nodes (good), no exotic ops I'm aware of (good), and Intel has demoed SegFormer variants on NPU before (good signal).

I'd put this at maybe **70% likely to make it through end-to-end on a recent Intel NPU** — worth the attempt, but do the 15-minute triage above before committing the full half-day.

## What would make me more confident before starting

1. Which Intel NPU generation? (Meteor Lake NPU 3720 vs Lunar Lake NPU 4 — Lunar Lake is *much* more permissive.)
2. Are you OK with FP16, or do you need INT8 for latency?
3. Is the 512x512 input shape acceptable, or do you need variable resolution at runtime?

If FP16 + fixed 512x512 + Lunar Lake → I'd be ~85% optimistic. If INT8 + Meteor Lake → I'd run the export-and-inspect step first and decide from there.

### Grading

| | Assertion | Result | Evidence |
|---|---|---|---|
| ✗ | Recommends `winml inspect` as the first, cheapest check (config-only, no weights download). | **FAIL** | Baseline doesn't use winml inspect -- recommends export + ONNX-level introspection as the pre-flight. |
| ✗ | Recommends `winml analyze` after inspect -- explicitly to find op patterns that won't survive optimize/quantize for the target EP. | **FAIL** | Baseline doesn't recommend winml analyze -- uses ad-hoc try-and-see / netron inspection. |
| ✓ | Identifies OpenVINO as the EP for Intel NPU. | PASS | General knowledge identifies OpenVINO as the Intel NPU EP. |
| ✗ | Explains analyze's role: a linter that classifies operators as supported / partial / unsupported per EP (so the user knows WHY this answers their question). | **FAIL** | No mention of an EP-coverage linter step. Pre-flight is much less specific. |
| ✗ | Does NOT just say "build the full pipeline and see what happens" -- that's exactly the workflow the user already said burned them. | **FAIL** | Baseline largely recommends try-and-see / run-export-then-quantize approach without the cheap pre-flight that would catch op-pattern issues. |
| ✓ | All `winml` commands quoted in the response are runnable -- every flag exists in `winml <cmd> --help`, no positional model arg where the CLI is flag-only. | PASS | Response uses no winml commands (trivially passes). |

---
