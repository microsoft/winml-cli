# winml-cli docs v3 — Known issues

> **Date:** 2026-05-26
> **Branch:** `docs/v3` (squashed as `gim-doc` tag)
> **Status:** Fact-checked findings from a 3-agent critical review pass. Each issue verified against the actual source/files.

Issues identified after the v3 doc set was assembled, fact-checked against `src/winml/modelkit/` and the actual doc files. Five issues are real and pending fix; three were claimed by reviewers but dismissed on second pass.

---

## Confirmed issues — pending fix

### 1. Stale link display text across 7 files (10+ occurrences)

Several pages were renamed during the Concepts restructure but their inbound link **display text** still uses the old titles. The link URLs themselves all resolve correctly (strict build passes); the issue is the visible label readers see.

| Stale text | Should be | Locations |
|---|---|---|
| `Quantization & QDQ` | `Datatype and Quantization` | `commands/eval.md:95`, `commands/hub.md:112`, `samples/convnext-primitives.md:83`, `samples/convnext-primitives.md:175`, `tutorials/npu-convnext.md:278` |
| `Quantization concepts` | `Datatype and Quantization` | `commands/quantize.md:115` |
| `Concepts → Quantization and QDQ` | `Concepts → Datatype and Quantization` | `tutorials/npu-convnext.md:137` |
| `ONNX & Execution Providers` / `ONNX and execution providers` | `EP and Device` | `commands/compile.md:110`, `commands/eval.md:96`, `commands/inspect.md:104`, `commands/overview.md:69`, `commands/perf.md:102`, `commands/sys.md:114`, `samples/convnext-primitives.md:108`, `samples/convnext-primitives.md:176` |
| `Load and export concept` | `Load and export` | `commands/export.md:105`, `commands/inspect.md:100`, `commands/perf.md:101` |

**Fix:** sed-sweep all five label patterns to the new titles.

### 2. WinML CLI concept sub-group ordering misaligned with workflow

`mkdocs.yml` lists the WinML CLI Concepts sub-group in this order:

```
Primitives and pipeline
Load and export
Analyze and optimize
Compile and EPContext
Perf and monitoring
Eval and datasets
Config and build      ← last
```

But `winml config` is **Step 1** of the End-to-End Tour (`getting-started/end-to-end.md`), so a reader who finishes the Tour and turns to Concepts to go deeper has to walk past 5 other pages before reaching `config-and-build.md`, which documents what they just did.

**Fix:** reorder so `Config and build` follows `Primitives and pipeline`:

```
Primitives and pipeline
Config and build
Load and export
Analyze and optimize
Compile and EPContext
Perf and monitoring
Eval and datasets
```

### 3. `graphs-and-ir.md:29` opset 17 / GroupNorm factual error

Current text:

> "Opset 17 introduced layer-normalisation and group-normalisation operators in native form, eliminating the multi-node decompositions required by earlier opsets…"

Per the ONNX changelog, **`LayerNormalization` was added in opset 17** but **`GroupNormalization` was added in opset 18**. The compound claim is wrong.

**Fix:** rewrite to "Opset 17 introduced LayerNormalization in native form; GroupNormalization arrived in opset 18." Or drop the GroupNorm mention entirely.

### 4. ConvNeXt "Pick the right page" admonition missing from `end-to-end.md`

The admonition appears at the top of `samples/convnext-primitives.md:3` and `tutorials/npu-convnext.md:3` but is **absent** from `getting-started/end-to-end.md`. The three pages all use `facebook/convnext-tiny-224` and a reader coming from the End-to-End Tour has no signpost telling them about the other two pages.

**Fix:** add a matching `!!! info "Pick the right ConvNeXt page"` admonition near the top of `getting-started/end-to-end.md`.

### 5. `end-to-end.md:108` capital-B inconsistency

Line 108 reads `[Config and Build](../concepts/config-and-build.md)` (capital B). The nav label and line 88 of the same file use lowercase `Config and build`.

**Fix:** change to lowercase `b` to match.

---

## Issues claimed by reviewers but rejected on fact-check

### #2 (rejected) — Quickstart link description

A UX reviewer claimed `quickstart.md:63` says "full pipeline against a Qualcomm NPU". Actual text is "full pipeline from Hugging Face to NPU". The exact phrasing the reviewer quoted is not present. The link wording is mildly NPU-leaning but not the misrepresentation claimed. Optional minor wording tweak; not pursued here.

### #5 (rejected) — `<artifact>.onnx` placeholder ambiguity

A UX reviewer claimed Step 3 leaves the reader guessing the per-device filename. The actual prose at `end-to-end.md:121-125` explicitly lists all three filenames (`convnext_tiny_qnn_ctx.onnx`, `convnext_tiny_dml_ctx.onnx`, `convnext_tiny.onnx`) and tells readers where to find them. Reviewer missed reading the next sentence.

### #7 (rejected) — `weight-and-activation.md` forward-reference to `w8a16`

A UX reviewer claimed the page mentions `w8a16` before defining it. Actual text at line 25 defines it inline: "The compound precision shorthand `w8a16` (8-bit weights, 16-bit activations)". Reviewer wrong.

### #9 (partial → effectively rejected) — `optim` fields not declared on dataclass

A factual reviewer flagged that `WinMLOptimizationConfig` is a free-form dict subclass with no declared fields, so the JSON example field names (`gelu_fusion`, `layer_norm_fusion`, `matmul_add_fusion`) "may not be real". Verified that the fields **are** real keys recognized by the optimizer at `src/winml/modelkit/optim/pipes/graph.py:242-243`. The example is correct. Not a defect.

---

## Items intentionally left as-is

- **"WinML CLI" sub-group naming.** The sub-group inside Concepts is named `WinML CLI`, which is recursive (the product is `winml-cli`). Suggested rename to "Workflows" was proposed and explicitly declined earlier. No change.
- **Singular vs plural style split between Fundamentals and WinML CLI sub-groups.** Fundamentals uses singular pair-topics ("Graph and IR", "Weight and Activation", "EP and Device", "Datatype and Quantization") per the user's preference; WinML CLI still uses plurals ("Primitives and pipeline", "Eval and datasets"). The user has not asked to reconcile.
