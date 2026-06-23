# PR: breezedeus/pix2text-mfr — VED image-to-text recipe pair (KNOWN-BROKEN regression-pin)

**Iter**: 6 (validated-negative shipped iter-5 as vision-encoder-decoder-003; this PR pins the recipes with `_status` markers per `_meta-014` convention)
**Producer**: main agent (2026-06-23)
**Claimed tier**: `(Effort = L0★, Goal = L0-FAIL-UPSTREAM, Outcome = L0)`

## Summary

This PR ships the `breezedeus/pix2text-mfr` image-to-text recipe pair as a **KNOWN-BROKEN regression-pin** per [`_meta-014`](../skill_meta/findings.json) `_status` marker convention. The recipes are structurally correct VED templates (would build any standard HF VisionEncoderDecoderModel checkpoint with covered inner decoder); they are blocked at the **upstream HF repo-layout layer**, not at the winml export layer. Each recipe carries a top-level `_status: "BROKEN — DO NOT USE. ..."` field documenting the exact failure and the recipe's regression-coverage purpose.

This PR is shipped as a contribution because:
1. The recipes record the canonical L0★ VED template for any future contributor who needs a starting point.
2. The `_status` marker pattern is itself the regression-test for `WinMLBuildConfig.from_dict` correctly ignoring unknown top-level keys (positive control verified in iter-5 on `opus-mt-fr-en` encoder build with `_status` field present — build SUCCEEDED).
3. The failure class is upstream (HF repo only ships pre-exported `.onnx` files, no PyTorch weights for `AutoModel.from_pretrained`), so dropping the recipe entirely loses the diagnostic value.

No source-code changes.

## 1. Recipe files

- [examples/recipes/breezedeus_pix2text-mfr/image-to-text_fp16_encoder_config.json](../../../examples/recipes/breezedeus_pix2text-mfr/image-to-text_fp16_encoder_config.json)
- [examples/recipes/breezedeus_pix2text-mfr/image-to-text_fp16_decoder_config.json](../../../examples/recipes/breezedeus_pix2text-mfr/image-to-text_fp16_decoder_config.json)

Both recipes carry `"_status": "BROKEN — DO NOT USE. winml build fails at fetch with \`breezedeus/pix2text-mfr does not appear to have a file named pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt or flax_model.msgpack.\`. The HF repo stores weights in a non-standard layout. Recipe is structurally correct and would build any standard VED checkpoint; pinned here as regression coverage. See research/adding-model-support/model_knowledge/vision_encoder_decoder.json finding vision-encoder-decoder-003."`

## 2. README index row

[examples/recipes/README.md](../../../examples/recipes/README.md) — row to add for `breezedeus/pix2text-mfr | image-to-text | composite (encoder + decoder) | recipe pair | **BROKEN (upstream HF repo layout)**`.

## 3. Build output directory + artifact inventory

`temp/ved_build/` (gitignored) — **EMPTY for this checkpoint**. The encoder build attempt aborts at the fetch stage:

```
Error: Build failed: breezedeus/pix2text-mfr does not appear to have a file
named pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt or
flax_model.msgpack.
```

No `model.onnx`, no `model.onnx.data`, no `analyze_result.json`, etc. — the failure precedes the export pipeline.

**External-data layout check** ([`_meta-023`](../skill_meta/findings.json)): N/A (no artifacts produced).

## 4. Build log

The recipe `_status` field IS the build-log substitute — see §1. Stderr captured in `vision-encoder-decoder-003` mechanism_notes.

## 5. Appended findings

### Per-model — `model_knowledge/vision_encoder_decoder.json`

- [vision-encoder-decoder-001](../model_knowledge/vision_encoder_decoder.json) — pre-build coverage (generic VED export covers any standard HF VED checkpoint).
- [vision-encoder-decoder-002](../model_knowledge/vision_encoder_decoder.json) — REFINEMENT: Optimum natively covers VED image-to-text; winml ADDS encoder/decoder split overrides.
- [vision-encoder-decoder-003](../model_knowledge/vision_encoder_decoder.json) — **VALIDATED-NEGATIVE for breezedeus/pix2text-mfr** (this PR's primary basis).
- [vision-encoder-decoder-004](../model_knowledge/vision_encoder_decoder.json) — VED template confirmed reusable for `nlpconnect/vit-gpt2-image-captioning` (the recommended canonical L0★ VED reference instead of the broken one here).

### Skill-meta

No new `_meta-NNN` findings in this PR.

## 6. Optimum-coverage probe verdict

```python
mt = "vision-encoder-decoder"
# vendor: feature-extraction, image-to-text, image-to-text-with-past
# after_winml: + image-feature-extraction encoder override + text2text-generation decoder override
# added_by_winml: ["image-feature-extraction", "text2text-generation"]  (composite split)
```

**Verdict**: VENDOR-COVERED on `image-to-text` (composite splits into `image-feature-extraction` encoder + `text2text-generation` decoder via winml overrides per `vision-encoder-decoder-002`). Effort L0★ is correct CLASSIFICATION; in PRACTICE this specific checkpoint is upstream-blocked.

## 7. Claimed (Effort, Goal, Outcome) tier

- **Effort = L0★** (recipe-only template; structurally identical to vit-gpt2-image-captioning)
- **Goal = L0-FAIL-UPSTREAM** (build fails at fetch — not a winml gate; the recipe + `_status` field together encode "we tried, here's why it can't proceed")
- **Outcome = L0** (recipe + finding append + this report)

## 8. Goal-ladder verdict table (per [`_meta-018`](../skill_meta/findings.json), per-half per [`_meta-020`](../skill_meta/findings.json))

| Half | Tier | Verdict | Evidence |
|---|---|---|---|
| **encoder** | L0 | **FAIL-UPSTREAM** | `winml build` aborts at fetch: HF repo `breezedeus/pix2text-mfr` ships only `encoder_model.onnx` + `decoder_model.onnx` (pre-exported ONNX) + `config.json` / `tokenizer*` / `preprocessor_config.json`. NO `pytorch_model.bin` / `model.safetensors` / `tf_model.h5` / `model.ckpt` / `flax_model.msgpack`. `AutoModel.from_pretrained` requires one of these to materialize the PyTorch graph that `winml build` then traces. |
| **decoder** | L0 | **FAIL-UPSTREAM** | Same root cause — the fetch failure happens before encoder/decoder split. |
| **L1..L3 (both halves)** | **NOT-REACHED** | Per [`_meta-018`](../skill_meta/findings.json), FAIL halts the march. Lower tiers are unreachable until L0 is unblocked. |

**Short-circuit honored**: L0 FAIL-UPSTREAM is the FIRST FAIL verdict in the ladder, halts the march. The `_status` marker on the recipe is the artifact-of-record for this halted state.

**Diligence ladder ([`_meta-037`](../skill_meta/findings.json))** — invoked and recorded:

1. Re-read `vision_encoder_decoder.json` — vision-encoder-decoder-003 fully documents the failure mode; no workaround documented because the gate is upstream.
2. PR-mine — no PR has unblocked this checkpoint family; vit-gpt2 sibling (vision-encoder-decoder-004) works on standard-layout checkpoints.
3. Re-run `winml config` — already succeeded; `winml build` is where it fails. `winml config` produced the recipe pair correctly from the HF config alone (doesn't need weights).
4. `--ep-options` — N/A (failure is at fetch, not at EP).
5. `value_range` / shape pinning — N/A (failure is at fetch, not at export).
6. Custom Python harness — N/A (we don't have a PyTorch model to compare against; the upstream gate is precisely "no PyTorch model in repo").
7. **Re-verify upstream repo layout (2026-06-23)**: `HfApi.list_repo_tree('breezedeus/pix2text-mfr', recursive=True)` returns: `.gitattributes`, `README.md`, `config.json`, `decoder_model.onnx`, `encoder_model.onnx`, `generation_config.json`, `preprocessor_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`. **NO PyTorch weights, only pre-exported ONNX.** The failure mode is unchanged from iter-5 (vision-encoder-decoder-003).

**Feature gap from step 7**: `winml build` could conceivably support a "fetch pre-exported ONNX from HF instead of PT→ONNX export" path for repos like this. Captured under `vision-encoder-decoder-003` `feature_gaps_filed[]`. Until then, this checkpoint stays BROKEN.

## 9. Methodology-evolution declaration (per [`_meta-031`](../skill_meta/findings.json))

**No NEW methodology friction in this PR.** The `_status` marker convention (`_meta-014`) was the iter-5 finding that enabled shipping known-broken recipes; this PR uses it as designed. Triggers:

- (1) CLI surprise — none.
- (2) Doc-code drift — none.
- (3) Silent-failure mode — none (failure is loud and reproducible).
- (4) New verdict shape — `FAIL-UPSTREAM` is already covered by the `FAIL` slot in `_meta-018` vocabulary (the `-UPSTREAM` suffix is descriptive, not a new verdict).
- (5) Reviewer-found gap — pending.
- (6) Effort mis-estimate — none.
- (7) PR-mining discovery — none.

Reviewer should confirm "no methodology friction observed" per `_meta-031` anti-trigger. The diligence ladder application IS the methodology working as designed — recipe ships with a documented FAIL-UPSTREAM verdict + feature_gap entry, not a silent failure.

## Reviewer hand-off package — Step 6 9-item self-check

1. Recipe files — §1 ✓ (both carry `_status` marker)
2. README row — §2 ✓ (to add in this PR; row marks the recipe as BROKEN)
3. Build output dir + artifact inventory — §3 ✓ (empty by design, reason documented)
4. Build log — §4 ✓ (substituted by `_status` field — exact stderr captured in vision-encoder-decoder-003 mechanism_notes)
5. Appended findings — §5 ✓
6. Optimum-coverage probe verdict — §6 ✓
7. Claimed tier — §7 ✓ (Goal = L0-FAIL-UPSTREAM honestly)
8. Goal-ladder verdict table — §8 ✓ (FAIL-UPSTREAM halts march per `_meta-018`)
9. Methodology-evolution declaration — §9 ✓
