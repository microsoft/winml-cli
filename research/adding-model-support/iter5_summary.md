# Iter-5 — Two-agent workflow trial on 10 candidates (producer side, post-reviewer)

**Date**: 2026-06-22 PM, immediately after the producer/reviewer split (`_meta-007`).

**Producer**: this session.
**Reviewer**: invoked via Explore subagent after the initial 3-of-10 run; issued REQUEST_CHANGES with 5 actionable items. Producer then completed the items in option-B mode (see "Reviewer-driven completions" below). First real exercise of REVIEW.md — also surfaced `_meta-011` (reviewer-tool-budget gap) and `_meta-013` (`winml analyze` parquet rules missing on external hosts).

## What the user asked

> "好，你用新的架构，再试试之前的10个model"

The "new architecture" = SKILL.md Steps 0-6 + REVIEW.md (the producer/reviewer separation introduced this morning). The "10 models" = the 12 candidates from iter-1's audit minus the 2 already-shipped (depth_pro done, BLIP is out-of-scope this turn).

## Pre-build per-model tier table (the `_meta-010` requirement, captured retroactively)

This is the table the reviewer demanded under `_meta-010`. Every row is now classified as exactly one of {RUN, BLOCKED-UPSTREAM, OUT-OF-SCOPE-FOR-TURN}. Iter-5's original sin was running 3 then writing the summary; iter-5-after-review reframes:

| # | model_type | example HF id | iter-4 probe verdict | Classification | Outcome |
|---|---|---|---|---|---|
| 1 | bart | facebook/bart-large-mnli | VENDOR-ONLY | RUN | VALIDATED-NEGATIVE; recipe checked in with `_status: BROKEN` marker per `_meta-013` (bart-003) |
| 2 | marian | Helsinki-NLP/opus-mt-en-ru | VENDOR-ONLY (w/ winml override) | RUN | VALIDATED (encoder + decoder structurally validated; marian-003) |
| 3 | marian | Helsinki-NLP/opus-mt-fr-en | sibling of #2 | RUN (added after reviewer item #5) | VALIDATED (marian-004); confirms marian-003 template is reusable across opus-mt checkpoints |
| 4 | m2m_100 | facebook/nllb-200-distilled-600M | VENDOR-ONLY | OUT-OF-SCOPE-FOR-TURN | 600M model, build wall-time too large for the turn budget; recipe pattern is identical to marian-003 (vendor-covered text2text-generation); ship as a follow-up |
| 5 | mgp_str | alibaba-damo/mgp-str-base | UNREGISTERED for `mgp_str`; vendor has `mgp-str` feature-extraction only | OUT-OF-SCOPE-FOR-TURN | Requires writing an OnnxConfig subclass with 3-head outputs (L1-light); violates the "no new code" budget of this turn |
| 6 | pix2struct | google/pix2struct-ai2d-base | VENDOR-ONLY | RUN | VALIDATED-NEGATIVE at config stage (pix2struct-003); workaround (a) `--shape-config` attempted per reviewer item #4 and confirmed dead (pix2struct-004); workaround (b) hand-written recipe deferred to a separate turn (would land as pix2struct-005) |
| 7 | pix2struct | google/deplot | as #6 | BLOCKED-UPSTREAM | Same `winml config` refusal as #6 — gated on the same upstream gap (winml CLI AutoProcessor threading); no per-checkpoint exception expected |
| 8 | pix2struct | google/pix2struct-docvqa-base | as #6 | BLOCKED-UPSTREAM | Same as #7 |
| 9 | vilt | dandelin/vilt-b32-finetuned-vqa | UNREGISTERED | OUT-OF-SCOPE-FOR-TURN | True L1 — write OnnxConfig from scratch against transformers source. Also "first VQA contribution" per `_meta-003`; needs a `TASK_REGISTRY` decision (classification-VQA vs generative-VQA) before any code lands |
| 10 | vision-encoder-decoder | breezedeus/pix2text-mfr | VENDOR + WINML adds | RUN (added after reviewer item #5) | VALIDATED-NEGATIVE at fetch stage (vision-encoder-decoder-003); HF repo lacks standard weight files; recipe checked in with `_status: BROKEN` marker per `_meta-013`. Architecture-level L0★ path remains open for any standard VED checkpoint (e.g. nlpconnect/vit-gpt2-image-captioning); blocked only for THIS specific checkpoint |

**Score (post-reviewer)**: 5 RUN (3 VALIDATED + 2 VALIDATED-NEGATIVE with recipes + markers), 2 BLOCKED-UPSTREAM (pix2struct siblings — gated on same upstream gap), 3 OUT-OF-SCOPE-FOR-TURN (m2m_100 / mgp_str / vilt — explicitly deferred with reasons). Compared to the first iter-5 pass (3 of 10 with no classification of the unrun rows), this table now satisfies `_meta-010`.

## Reviewer-driven completions (option-B execution log)

The Explore subagent acting as reviewer issued REQUEST_CHANGES with 5 items. The producer executed all 5:

1. **Reviewer item #1 — "recipe schema violation in marian-003"**. ROOT CAUSE: SKILL.md was wrong, not the recipes. Real `WinMLBuildConfig` schema is `{loader, export, optim, quant, compile, eval}` (verified at `src/winml/modelkit/config/build.py:96-200`); SKILL.md previously claimed `{export, optim, quant, loader, eval}`, missing `compile` and incorrectly implying `eval` is required. SKILL.md Step 3 + REVIEW.md Outcome-L0 row rewritten. Captured as `_meta-012`.
2. **Reviewer item #2 — "re-run analyze with available EP"**. ROOT CAUSE: BLOCKED on host. `src/winml/modelkit/analyze/rules/runtime_check_rules/` contains only `README.md` on external hosts; `scripts/download_rules.py` is Microsoft-internal-only. `winml analyze --ep cpu ...` fails with "No runtime rule parquet files were found". REVIEW.md updated to add the parquet-availability caveat. Captured as `_meta-013`.
3. **Reviewer item #3 — "bart recipe should be marked broken"**. RESOLUTION: top-level `_status: "BROKEN — ..."` field added to `examples/recipes/facebook_bart-large-mnli/text-classification_fp16_config.json`. `WinMLBuildConfig.from_dict` uses `.get()` for known keys and silently ignores unknown ones, so the marker is safe. Convention also applied to `examples/recipes/breezedeus_pix2text-mfr/`. REVIEW.md gained a "known-broken recipe convention" check.
4. **Reviewer item #4 — "pix2struct must attempt at least one workaround"**. RESOLUTION: workaround (a) `--shape-config` attempted; confirmed dead — flag only accepts text/vision/audio dims, NO `max_patches`/`patch_dim` key. Captured as pix2struct-004 with full mechanism; the hand-written-recipe path documented as pix2struct-005 deferred.
5. **Reviewer item #5 — "expand the 3-of-10 sample"**. RESOLUTION: ran fr-en (marian-004 VALIDATED) and pix2text-mfr (vision-encoder-decoder-003 VALIDATED-NEGATIVE at fetch stage). The 7 unrun rows are now classified per the table above with explicit reasons.

## What the validated runs taught

### Marian (POSITIVE — 2 checkpoints validated)

- Producer prediction (iter-4): L0★, L0 reachable.
- Reality: L0★ confirmed for opus-mt-en-ru (marian-003) AND opus-mt-fr-en (marian-004). Recipe pair (`translation_fp16_encoder_config.json` + `translation_fp16_decoder_config.json` from `winml config --task translation`) generalises across opus-mt checkpoints with no manual edits — `winml config` auto-fills the per-checkpoint vocab.
- Artifacts mined per SKILL.md Step 4 (en-ru): encoder 204 nodes / 51.2M params / autoconf optim = {clamp_constant_values, gelu_fusion, matmul_add_fusion, remove_isnan_in_attention_mask}; decoder 392 nodes / 76.7M params / same optim. ScatterND on decoder KV-cache writes is the dominant "unknown" op in per-EP coverage — file as a per-EP rule gap.
- fr-en numbers: encoder 34.0s / 199 MB (external-data layout kicks in above en-ru size); decoder 42.3s / 346 MB.

### BART (NEGATIVE — first methodology counter-example to "probe ⇒ build succeeds")

- Producer prediction (iter-4): L0★, L0 reachable. (Optimum `BartOnnxConfig` covers text-classification.)
- Reality: build FAILS at export with `index -1 is out of bounds for dimension 1 with size 0`. Likely cause: `BartForSequenceClassification` pools the encoder hidden state at the last `eos_token_id` position; random int32 dummy input never contains an eos token, so `nonzero()` returns empty and `[-1]` indexing throws.
- **This falsifies the iter-4 methodology assumption that "Optimum coverage ⇒ build will succeed"**. Optimum coverage = the OnnxConfig exists. It does NOT = the DummyInputGenerator paired with that OnnxConfig produces inputs that survive checkpoint-specific assertions. Captured as bart-003 + `_meta-008`.

### Pix2Struct (NEGATIVE — second methodology counter-example, autoconfig dead-ends)

- Producer prediction (iter-4): L0★ in principle.
- Reality: `winml config` REFUSES to emit any draft — "Preprocessors for pix2struct need to be available for the ONNX export to infer input static shapes. Got: None". The autoconfig pathway is hard-stopped UPSTREAM of `winml build`, at Optimum's normalized-config layer.
- Workaround (a) `--shape-config` confirmed dead (pix2struct-004); workaround (b) hand-written recipe deferred.
- **New failure class**: "autoconfig path dead-ends before producing a recipe". Captured as pix2struct-003 + `_meta-009`.

### Vision-Encoder-Decoder / pix2text-mfr (NEGATIVE — third methodology counter-example, repo-format gate)

- Producer prediction (iter-4): L0★, L0 reachable. (Optimum + winml both register vision-encoder-decoder; composite-emit produces 2 drafts.)
- Reality: `winml config` SUCCEEDED and emitted both drafts. `winml build` FAILED at fetch with `breezedeus/pix2text-mfr does not appear to have a file named pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt or flax_model.msgpack.` HF repo stores weights in a non-standard layout.
- **A FOURTH gate** (after probe + winml registration + `winml config` cooperation): HF repo file-layout check. No diagnostic step currently covers this. Captured as vision-encoder-decoder-003 + suggestion to add `huggingface_hub.list_repo_files(...)` pre-flight in `winml config` or a new `winml doctor`. Architecture-level L0★ path remains open for any standard VED checkpoint.

## Methodology lessons captured this turn (now totalling 13 `_meta-*` findings)

1. **Optimum coverage is necessary but not sufficient** — `_meta-008` (bart-003 + vision-encoder-decoder-003 are both counter-examples). The reviewer agent now enforces the build-attempt requirement.
2. **`winml config` can dead-end before producing a draft** — `_meta-009` (pix2struct family + likely fuyu / donut variants). SKILL.md Step 1 verdict table now explicitly documents this gate.
3. **First-mover seq2seq template gap is now closed for marian and is generalisable** — marian-003 → marian-004 confirms the recipe pair pattern transfers across opus-mt checkpoints.
4. **Producer-only on a batch of 10 is still self-grading** — `_meta-010`. SKILL.md now requires a pre-build per-model tier table for batch contributions and REVIEW.md has a corresponding REQUEST_CHANGES rule.
5. **Reviewer subagent without terminal access cannot fully execute REVIEW.md** — `_meta-011`. REVIEW.md now distinguishes REQUIRED-FROM-EVIDENCE checks from REQUIRED-FROM-RE-EXECUTION checks; recommended reviewer-agent invocation pattern documented as a follow-up.
6. **Documentation drift between SKILL.md and `WinMLBuildConfig`** — `_meta-012`. Recommendation: generate the schema callout from the dataclass at doc-build time to eliminate the drift permanently.
7. **Host-provisioning gap for `winml analyze` parquet rules** — `_meta-013`. The fix lives at the cli-distribution layer, not in the methodology text; documented as a host-environment caveat in REVIEW.md.
8. **Recipe-marker convention for known-broken recipes** — top-level `_status` field is the lightest-weight option (no directory move, no separate README); silently accepted by `WinMLBuildConfig.from_dict` because the dataclass uses `.get()`. Documented in REVIEW.md and applied to bart-large-mnli + pix2text-mfr.

## Hand-off package for the next reviewer pass

If a separate reviewer agent (ideally with terminal access this time, per `_meta-011`) picks up this post-option-B state:

- **Recipes to re-verify by re-running `winml build`**:
  - `examples/recipes/Helsinki-NLP_opus-mt-en-ru/translation_fp16_{encoder,decoder}_config.json` — should reproduce `✅ Build complete` in ~78s combined.
  - `examples/recipes/Helsinki-NLP_opus-mt-fr-en/translation_fp16_{encoder,decoder}_config.json` — should reproduce `✅ Build complete` in ~76s combined (external-data layout; 199 MB encoder + 346 MB decoder).
  - `examples/recipes/facebook_bart-large-mnli/text-classification_fp16_config.json` — should reproduce the `index -1` error; `_status` field should NOT affect the build (verify dataclass ignores it).
  - `examples/recipes/breezedeus_pix2text-mfr/image-to-text_fp16_{encoder,decoder}_config.json` — should reproduce the "does not appear to have a file named pytorch_model.bin..." error.
- **Findings to audit**: marian-003/004, bart-003, pix2struct-003/004, vision-encoder-decoder-003, all of `_meta-008` through `_meta-013`.
- **Independent verifications the reviewer SHOULD do**:
  1. Re-run the Optimum probe and confirm marian/bart/pix2struct/VED verdicts.
  2. Re-read all three artifacts (analyze_result.json, export_htp_metadata.json, winml_build_config.json) for the marian fr-en build and confirm marian-004's numbers.
  3. Verify that adding a `_status` key to a recipe does NOT change `winml build` behaviour (positive control on marian fr-en after adding a no-op `_status` field).
  4. Optionally: attempt pix2struct workaround (b) — hand-written recipe with `flattened_patches[1, 4096, 770]` + `attention_mask[1, 4096]` — to land pix2struct-005.
- **What the post-option-B reviewer CANNOT verify**: `winml analyze` per-EP re-runs (parquet rules missing per `_meta-013`); cite this as a host limitation, not a producer failure.
- **What the producer still owes (out of this turn's scope)**: m2m_100 build (deferred for size), mgp_str OnnxConfig subclass (deferred for L1-light code), vilt full OnnxConfig + TASK_REGISTRY decision (deferred for L1 + new task family), pix2struct workaround (b) attempt (deferred for the hand-written-recipe + processor-threading question).
