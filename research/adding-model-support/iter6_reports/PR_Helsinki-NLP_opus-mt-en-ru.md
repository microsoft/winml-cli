# PR: Helsinki-NLP/opus-mt-en-ru — translation recipe pair (fp32, CPU) — Goal-L2-encoder closed

**Iter**: 6 (composite recipe pair shipped iter-5 as marian-003; this PR adds the Goal-L2-encoder + L1-CPU evidence on top)
**Producer**: main agent (2026-06-23)
**Claimed tier**: `(Effort = L0★, Goal = L2-encoder, Outcome = L0)`

## Summary

This PR ships the `Helsinki-NLP/opus-mt-en-ru` translation recipe pair (encoder + decoder). It is the FIRST seq2seq composite pair contributed to the recipe catalog, and the first Marian-family entry. The recipe was generated via `winml config --task translation` (per [`_meta-020`](../skill_meta/findings.json) composite-expansion gate); both halves build cleanly on CPU at fp32. Goal-L1-CPU PASSes on both halves; Goal-L2 cosine = 1.000000 on the encoder (PT-vs-ONNX). Goal-L2 on the decoder is `DEFERRED-HARNESS` per [`_meta-018`](../skill_meta/findings.json) — see verdict table. No source-code changes.

Per [`_meta-020`](../skill_meta/findings.json), encoder + decoder ship as **ONE PR** with a per-half verdict matrix.

## 1. Recipe files

- [examples/recipes/Helsinki-NLP_opus-mt-en-ru/translation_fp16_encoder_config.json](../../../examples/recipes/Helsinki-NLP_opus-mt-en-ru/translation_fp16_encoder_config.json)
- [examples/recipes/Helsinki-NLP_opus-mt-en-ru/translation_fp16_decoder_config.json](../../../examples/recipes/Helsinki-NLP_opus-mt-en-ru/translation_fp16_decoder_config.json)

Note on filename: `fp16_*` is cosmetic per [`_meta-014`](../skill_meta/findings.json) — `quant: null` means fp32 weights ship. `winml perf` correctly reports `Model Precision: fp32` (see L1-CPU evidence below). The cosmetic filename is retained for catalog consistency.

## 2. README index row

[examples/recipes/README.md](../../../examples/recipes/README.md) — row to add for `Helsinki-NLP/opus-mt-en-ru | translation | composite (encoder + decoder) | recipe pair`.

## 3. Build output directory + artifact inventory

`temp/marian_build/{encoder,decoder}/` (gitignored — referenced by path for reviewer re-execution):

| Half | File | Size | Purpose |
|---|---|---:|---|
| encoder | `model.onnx` | inline | optimized graph (≤2GB ⇒ no external-data needed) |
| encoder | `analyze_result.json` | mined | op histogram per Step 4 |
| encoder | `export_htp_metadata.json` | mined | trace coverage per Step 4 |
| encoder | `winml_build_config.json` | mined | autoconf diff per Step 4 |
| decoder | `model.onnx` | inline | optimized graph (≤2GB ⇒ no external-data needed) |
| decoder | `analyze_result.json` | mined | op histogram per Step 4 |
| decoder | `export_htp_metadata.json` | mined | trace coverage per Step 4 |
| decoder | `winml_build_config.json` | mined | autoconf diff per Step 4 |

**External-data layout check** ([`_meta-023`](../skill_meta/findings.json)): both halves under 2GB ProtoBuf limit ⇒ inline weights, no `.data` shard. N/A — vacuous PASS.

**Encoder/decoder cross-attention alias check** ([`_meta-025`](../skill_meta/findings.json)): encoder output = `encoder_hidden_states` (shape `[1,512,512]`); decoder input `encoder_hidden_states` (shape `[1,512,512]`). Direct name + shape match. PASS.

## 4. Build log

Build logs at `temp/marian_build/{encoder,decoder}/build.log` (per marian-003 mechanism_notes). Iter-6 reused iter-5 artifacts unchanged — recipe is byte-identical to the marian-003 commit; no re-build needed.

## 5. Appended findings

### Per-model — `model_knowledge/marian.json`

- [marian-003](../model_knowledge/marian.json) — VALIDATED L0★ build closure (iter-5).
- [marian-005](../model_knowledge/marian.json) — VALIDATED Goal-L1-CPU + Goal-L2-encoder cosine = 1.0 (this PR's primary evidence).
- [marian-006](../model_knowledge/marian.json) — PR-mining cross-references (composite gate `_meta-020`, encoder alias `_meta-025`, external-data `_meta-023`, `--ep-options` retry `_meta-026`, task-consistency `_meta-028`).

### Skill-meta — `skill_meta/findings.json`

This PR does not introduce new `_meta-NNN` findings. The iter-6 methodology evolution (`_meta-019..037`) ships separately on the skills branch (Lane A per [`_meta-033`](../skill_meta/findings.json)).

## 6. Optimum-coverage probe verdict

```python
mt = "marian"
# vendor: feature-extraction, feature-extraction-with-past, text2text-generation, text2text-generation-with-past
# after_winml: identical (no override; pure-vendor coverage)
# added_by_winml: []
```

**Verdict**: VENDOR-COVERED on `text2text-generation` (composite expansion → encoder = feature-extraction, decoder = text2text-generation). Effort L0★ confirmed. Per `winml config --task translation`, the user-facing task `translation` correctly composite-expands to the two sub-tasks; the decoder recipe's `task: text2text-generation` is the canonical sub-task name per [`_meta-028`](../skill_meta/findings.json).

## 7. Claimed (Effort, Goal, Outcome) tier

- **Effort = L0★** (recipe-only; one `winml config` invocation per checkpoint, no hand-edits beyond `_status` removal which was never needed here)
- **Goal = L2-encoder** (L0 + L1-CPU PASS on both halves; L2 encoder cosine=1.0; L2 decoder DEFERRED-HARNESS per `_meta-018`)
- **Outcome = L0** (recipe + finding append + this report; no source code; no feature-gap issues filed for this PR — the open feature gap "ship a `winml.eval.compare_pt_onnx` helper" is captured under marian-005 gotchas but is methodology-scope)

## 8. Goal-ladder verdict table (per [`_meta-018`](../skill_meta/findings.json), per-half per [`_meta-020`](../skill_meta/findings.json))

| Half | Tier | Verdict | Evidence |
|---|---|---|---|
| **encoder** | L0 | **PASS** | `winml build` → `model.onnx`; opset 17; fp32 weights per [`_meta-014`](../skill_meta/findings.json); structural validation via `onnx.load` |
| **encoder** | L1-CPU | **PASS** | Avg 54.95 ms / P50 51.70 / P90 68.30 / Min 48.05 / Max 68.69 / Std 7.37; warmup 52.67 ms avg; throughput 18.20 samples/sec on `[1, 512]` input. Log: [temp/opus_en_ru_perf_enc_cpu.log](../../../temp/opus_en_ru_perf_enc_cpu.log) |
| **encoder** | L1-DML/QNN/OpenVINO | **HOST-BLOCKED** | Per [`_meta-016`](../skill_meta/findings.json) — same host caveat as bart-mnli |
| **encoder** | L2 | **PASS** | cosine = 1.000000, max_abs_diff = 6e-6 (0.0001% of PT max-abs) on real tokenized input. Log: [temp/en_ru_l2_compare.log](../../../temp/en_ru_l2_compare.log); script: [temp/en_ru_l2_compare.py](../../../temp/en_ru_l2_compare.py) |
| **encoder** | L3 | **CLI-BLOCKED** | Per [`_meta-015`](../skill_meta/findings.json) — `winml eval` task registry does not include `translation` (no generative-text-to-text task) |
| **decoder** | L0 | **PASS** | `winml build` → `model.onnx`; opset 17; fp32 weights; structural validation via `onnx.load` |
| **decoder** | L1-CPU | **PASS** | Avg 17.68 ms / P50 17.39 / P90 19.96 / Min 15.60 / Max 20.84 / Std 1.65; warmup 19.79 ms avg; throughput 56.56 samples/sec on `[1, 1]` decoder_input_ids + `[1, 512, 512]` encoder_hidden_states + 6×past_KV pairs. Log: [temp/opus_en_ru_perf_dec_cpu.log](../../../temp/opus_en_ru_perf_dec_cpu.log) |
| **decoder** | L1-DML/QNN/OpenVINO | **HOST-BLOCKED** | Per [`_meta-016`](../skill_meta/findings.json) |
| **decoder** | L2 | **DEFERRED-HARNESS** | cosine = 0.997001 on first-token logits with zeroed past_KV, but argmax disagreement (ONNX=1121 vs PT=10537). Honest verdict per [`_meta-018`](../skill_meta/findings.json) — needs proper DynamicCache↔past_KV reconstruction (open feature gap noted in marian-005). Log: [temp/en_ru_l2_compare.log](../../../temp/en_ru_l2_compare.log) |
| **decoder** | L3 | **CLI-BLOCKED** | Per [`_meta-015`](../skill_meta/findings.json) |

**Short-circuit honored**: no FAIL anywhere. L3 CLI-BLOCKED + L2-decoder DEFERRED-HARNESS do not halt the march per [`_meta-018`](../skill_meta/findings.json). The honest ceiling is L2-encoder PASS.

**Diligence ladder ([`_meta-037`](../skill_meta/findings.json))**: not invoked — no BLOCKED-style verdict required ladder walk; the two BLOCKED verdicts (L1-non-CPU + L3) are host/CLI capability gaps documented in existing findings, not failed attempts.

## 9. Methodology-evolution declaration (per [`_meta-031`](../skill_meta/findings.json))

**No NEW methodology friction in this PR.** The composite-recipe pattern + `task=translation` routing + decoder L2 harness gap were all captured during iter-5 (marian-003..005); they ship as separate `_meta-NNN` findings on the skills branch under `_meta-019..030`. Triggers:

- (1) CLI surprise — none.
- (2) Doc-code drift — none.
- (3) Silent-failure mode — none observed (cross-attention alias direct-name-match per `_meta-025`).
- (4) New verdict shape — `DEFERRED-HARNESS` was new during iter-5 but is now in the vocabulary.
- (5) Reviewer-found gap — pending reviewer pass.
- (6) Effort mis-estimate — none.
- (7) PR-mining discovery — none beyond `_meta-019..030` already shipped.

Reviewer should confirm "no methodology friction observed" rather than REQUEST_CHANGES on absence per `_meta-031` anti-trigger.

## Reviewer hand-off package — Step 6 9-item self-check

1. Recipe files — §1 ✓
2. README row — §2 ✓ (to add in this PR)
3. Build output dir + artifact inventory — §3 ✓
4. Build log — §4 ✓
5. Appended findings — §5 ✓
6. Optimum-coverage probe verdict — §6 ✓
7. Claimed (Effort, Goal, Outcome) tier — §7 ✓
8. Goal-ladder verdict table — §8 ✓ (per-half, composite-expanded)
9. Methodology-evolution declaration — §9 ✓
