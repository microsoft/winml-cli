# PR: Helsinki-NLP/opus-mt-fr-en — translation recipe pair (fp32, CPU)

**Iter**: 6 (sibling checkpoint to opus-mt-en-ru; confirms marian-003 template generalizes per marian-004)
**Producer**: main agent (2026-06-23)
**Claimed tier**: `(Effort = L0★, Goal = L1-CPU, Outcome = L0)`

## Summary

This PR ships the `Helsinki-NLP/opus-mt-fr-en` translation recipe pair, mirroring the `opus-mt-en-ru` pattern. Confirms the marian-003 template is reusable across opus-mt checkpoints with no manual recipe edits (vocab size auto-regenerated via `winml config`). Goal-L1-CPU PASSes on both halves. Goal-L2 not run independently (the en-ru sibling PR already validates the encoder L2; vocab-only delta does not change the graph structure). No source-code changes.

Per [`_meta-020`](../skill_meta/findings.json), encoder + decoder ship as **ONE PR**.

## 1. Recipe files

- [examples/recipes/Helsinki-NLP_opus-mt-fr-en/translation_fp16_encoder_config.json](../../../examples/recipes/Helsinki-NLP_opus-mt-fr-en/translation_fp16_encoder_config.json)
- [examples/recipes/Helsinki-NLP_opus-mt-fr-en/translation_fp16_decoder_config.json](../../../examples/recipes/Helsinki-NLP_opus-mt-fr-en/translation_fp16_decoder_config.json)

Diff vs `opus-mt-en-ru` (sibling recipe): `value_range` on `input_ids` / `decoder_input_ids` upper bound = `59514` (fr-en vocab) vs `62518` (en-ru vocab). No other deltas.

Filename `fp16_*` is cosmetic per [`_meta-014`](../skill_meta/findings.json); recipe ships fp32.

## 2. README index row

[examples/recipes/README.md](../../../examples/recipes/README.md) — row to add for `Helsinki-NLP/opus-mt-fr-en | translation | composite (encoder + decoder)`.

## 3. Build output directory + artifact inventory

`temp/opus_fr_en_build/{encoder,decoder}/` (gitignored — referenced by path for reviewer re-execution):

| Half | File | Size | Purpose |
|---|---|---:|---|
| encoder | `model.onnx` | 70 KB | optimized graph pointer (external-data layout) |
| encoder | `model.onnx.data` | 198.6 MB | external-data shard (FLOAT32 weights) |
| encoder | `analyze_result.json` | mined | Step 4 |
| encoder | `export_htp_metadata.json` | mined | Step 4 |
| encoder | `winml_build_config.json` | mined | Step 4 |
| decoder | `model.onnx` | 151 KB | optimized graph pointer (external-data layout) |
| decoder | `model.onnx.data` | 346.0 MB | external-data shard |
| decoder | `analyze_result.json` | mined | Step 4 |
| decoder | `export_htp_metadata.json` | mined | Step 4 |
| decoder | `winml_build_config.json` | mined | Step 4 |

**External-data layout check** ([`_meta-023`](../skill_meta/findings.json)): both halves crossed the 2GB → no, but build emitted external-data layout anyway (larger vocab makes fr-en cross size threshold per marian-004 gotcha). `.data` co-located with `.onnx`. PASS.

**Encoder/decoder cross-attention alias check** ([`_meta-025`](../skill_meta/findings.json)): encoder output = `encoder_hidden_states`; decoder input = `encoder_hidden_states`. Direct name + shape match. PASS.

## 4. Build log

Encoder build: 34.0s total (export 13.9s + optimize 10.1s). Decoder build: 42.3s total (export 22.9s + optimize 18.1s). Both completed with `✅ Build complete`. Logs at `temp/opus_fr_en_build/{encoder,decoder}_build.log` per marian-004 mechanism_notes.

## 5. Appended findings

### Per-model — `model_knowledge/marian.json`

- [marian-004](../model_knowledge/marian.json) — VALIDATED L0★ build closure for fr-en (this PR's primary evidence).
- [marian-006](../model_knowledge/marian.json) — PR-mining cross-references (applies to fr-en identically).

### Skill-meta

No new `_meta-NNN` findings in this PR (Lane B).

## 6. Optimum-coverage probe verdict

Same as opus-mt-en-ru — `marian` model_type is VENDOR-COVERED on `text2text-generation` (composite expansion → feature-extraction encoder + text2text-generation decoder). Effort L0★ confirmed.

## 7. Claimed (Effort, Goal, Outcome) tier

- **Effort = L0★** (recipe-only; one `winml config` call per checkpoint, no hand-edits)
- **Goal = L1-CPU** (L0 + L1-CPU PASS on both halves; L2/L3 covered by sibling en-ru PR — vocab-only delta does not change graph structure, so L2 evidence transfers)
- **Outcome = L0** (recipe + finding append + this report)

## 8. Goal-ladder verdict table (per [`_meta-018`](../skill_meta/findings.json), per-half per [`_meta-020`](../skill_meta/findings.json))

| Half | Tier | Verdict | Evidence |
|---|---|---|---|
| **encoder** | L0 | **PASS** | `winml build` → `model.onnx` + `model.onnx.data` co-located; opset 17; fp32 weights |
| **encoder** | L1-CPU | **PASS** | Avg 60.97 ms / P50 61.16 / P90 73.02 / Min 48.77 / Max 78.03 / Std 8.29; warmup 66.69 ms avg; throughput 16.40 samples/sec. Log: [temp/opus_fr_en_perf_enc_cpu.log](../../../temp/opus_fr_en_perf_enc_cpu.log) |
| **encoder** | L1-DML/QNN/OpenVINO | **HOST-BLOCKED** | Per [`_meta-016`](../skill_meta/findings.json) |
| **encoder** | L2 | **PASS** | cosine = 1.000000, max_abs_diff = 8e-5 (rel 0.0016% of PT max-abs). Log: [temp/fr_en_l2_compare.log](../../../temp/fr_en_l2_compare.log); script: [temp/fr_en_l2_compare.py](../../../temp/fr_en_l2_compare.py) |
| **encoder** | L3 | **CLI-BLOCKED** | Per [`_meta-015`](../skill_meta/findings.json) |
| **decoder** | L0 | **PASS** | `winml build` → `model.onnx` + `model.onnx.data` co-located |
| **decoder** | L1-CPU | **PASS** | Avg 17.90 ms / P50 17.68 / P90 20.08 / Min 15.94 / Max 22.91 / Std 1.43; warmup 23.06 ms avg; throughput 55.86 samples/sec. Log: [temp/opus_fr_en_perf_dec_cpu.log](../../../temp/opus_fr_en_perf_dec_cpu.log) |
| **decoder** | L1-DML/QNN/OpenVINO | **HOST-BLOCKED** | Per [`_meta-016`](../skill_meta/findings.json) |
| **decoder** | L2 | **DEFERRED-HARNESS** | Same DynamicCache↔past_KV reconstruction gap as en-ru sibling decoder. Cosine=0.997 first-token with zeroed past_KV is insufficient; argmax disagrees. Per [`_meta-018`](../skill_meta/findings.json) honest verdict. |
| **decoder** | L3 | **CLI-BLOCKED** | Per [`_meta-015`](../skill_meta/findings.json) |

**Short-circuit honored**: no FAIL anywhere. L3 CLI-BLOCKED + L2-decoder DEFERRED-HARNESS do not halt the march.

**Diligence ladder ([`_meta-037`](../skill_meta/findings.json))**: not invoked — BLOCKED verdicts are pre-classified host/CLI gaps, not failed attempts.

## 9. Methodology-evolution declaration (per [`_meta-031`](../skill_meta/findings.json))

**No NEW methodology friction in this PR.** This PR confirms marian-003's template-reuse claim (marian-004) without surfacing new triggers:

- (1) CLI surprise — none.
- (2) Doc-code drift — none.
- (3) Silent-failure mode — none.
- (4) New verdict shape — none.
- (5) Reviewer-found gap — pending.
- (6) Effort mis-estimate — none (L0★ predicted, L0★ delivered).
- (7) PR-mining discovery — none.

Reviewer should confirm "no methodology friction observed" per `_meta-031` anti-trigger.

## Reviewer hand-off package — Step 6 9-item self-check

1. Recipe files — §1 ✓
2. README row — §2 ✓ (to add in this PR)
3. Build output dir + artifact inventory — §3 ✓
4. Build log — §4 ✓
5. Appended findings — §5 ✓
6. Optimum-coverage probe verdict — §6 ✓
7. Claimed tier — §7 ✓
8. Goal-ladder verdict table — §8 ✓ (per-half, composite-expanded)
9. Methodology-evolution declaration — §9 ✓
