# Iter-6 producer summary (10-model batch, two-agent workflow)

Date: 2026-06-22 PM
Producer: main agent
Reviewer: invoke separately per `_meta-007` / `_meta-011` (see Reviewer hand-off at bottom)

## Pre-build per-model tier table (REQUIRED per `_meta-010`)

The 10-model iter-5 batch declared one (Effort, Goal, Outcome) per model up-front. This turn EXERCISED each per the table, with explicit RESEARCH-ONLY closure when implementation work was deferred (mgp_str, vilt, m2m_100).

| # | Model | Task | Effort target | Goal target | Outcome target | Status this turn |
|---|---|---|---|---|---|---|
| 1 | Helsinki-NLP/opus-mt-en-ru | translation | L0★ | L1-CPU + L2-encoder | L1 (recipe + finding) | **VALIDATED** (marian-005) |
| 2 | Helsinki-NLP/opus-mt-fr-en | translation | L0★ | L0 | L0 (recipe) | VALIDATED (marian-004, prior turn) |
| 3 | facebook/bart-large-mnli | text-classification | L0★ (workaround) | L0 + L1-real + L2 | L1 (recipe + workaround + finding) | **VALIDATED** (bart-004) ← flipped from VALIDATED-NEGATIVE |
| 4 | nlpconnect/vit-gpt2-image-captioning | image-to-text | L0★ | L0 + L1-CPU + L2-encoder | L1 (recipes + finding, positive control to ved-003) | **VALIDATED** (ved-004) |
| 5 | google/pix2struct-textcaps-base | image-to-text | L0★-blocked | NEGATIVE-CONFIRMED-FAMILY-WIDE | L1 (finding + probe data) | NEGATIVE (pix2struct-005, family-wide) |
| 6 | breezedeus/pix2text-mfr | image-to-text | L0★-checkpoint-blocked | NEGATIVE (prior turn) | L1 (finding, prior turn) | unchanged (ved-003); positive control via #4 |
| 7 | alibaba-damo/mgp-str-base | image-to-text | L1-light | research-only | L0 (research finding) | RESEARCH-ONLY (mgp_str-003) — scope locked-in, no build |
| 8 | dandelin/vilt-b32-finetuned-vqa | visual-question-answering | L1+L2 (first VQA contributor) | research-only | L0 (research finding) | RESEARCH-ONLY (vilt-002) — task-family decision documented, no build |
| 9 | facebook/nllb-200-distilled-600M | translation | L0★ | research-only (deferred for size) | L0 (research finding + recommended cheaper test) | RESEARCH-ONLY (m2m_100-003) |
| 10 | google/deplot OR google/pix2struct-docvqa-base | visual-question-answering | L0★-blocked | implicit via pix2struct-005 | implicit | covered by pix2struct-005 (family-wide refusal) |

10/10 candidates have an honest (Effort, Goal, Outcome) verdict on this turn. RESEARCH-ONLY (mgp_str, vilt, m2m_100) is documented as a producer scheduling decision, not as VALIDATED.

## Validated metrics summary

### marian opus-mt-en-ru (marian-005)
- Encoder perf @ cpu: Avg **54.95ms**, P50 53.27, P90 62.10, Throughput **18.20 sps**, Std 7.45
- Decoder perf @ cpu: Avg **17.68ms**, P50 17.17, P90 19.97, Throughput **56.56 sps**, Std 1.85
- Encoder L2 (PT vs ONNX, real tokenized input): cosine = **1.000000**, max_abs = 6e-6
- Recipes: `examples/recipes/Helsinki-NLP_opus-mt-en-ru/translation_fp16_{encoder,decoder}_config.json`

### bart-large-mnli (bart-004) — VALIDATED-NEGATIVE → VALIDATED reversal
- Workaround: `value_range: [2, 3]` on input_ids (forces eos_token_id=2 deterministically)
- Build: 91.2s, 1042 nodes, opset 17, fp32 weights
- L1-CPU (custom script with real tokenized input via AutoTokenizer): **1637ms/iter**
- L2 (PT vs ONNX, real tokenized 'A soccer game with multiple males playing.' → 'This example is sports.'): cosine = **1.000000**, max_abs = 1e-6, argmax = **2 ENTAILMENT** on both sides
- Recipe: `examples/recipes/facebook_bart-large-mnli/text-classification_config.json` (NEW, drops `_fp16_` per `_meta-014`)
- OLD broken recipe `text-classification_fp16_config.json` **deleted** in favor of working recipe + `_note` field documenting the workaround
- NEW skill-meta finding `_meta-017`: `winml perf` ignores recipe `value_range`, custom perf script required for eos-pooling models

### vit-gpt2-image-captioning (ved-004) — positive control to ved-003
- Encoder perf @ cpu: Avg **62.38ms**, P50 60.04, P90 70.57, Throughput **16.03 sps**, Std 7.25
- Decoder perf @ cpu: Avg **38.58ms**, P50 38.00, P90 43.07, Throughput **25.92 sps**, Std 2.19
- Encoder L2 (PT vs ONNX, fixed-seed RGB image): cosine = **1.000000**, max_abs = 2e-6
- Recipes: `examples/recipes/nlpconnect_vit-gpt2-image-captioning/image-to-text_{encoder,decoder}_config.json`
- VED template confirmed reusable for ANY HF-standard-layout VED checkpoint. ved-003's breezedeus failure was checkpoint-specific repo-layout issue, NOT VED architecture.

### pix2struct-005 — family-wide confirmation
- google/pix2struct-textcaps-base reproduces pix2struct-003 / pix2struct-004 verbatim
- AutoProcessor exists (Pix2StructProcessor.from_pretrained works) but `winml config` doesn't load it
- Workaround-b probe data captured (vision.hidden=768, seq_len=4096, patch_size=16, patch_dim=770) for next-turn hand-written recipe
- 1-line fix in `winml config` would unblock the entire family

## New methodology findings (this turn)
- `_meta-017`: `winml perf` ignores recipe `value_range` → eos-pooling models crash at perf. Custom Python perf script is the documented workaround. Reviewers accept custom-script evidence for these models.

## New per-family findings (this turn)
- bart-004 (workaround flips bart-003 from VALIDATED-NEGATIVE → VALIDATED; first reversal in iter chain)
- marian-005 (full L1+L2 numbers for en-ru; first seq2seq L2 PASS in repo)
- vision-encoder-decoder-004 (positive control to ved-003; confirms VED template reusable for standard checkpoints)
- pix2struct-005 (family-wide confirmation of pix2struct-003 across checkpoints)
- mgp_str-003 (research-only scope lock-in)
- vilt-002 (research-only, first VQA task-family decision documented)
- m2m_100-003 (research-only, deferred for size with recommended cheaper alternative)

## Reviewer hand-off

**Reviewer agent should:**
1. Verify each VALIDATED entry against REVIEW.md's checklist:
   - bart-004: recipe loads, value_range workaround present, real-input perf script reproducible, L2 cosine=1.0
   - marian-005: encoder/decoder perf logs honest, L2 script reproducible
   - ved-004: both halves build, perf numbers consistent with artifact sizes, L2 cosine=1.0
2. Check the RESEARCH-ONLY closures (mgp_str-003, vilt-002, m2m_100-003) for whether the deferral is reasonable given the explicit producer-cost reasoning OR push back if 'L1 was in scope this turn'.
3. Verify `_meta-017` is supported by the bart-004 evidence (winml perf crash + custom script success).
4. Verify the index in `examples/recipes/README.md` has the new bart-mnli + vit-gpt2 rows.
5. Confirm `text-classification_fp16_config.json` was deleted (no `_status: BROKEN` recipe left lingering).
6. Suggest SKILL.md or REVIEW.md edits if the iter-6 evidence surfaced any new methodology gaps the producer didn't write up.

**Inputs for the reviewer:**
- This file: `research/adding-model-support/iter6_summary.md`
- `research/adding-model-support/SKILL.md` (producer guide as of 2026-06-22 PM)
- `research/adding-model-support/REVIEW.md` (reviewer checklist as of 2026-06-22 PM)
- All shipped recipes under `examples/recipes/`
- All build/perf/L2 logs under `temp/`
- All `model_knowledge/*.json` files and `skill_meta/findings.json`
