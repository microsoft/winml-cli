# Model Scale — microsoft/ModelKit
**Period**: 2026-03-23 to 2026-04-08
**Generated**: 2026-04-08

Model scale tracks whether every model in a given family passes `winml perf` end-to-end across target EPs. Each model-family tracking issue (#118–#140) represents a definition-of-done: all models in that family must pass `winml perf` on the target EP(s).

---

## E2E Coverage Context

**Snapshot dates**: QNN/OV/VitisAI 0403 (updated from 0327)
**Total model-task combinations tested per EP**: 216

| EP | PASS | FAIL | Pass Rate | Delta vs 0327 |
|----|------|------|-----------|---------------|
| QNNExecutionProvider_NPU | 137 | 79 | 63.4% | +3.7pp |
| OpenVINOExecutionProvider_NPU | 128 | 88 | 59.3% | -0.4pp |
| VitisAIExecutionProvider_NPU | 109 | 107 | 50.5% | -5.1pp |
| **All Three EPs** | **98** | — | **45.4%** | **+1.0pp** |

Primary bottleneck: VitisAI at 50.5% (down 5.1pp from 55.6%). QNN improved significantly (+3.7pp), but VitisAI regression offsets overall progress.

Zero-coverage task categories (no model-task combination passes all three EPs): document-question-answering, mask-generation, summarization, text-generation, translation, visual-question-answering.

---

## Infrastructure — Model Registry and E2E Framework

| # | Title | Milestone | Owner | Status | Notes |
|---|-------|-----------|-------|--------|-------|
| #97 | P0-MODEL-010: Model Registry Database | `202604 Release` | Qiong Wu | Open | Central model registry backing `winml hub`; partially unblocked by #117 (wmk hub CLI, closed 2026-04-01) |
| #150 | P1-MODEL-007: E2E Test Framework — Model x EP Matrix | `202604 Release` | Qiong Wu | Open | Framework for tracking pass/fail per model-task-EP combination |
| #151 | P1-MODEL-008: Metrics Dashboard — Model x EP Coverage | `202604 Release` | Qiong Wu | Open | Dashboard to surface the E2E coverage data |

---

## Model-Family Tracking Issues (no milestone or 202604)

Each issue below is labeled `model / task scale`. Completion means all models in the family pass `winml perf` on the target EP(s). All issues were created 2026-03-31 and are currently open.

Note: #122 (xlm-roberta) moved to the `202604` milestone in the current data; all others remain without a milestone.

### Open — Model-Family Tracking

| # | Family / Task | Owner | Status | Coverage Context |
|---|---------------|-------|--------|-----------------|
| #118 | segformer / image-segmentation | Unassigned | Open | 9 segformer models pass all 3 EPs in image-segmentation (unchanged) |
| #119 | t5 / summarization + translation | Unassigned | Open | Zero-coverage for summarization and translation tasks — all 3 EPs fail |
| #120 | bart + mbart / summarization + text-classification + zero-shot-classification | Unassigned | Open | Summarization zero-coverage; zero-shot-classification has some passes |
| #121 | marian + m2m_100 / translation | Unassigned | Open | Translation is a zero-coverage task across all 3 EPs |
| #122 | xlm-roberta / fill-mask + text-classification + feature-extraction + sentence-similarity + token-classification | Yue Sun (ssss141414) | Open | 10 fill-mask models pass all 3 EPs including xlm-roberta-base and xlm-roberta-large; moved to 202604 milestone |
| #123 | depth_anything + dpt + zoedepth / depth-estimation | Unassigned | Open | Intel/dpt-hybrid-midas passes all 3 EPs; dpt-large and zoedepth pass QNN+VitisAI only (OV bottleneck) |
| #124 | sam + sam2 / mask-generation | Charles Zhang | Open | mask-generation is zero all-3 coverage; SAM models pass QNN+VitisAI only (OV bottleneck) |
| #125 | qwen2 + qwen3 / text-generation | Unassigned | Open | text-generation is a zero-coverage task across all 3 EPs |
| #126 | clip / zero-shot-classification | Zhenchao Ni | Open | CLIP zero-shot-image-classification: some models pass QNN+OV only (VitisAI bottleneck) |
| #127 | mpnet / fill-mask + feature-extraction + sentence-similarity | Qiong Wu | Open | all-mpnet-base-v2 passes all 3 EPs in fill-mask, feature-extraction, sentence-similarity |
| #128 | deberta + deberta-v2 / text-classification + zero-shot-classification | Unassigned | Open | DeBERTa-v3 variants pass all 3 EPs in zero-shot-classification; OV/VitisAI only for some larger variants |
| #129 | swin / image-classification | Unassigned | Open | swin-large passes OV+VitisAI only (QNN bottleneck) |
| #130 | gpt2 / text-generation | Unassigned | Open | text-generation is a zero-coverage task across all 3 EPs |
| #131 | blip + blip-2 / visual-question-answering | Unassigned | Open | visual-question-answering is zero-coverage; blip image captioning passes all 3 EPs (image-to-text task) |
| #132 | pix2struct + vilt / visual-question-answering | Unassigned | Open | visual-question-answering is a zero-coverage task |
| #133 | vision-encoder-decoder / image-to-text + document-question-answering | Charles Zhang | Open | 7 image-to-text models pass all 3 EPs (trocr-base, blip captioning, vit-gpt2 and others); trocr-large passes OV+VitisAI only |
| #134 | layoutlm + layoutlmv3 / document-question-answering | Unassigned | Open | document-question-answering is a zero-coverage task |
| #135 | distilbert + camembert / question-answering + token-classification | Unassigned | Open | distilbert passes all 3 EPs in question-answering and fill-mask; camembert passes token-classification |
| #136 | siglip / zero-shot-image-classification | Unassigned | Open | siglip-base-patch16-224 passes all 3 EPs; siglip-so400m passes OV+VitisAI only (QNN bottleneck) |
| #137 | siglip_vision_model / image-feature-extraction | Unassigned | Open | Marqo fashionSigLIP passes all 3 EPs in zero-shot-image-classification |
| #138 | dinov2 / image-feature-extraction | Unassigned | Open | dinov2-base, -large, -small all pass all 3 EPs |
| #139 | bert / fill-mask + feature-extraction + document-question-answering | Yue Sun | Open | BERT variants pass all 3 EPs in fill-mask, feature-extraction, question-answering; document-QA is zero-coverage |
| #140 | internlm2 + phi4mm / visual-question-answering | Unassigned | Open | visual-question-answering is a zero-coverage task |

**Total open model-family tracking issues**: 23 of 23

---

## Additional Model Scale Issues (202605 Release)

| # | Title | Milestone | Owner | Status | Notes |
|---|-------|-----------|-------|--------|-------|
| #61 | [Test] Config test coverage for all 166 optimum-onnx architectures | `202605 Release` | Te Zheng | Open | Comprehensive architecture export coverage for test suite |
| #62 | [Test] Fix 6 skipped + 1 xfailed architecture export tests | `202605 Release` | Te Zheng | Open | Resolves skipped/xfailed tests that hide coverage gaps |
| #66 | feat: add built-in model support for Swin2SR | `202605 Release` | Te Zheng | Open | Swin2SR super-resolution built-in model registration |
| #68 | feat: add built-in model support for ESRGAN | `202604 Release` | Te Zheng, Zhenchao Ni | Open | ESRGAN super-resolution built-in model registration |
| #69 | feat: add built-in model support for Whisper | `202605 Release` | Te Zheng | Open | Whisper ASR built-in model registration |
| #70 | feat: add built-in model support for Stable Diffusion | `202605 Release` | Te Zheng | Open | Stable Diffusion image generation built-in model registration |
| #149 | P1-MODEL-003: 20 ISV Models — Architecture Analysis Report | `202605 Release` | Unassigned | Open | Analysis report for 20 ISV partner models |
| #177 | Put built-in model evaluation dataset in eval's prod code | No milestone | Zhenchao Ni | Open | Move eval datasets from ad-hoc location into production code path |
| #178 | Model scale for wmk perf | `202605 Release` | vortex-captain | Open | General model scale expansion for the perf command |
| #179 | Quantization causes big accuracy loss | No milestone | Zhenchao Ni | Open | Accuracy regression observed after quantization on some model families |
| #181 | Model export needs to keep pixel_mask input for detr-like models | No milestone | Te Zheng | Open | Export bug causing pixel_mask input to be dropped for DETR-family models |

### Closed During Period

| # | Title | Milestone | Owner | Closed | Notes |
|---|-------|-----------|-------|--------|-------|
| #67 | feat: add built-in model support for SAM2 | `202604 Release` | Te Zheng, Charles Zhang | 2026-04-07 | SAM2 mask-generation registered as built-in model |

---

## Summary

| Category | Open | Closed |
|----------|------|--------|
| Model-family tracking (#118–#140) | 23 | 0 |
| Model registry / E2E infra (#97, #150, #151) | 3 | 0 |
| Built-in model additions (202604/202605) | 8 | 1 (#67 SAM2, closed 2026-04-07) |
| Additional model scale issues (#177, #178, #179, #181) | 4 | 0 |
| **Total** | **38** | **1** |

The model scale work remains heavily backloaded. No model-family tracking issue has been closed; 15 of the 23 family issues are completely unassigned. E2E coverage reached 45.4% all-EP pass rate (up 1.0pp from 44.4%), driven by QNN improvement (+3.7pp), though partially offset by a VitisAI regression (-5.1pp, now the primary bottleneck at 50.5%). Six task categories remain at zero all-EP coverage. Closing model-family tracking issues will require either increasing VitisAI pass rates for generative/decoder models (text-generation, summarization, visual-QA) or scoping down which task variants are in scope.
