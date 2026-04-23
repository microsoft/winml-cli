# E2E Model Coverage Analysis — Three-EP Intersection
**Report Date**: 2026-04-08
**Data Sources** (combined perf + accuracy per EP):
- QNN: `qnn_report_0403.csv` (QNNExecutionProvider_NPU, snapshot 0403)
- OV: `ov_report_0403.csv` (OpenVINOExecutionProvider_NPU, snapshot 0403)
- VITISAI: `vitisai_report_0403.csv` (VitisAIExecutionProvider_NPU, snapshot 0403)

## 1. Per-EP Summary Statistics

| EP | Total | PASS | FAIL | Pass Rate |
|----|-------|------|------|-----------|
| QNNExecutionProvider_NPU | 216 | 137 | 79 | 63.4% |
| OpenVINOExecutionProvider_NPU | 216 | 128 | 88 | 59.3% |
| VitisAIExecutionProvider_NPU | 216 | 109 | 107 | 50.5% |
| **All Three EPs** | 216 | 98 | — | 45.4% |

**Key observation**: Bottleneck EP is VitisAIExecutionProvider_NPU (109/216 = 50.5% pass rate).

## 2. Models Passing All Three EPs

**98 model-task combinations pass all three EPs.**

### NLP Tasks

#### feature-extraction (9 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| BAAI/bge-large-en-v1.5 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| BAAI/bge-m3 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| openai/clip-vit-base-patch16 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| openai/clip-vit-base-patch32 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| sentence-transformers/all-MiniLM-L6-v2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| sentence-transformers/all-mpnet-base-v2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### fill-mask (10 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| FacebookAI/roberta-base | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| FacebookAI/roberta-large | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| FacebookAI/xlm-roberta-base | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| FacebookAI/xlm-roberta-large | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| distilbert/distilbert-base-uncased | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| google-bert/bert-base-multilingual-cased | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| google-bert/bert-base-multilingual-uncased | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| google-bert/bert-base-uncased | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| sentence-transformers/all-mpnet-base-v2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### question-answering (9 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| ahotrod/electra_large_discriminator_squad2_512 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| deepset/bert-large-uncased-whole-word-masking-squad2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| deepset/roberta-base-squad2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| deepset/tinyroberta-squad2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| distilbert/distilbert-base-cased-distilled-squad | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| distilbert/distilbert-base-uncased-distilled-squad | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| google-bert/bert-large-uncased-whole-word-masking-finetuned-squad | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| monologg/koelectra-small-v2-distilled-korquad-384 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| timpal0l/mdeberta-v3-base-squad2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### sentence-similarity (7 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| BAAI/bge-large-en-v1.5 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| BAAI/bge-m3 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| intfloat/multilingual-e5-large | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| sentence-transformers/all-MiniLM-L6-v2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| sentence-transformers/all-mpnet-base-v2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### text-classification (8 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| BAAI/bge-reranker-base | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| BAAI/bge-reranker-v2-m3 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| cardiffnlp/twitter-roberta-base-sentiment-latest | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| cross-encoder/ms-marco-MiniLM-L4-v2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| cross-encoder/ms-marco-MiniLM-L6-v2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| distilbert/distilbert-base-uncased-finetuned-sst-2-english | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| microsoft/deberta-xlarge-mnli | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| mixedbread-ai/mxbai-rerank-xsmall-v1 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### token-classification (7 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| Isotonic/distilbert_finetuned_ai4privacy_v2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| Jean-Baptiste/camembert-ner-with-dates | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| dbmdz/bert-large-cased-finetuned-conll03-english | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| kredor/punctuate-all | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| obi/deid_roberta_i2b2 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| oliverguhr/fullstop-punctuation-multilang-large | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| w11wo/indonesian-roberta-base-posp-tagger | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### zero-shot-classification (7 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| MoritzLaurer/mDeBERTa-v3-base-mnli-xnli | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| cross-encoder/nli-deberta-v3-small | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| joeddav/xlm-roberta-large-xnli | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| lxyuan/distilbert-base-multilingual-cased-sentiments-student | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

### Computer Vision Tasks

#### depth-estimation (1 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| Intel/dpt-hybrid-midas | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### image-classification (8 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| AdamCodd/vit-base-nsfw-detector | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| Falconsai/nsfw_image_detection | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| amunchet/rorshark-vit-base | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| apple/mobilevit-small | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| dima806/fairface_age_image_detection | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| google/vit-base-patch16-224 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| microsoft/resnet-50 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| rizvandwiki/gender-classification | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### image-feature-extraction (7 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| StanfordAIMI/dinov2-base-xray-224 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| facebook/dino-vitb16 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| facebook/dinov2-base | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| facebook/dinov2-large | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| facebook/dinov2-small | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| google/vit-base-patch16-224-in21k | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| microsoft/rad-dino | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### image-segmentation (9 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| fashn-ai/fashn-human-parser | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| jonathandinu/face-parsing | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| mattmdjaga/segformer_b2_clothes | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| nvidia/segformer-b0-finetuned-ade-512-512 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| nvidia/segformer-b0-finetuned-cityscapes-1024-1024 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| nvidia/segformer-b1-finetuned-ade-512-512 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| nvidia/segformer-b2-finetuned-ade-512-512 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| nvidia/segformer-b5-finetuned-ade-640-640 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| nvidia/segformer-b5-finetuned-cityscapes-1024-1024 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### image-to-text (7 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| Salesforce/blip-image-captioning-base | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| kha-white/manga-ocr-base | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| microsoft/trocr-base-handwritten | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| microsoft/trocr-base-printed | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| microsoft/trocr-large-handwritten | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| microsoft/trocr-large-printed | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| nlpconnect/vit-gpt2-image-captioning | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### masked-lm (1 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| google-bert/bert-base-multilingual-cased | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### object-detection (5 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| TahaDouaji/detr-doc-table-detection | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| facebook/detr-resnet-50 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| hustvl/yolos-small | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| microsoft/table-transformer-structure-recognition | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| valentinafeve/yolos-fashionpedia | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

#### zero-shot-image-classification (3 models)

| model_id | QNN | OV | VITISAI | Accuracy | Delta |
|----------|---|---|---|----------|-------|
| Marqo/marqo-fashionSigLIP | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| google/siglip-base-patch16-224 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |
| google/siglip-so400m-patch14-384 | ✅ PASS | ✅ PASS | ✅ PASS | N/A |  |

## 3. Pass Rate by Task Category

| Task | All-3 Pass | Accuracy PASS | Accuracy REGRESSION | N/A |
|------|-----------|---------------|---------------------|-----|
|  ❌ | 0 | 0 | 0 | 0 |
| depth-estimation | 1 | 0 | 0 | 1 |
| document-question-answering ❌ | 0 | 0 | 0 | 0 |
| feature-extraction | 9 | 0 | 0 | 9 |
| fill-mask | 10 | 0 | 0 | 10 |
| image-classification | 8 | 0 | 0 | 8 |
| image-feature-extraction | 7 | 0 | 0 | 7 |
| image-segmentation | 9 | 0 | 0 | 9 |
| image-to-text | 7 | 0 | 0 | 7 |
| mask-generation ❌ | 0 | 0 | 0 | 0 |
| masked-lm | 1 | 0 | 0 | 1 |
| object-detection | 5 | 0 | 0 | 5 |
| question-answering | 9 | 0 | 0 | 9 |
| sentence-similarity | 7 | 0 | 0 | 7 |
| summarization ❌ | 0 | 0 | 0 | 0 |
| text-classification | 8 | 0 | 0 | 8 |
| text-generation ❌ | 0 | 0 | 0 | 0 |
| token-classification | 7 | 0 | 0 | 7 |
| translation ❌ | 0 | 0 | 0 | 0 |
| visual-question-answering ❌ | 0 | 0 | 0 | 0 |
| zero-shot-classification | 7 | 0 | 0 | 7 |
| zero-shot-image-classification | 3 | 0 | 0 | 3 |

**Zero-coverage tasks**: , document-question-answering, mask-generation, summarization, text-generation, translation, visual-question-answering

## 4. Accuracy Summary (all-3 EP PASS models)

| Metric | Count |
|--------|-------|
| PASS | 0 |
| REGRESSION | 0 |
| EVAL_ERROR | 0 |
| N/A | 98 |

## 5. Architecture Pattern Analysis

_[Encoder-only Transformer / ViT families tend to pass; decoder / generative models fail due to dynamic shapes and attention op coverage.]_

## 6. Notable Gaps and Failure Patterns

_[Fill in from manual review of FAIL rows and zero-coverage tasks above.]_

## 7. Models Passing Exactly 2 EPs (Partial Coverage)

### qnn+ov only (vitisai bottleneck) (28 combinations)

- BAAI/bge-base-en-v1.5 / feature-extraction
- BAAI/bge-base-en-v1.5 / sentence-similarity
- BAAI/bge-small-en-v1.5 / feature-extraction
- BAAI/bge-small-en-v1.5 / sentence-similarity
- Babelscape/wikineural-multilingual-ner / token-classification
- Intel/bert-base-uncased-mrpc / feature-extraction
- Intel/bert-base-uncased-mrpc / text-classification
- PekingU/rtdetr_r101vd_coco_o365 / object-detection
- PekingU/rtdetr_v2_r18vd / object-detection
- ProsusAI/finbert / text-classification
- StanfordAIMI/stanford-deidentifier-base / token-classification
- dslim/bert-base-NER / token-classification
- facebook/convnext-tiny-224 / image-classification
- facebook/detr-resnet-50 / feature-extraction
- facebook/dino-vits16 / image-feature-extraction
- google-bert/bert-base-multilingual-cased / feature-extraction
- laion/CLIP-ViT-B-32-laion2B-s34B-b79K / zero-shot-image-classification
- microsoft/beit-base-patch16-224-pt22k-ft22k / image-classification
- microsoft/table-transformer-detection / object-detection
- microsoft/table-transformer-structure-recognition-v1.1-all / object-detection
- openai/clip-vit-base-patch16 / zero-shot-image-classification
- openai/clip-vit-base-patch32 / zero-shot-image-classification
- openai/clip-vit-large-patch14 / zero-shot-image-classification
- openai/clip-vit-large-patch14-336 / zero-shot-image-classification
- patrickjohncyh/fashion-clip / zero-shot-image-classification
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 / feature-extraction
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 / sentence-similarity
- tau/splinter-base / question-answering

### qnn+vitisai only (ov bottleneck) (6 combinations)

- Intel/dpt-large / depth-estimation
- Intel/zoedepth-nyu-kitti / depth-estimation
- facebook/sam2-hiera-large / mask-generation
- facebook/sam2.1-hiera-small / mask-generation
- facebook/sam2.1-hiera-tiny / mask-generation
- laion/CLIP-ViT-H-14-laion2B-s32B-b79K / zero-shot-image-classification

### ov+vitisai only (qnn bottleneck) (1 combinations)

- microsoft/swin-large-patch4-window7-224 / image-classification

## 8. Implications for Milestone Target

_[Fill in based on team milestone targets and current pace.]_

## 9. Summary

| Metric | Value |
|--------|-------|
| Total model-task combinations tested | 216 |
| Combinations passing all three EPs (perf) | 98 (45.4% of 216) |
| Of those: accuracy PASS | 0 |
| Of those: accuracy REGRESSION | 0 |
| Of those: accuracy N/A | 98 |
| Primary bottleneck EP | VitisAIExecutionProvider_NPU (50.5%) |
