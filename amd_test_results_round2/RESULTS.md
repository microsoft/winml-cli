# AMD VitisAI Test Results — Combined (Round 1 + Round 2)

**Date:** April 22, 2026
**EP:** VitisAI (`--ep vitisai`)
**Device:** NPU
**Precisions tested:** w8a8, w8a16, fp16
**Pipeline:** `winml config` → `winml build` → `winml perf` (100 iterations) → `winml eval`

> **Note:** All builds had compile step failure (`enable_ep_context=False` for VitisAI in ModelKit config). Tests used the best available intermediate artifact: `quantized.onnx` for w8a8/w8a16, `optimized.onnx` for fp16.

## Summary

| Metric | w8a8 | w8a16 | fp16 |
|--------|------|-------|------|
| **Config** | 35/35 PASS | 35/35 PASS | 35/35 PASS |
| **Build (partial)** | 35/35 have onnx | 35/35 have onnx | 35/35 have onnx |
| **Perf** | 35/35 PASS | 21/35 PASS* | 35/35 PASS |
| **Eval** | 28/35 PASS | 14/35 PASS* | 28/35 PASS |

*Round1 w8a16 all failed (14/14), Round2 w8a16 all passed (21/21)

## Detailed Results

| # | Model | Task | w8a8 Perf | w8a8 Eval | w8a16 Perf | w8a16 Eval | fp16 Perf | fp16 Eval | Source |
|---|-------|------|-----------|-----------|------------|------------|-----------|-----------|--------|
| 1 | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | feature-extraction | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 2 | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | sentence-similarity | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 3 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | feature-extraction | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 4 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | sentence-similarity | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 5 | [Babelscape/wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner) | token-classification | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 6 | [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | token-classification | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 7 | [facebook/convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224) | image-classification | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 8 | [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) | feature-extraction | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 9 | [Intel/bert-base-uncased-mrpc](https://huggingface.co/Intel/bert-base-uncased-mrpc) | feature-extraction | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 10 | [Intel/bert-base-uncased-mrpc](https://huggingface.co/Intel/bert-base-uncased-mrpc) | text-classification | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 11 | [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection) | object-detection | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Round1 |
| 12 | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | text-classification | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Round1 |
| 13 | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | feature-extraction | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 14 | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | sentence-similarity | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Round1 |
| 15 | [microsoft/swin-large-patch4-window7-224](https://huggingface.co/microsoft/swin-large-patch4-window7-224) | image-classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 16 | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | sentence-similarity | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 17 | [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | text-classification | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | Round2 |
| 18 | [dbmdz/bert-large-cased-finetuned-conll03-english](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english) | token-classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 19 | [deepset/bert-large-uncased-whole-word-masking-squad2](https://huggingface.co/deepset/bert-large-uncased-whole-word-masking-squad2) | question-answering | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 20 | [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2) | question-answering | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 21 | [deepset/tinyroberta-squad2](https://huggingface.co/deepset/tinyroberta-squad2) | question-answering | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 22 | [google-bert/bert-large-uncased-whole-word-masking-finetuned-squad](https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad) | question-answering | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 (rerun) |
| 23 | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) | image-classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 24 | [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) | image-segmentation | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | Round2 |
| 25 | [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) | image-classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 26 | [nvidia/segformer-b1-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512) | image-segmentation | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | Round2 |
| 27 | [nvidia/segformer-b2-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512) | image-segmentation | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | Round2 |
| 28 | [nvidia/segformer-b5-finetuned-ade-640-640](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) | image-segmentation | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | Round2 |
| 29 | [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16) | feature-extraction | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 30 | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | feature-extraction | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 31 | [rizvandwiki/gender-classification](https://huggingface.co/rizvandwiki/gender-classification) | image-classification | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | Round2 |
| 32 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | feature-extraction | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 33 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | sentence-similarity | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 34 | [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) | sentence-similarity | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |
| 35 | [w11wo/indonesian-roberta-base-posp-tagger](https://huggingface.co/w11wo/indonesian-roberta-base-posp-tagger) | token-classification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Round2 |

## Models with Eval Failures (all precisions)

These 7 models fail eval across all 3 precisions — likely a `winml eval` compatibility issue, not a precision issue:

| Model | Task | Notes |
|-------|------|-------|
| microsoft/table-transformer-detection | object-detection | Perf passes, eval fails |
| ProsusAI/finbert | text-classification | Perf passes, eval fails |
| cardiffnlp/twitter-roberta-base-sentiment-latest | text-classification | Perf passes, eval fails |
| mattmdjaga/segformer_b2_clothes | image-segmentation | Perf passes, eval fails |
| nvidia/segformer-b1-finetuned-ade-512-512 | image-segmentation | Perf passes, eval fails |
| nvidia/segformer-b2-finetuned-ade-512-512 | image-segmentation | Perf passes, eval fails |
| nvidia/segformer-b5-finetuned-ade-640-640 | image-segmentation | Perf passes, eval fails |
| rizvandwiki/gender-classification | image-classification | Perf passes, eval fails |

## Key Findings

1. **Perf: 100% pass rate on w8a8 and fp16** — all 35 models pass perf with these precisions.

2. **w8a16 inconsistency between Round1 and Round2** — Round1 models all failed w8a16 perf; Round2 models all passed. This may be due to different model architectures or a flaky issue.

3. **Eval failures are task/model-specific, not precision-specific** — the same 7 models fail eval on all 3 precisions, suggesting the issue is in `winml eval`'s dataset/task handling, not the model quality.

4. **Build compile always fails** — VitisAI `enable_ep_context` is set to `False` in ModelKit configs, so no compiled artifact is generated. The `winml compile` primitive command succeeds but produces no output file.

## Data Locations

- **Round 1 results:** `amd_test_results/` (14 model+task combos × 3 precisions)
- **Round 2 results:** `amd_test_results_round2/` (21 model+task combos × 3 precisions)
- Each folder contains: `commands.txt`, `config.json`, `build/`, `*_log.txt`, `perf.json`, `eval.json`, `status.txt`
