# Builtin Models (perf-pass × eval-supported)

Models that:
1. Appear in the external 57 (model, task) perf list, AND
2. Are eval-supported (present in `scripts/e2e_eval/testsets/models_with_acc.json`).

Total: **42** (model, task) tuples.

| Model | Task |
|---|---|
| BAAI/bge-large-en-v1.5 | sentence-similarity |
| BAAI/bge-small-en-v1.5 | sentence-similarity |
| FacebookAI/roberta-base | fill-mask |
| FacebookAI/xlm-roberta-base | fill-mask |
| Isotonic/distilbert_finetuned_ai4privacy_v2 | token-classification |
| ahotrod/electra_large_discriminator_squad2_512 | question-answering |
| apple/mobilevit-small | image-classification |
| cardiffnlp/twitter-roberta-base-sentiment-latest | text-classification |
| deepset/bert-large-uncased-whole-word-masking-squad2 | question-answering |
| deepset/roberta-base-squad2 | question-answering |
| deepset/tinyroberta-squad2 | question-answering |
| dima806/fairface_age_image_detection | image-classification |
| distilbert/distilbert-base-cased-distilled-squad | question-answering |
| distilbert/distilbert-base-uncased | fill-mask |
| distilbert/distilbert-base-uncased-distilled-squad | question-answering |
| distilbert/distilbert-base-uncased-finetuned-sst-2-english | text-classification |
| facebook/dino-vitb16 | image-feature-extraction |
| facebook/dino-vits16 | image-feature-extraction |
| facebook/dinov2-base | image-feature-extraction |
| facebook/dinov2-large | image-feature-extraction |
| facebook/dinov2-small | image-feature-extraction |
| google-bert/bert-base-multilingual-cased | fill-mask |
| google-bert/bert-base-multilingual-uncased | fill-mask |
| google-bert/bert-base-uncased | fill-mask |
| google/vit-base-patch16-224 | image-classification |
| google/vit-base-patch16-224-in21k | image-feature-extraction |
| hustvl/yolos-small | object-detection |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | feature-extraction |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | zero-shot-image-classification |
| microsoft/rad-dino | image-feature-extraction |
| monologg/koelectra-small-v2-distilled-korquad-384 | question-answering |
| openai/clip-vit-base-patch16 | feature-extraction |
| openai/clip-vit-base-patch32 | feature-extraction |
| rizvandwiki/gender-classification | image-classification |
| sentence-transformers/all-MiniLM-L6-v2 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | sentence-similarity |
| sentence-transformers/all-mpnet-base-v2 | sentence-similarity |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | sentence-similarity |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-similarity |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | sentence-similarity |
| valentinafeve/yolos-fashionpedia | object-detection |
| w11wo/indonesian-roberta-base-posp-tagger | token-classification |

## w8a16 eval pass on ALL 3 NPU EPs

Subset of the 42 above where `*_w8a16_eval_result.json` exists in **every** NPU EP (QNN, OpenVINO, VitisAI).

Total: **19** (model, task) tuples.

| Model | Task |
|---|---|
| BAAI/bge-large-en-v1.5 | sentence-similarity |
| FacebookAI/roberta-base | fill-mask |
| cardiffnlp/twitter-roberta-base-sentiment-latest | text-classification |
| deepset/bert-large-uncased-whole-word-masking-squad2 | question-answering |
| deepset/roberta-base-squad2 | question-answering |
| deepset/tinyroberta-squad2 | question-answering |
| facebook/dino-vitb16 | image-feature-extraction |
| facebook/dinov2-base | image-feature-extraction |
| facebook/dinov2-large | image-feature-extraction |
| facebook/dinov2-small | image-feature-extraction |
| google-bert/bert-base-uncased | fill-mask |
| google/vit-base-patch16-224-in21k | image-feature-extraction |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | feature-extraction |
| microsoft/rad-dino | image-feature-extraction |
| openai/clip-vit-base-patch16 | feature-extraction |
| openai/clip-vit-base-patch32 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | sentence-similarity |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | sentence-similarity |
