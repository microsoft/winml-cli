# Builtin Model Coverage

Five views over the eval-supported model set.

---

## 1. Target Builtin Models

Canonical 57 (model, task) pairs used by the config+eval workflow.

Total: **57** (model, task) tuples.

| Model | Task |
|---|---|
| AdamCodd/vit-base-nsfw-detector | image-classification |
| BAAI/bge-large-en-v1.5 | sentence-similarity |
| BAAI/bge-small-en-v1.5 | sentence-similarity |
| FacebookAI/roberta-base | fill-mask |
| FacebookAI/xlm-roberta-base | fill-mask |
| Falconsai/nsfw_image_detection | image-classification |
| Intel/dpt-hybrid-midas | depth-estimation |
| Isotonic/distilbert_finetuned_ai4privacy_v2 | token-classification |
| Jean-Baptiste/camembert-ner-with-dates | token-classification |
| ahotrod/electra_large_discriminator_squad2_512 | question-answering |
| amunchet/rorshark-vit-base | image-classification |
| apple/mobilevit-small | image-classification |
| cardiffnlp/twitter-roberta-base-sentiment-latest | text-classification |
| cross-encoder/ms-marco-MiniLM-L4-v2 | text-classification |
| cross-encoder/ms-marco-MiniLM-L6-v2 | text-classification |
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
| google-bert/bert-base-multilingual-cased | masked-lm |
| google-bert/bert-base-multilingual-uncased | fill-mask |
| google-bert/bert-base-uncased | fill-mask |
| google/vit-base-patch16-224 | image-classification |
| google/vit-base-patch16-224-in21k | image-feature-extraction |
| hustvl/yolos-small | object-detection |
| kredor/punctuate-all | token-classification |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | feature-extraction |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | zero-shot-image-classification |
| lxyuan/distilbert-base-multilingual-cased-sentiments-student | zero-shot-classification |
| microsoft/rad-dino | image-feature-extraction |
| microsoft/resnet-18 | image-classification |
| monologg/koelectra-small-v2-distilled-korquad-384 | question-answering |
| openai/clip-vit-base-patch16 | feature-extraction |
| openai/clip-vit-base-patch32 | feature-extraction |
| rizvandwiki/gender-classification | image-classification |
| sentence-transformers/all-MiniLM-L6-v2 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | sentence-similarity |
| sentence-transformers/all-mpnet-base-v2 | feature-extraction |
| sentence-transformers/all-mpnet-base-v2 | fill-mask |
| sentence-transformers/all-mpnet-base-v2 | sentence-similarity |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | feature-extraction |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | fill-mask |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | sentence-similarity |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-similarity |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | sentence-similarity |
| valentinafeve/yolos-fashionpedia | object-detection |
| w11wo/indonesian-roberta-base-posp-tagger | token-classification |

---

## 2. fp16 eval pass on ALL 9 EPs

Subset of the target list where fp16 eval pass on every one of the 9 (EP, device) buckets (CPU/GPU rows use plain `<task>_eval_result.json` or `<task>_fp16_eval_result.json`; NPU rows use `<task>_fp16_eval_result.json`).

Total: **12** (model, task) tuples.

| Model | Task |
|---|---|
| BAAI/bge-large-en-v1.5 | sentence-similarity |
| cardiffnlp/twitter-roberta-base-sentiment-latest | text-classification |
| deepset/roberta-base-squad2 | question-answering |
| deepset/tinyroberta-squad2 | question-answering |
| facebook/dinov2-base | image-feature-extraction |
| facebook/dinov2-small | image-feature-extraction |
| google/vit-base-patch16-224-in21k | image-feature-extraction |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | feature-extraction |
| microsoft/rad-dino | image-feature-extraction |
| openai/clip-vit-base-patch16 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | sentence-similarity |

---

## 3. fp16 eval pass on AT LEAST ONE EP

Subset of the target list where fp16 eval pass on at least one of the 9 EPs.

Total: **49** (model, task) tuples.

| Model | Task | EPs Passed |
|---|---|---|
| BAAI/bge-large-en-v1.5 | sentence-similarity | 9/9 |
| BAAI/bge-small-en-v1.5 | sentence-similarity | 8/9 |
| FacebookAI/roberta-base | fill-mask | 8/9 |
| FacebookAI/xlm-roberta-base | fill-mask | 2/9 |
| Isotonic/distilbert_finetuned_ai4privacy_v2 | token-classification | 1/9 |
| ahotrod/electra_large_discriminator_squad2_512 | question-answering | 4/9 |
| apple/mobilevit-small | image-classification | 4/9 |
| cardiffnlp/twitter-roberta-base-sentiment-latest | text-classification | 9/9 |
| deepset/bert-large-uncased-whole-word-masking-squad2 | question-answering | 8/9 |
| deepset/roberta-base-squad2 | question-answering | 9/9 |
| deepset/tinyroberta-squad2 | question-answering | 9/9 |
| dima806/fairface_age_image_detection | image-classification | 1/9 |
| distilbert/distilbert-base-cased-distilled-squad | question-answering | 6/9 |
| distilbert/distilbert-base-uncased | fill-mask | 4/9 |
| distilbert/distilbert-base-uncased-distilled-squad | question-answering | 6/9 |
| distilbert/distilbert-base-uncased-finetuned-sst-2-english | text-classification | 4/9 |
| facebook/dino-vitb16 | image-feature-extraction | 7/9 |
| facebook/dino-vits16 | image-feature-extraction | 8/9 |
| facebook/dinov2-base | image-feature-extraction | 9/9 |
| facebook/dinov2-large | image-feature-extraction | 8/9 |
| facebook/dinov2-small | image-feature-extraction | 9/9 |
| google-bert/bert-base-multilingual-cased | fill-mask | 5/9 |
| google-bert/bert-base-multilingual-uncased | fill-mask | 4/9 |
| google-bert/bert-base-uncased | fill-mask | 8/9 |
| google/vit-base-patch16-224 | image-classification | 8/9 |
| google/vit-base-patch16-224-in21k | image-feature-extraction | 9/9 |
| hustvl/yolos-small | object-detection | 6/9 |
| kredor/punctuate-all | token-classification | 7/9 |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | feature-extraction | 9/9 |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | zero-shot-image-classification | 7/9 |
| lxyuan/distilbert-base-multilingual-cased-sentiments-student | zero-shot-classification | 7/9 |
| microsoft/rad-dino | image-feature-extraction | 9/9 |
| microsoft/resnet-18 | image-classification | 7/9 |
| monologg/koelectra-small-v2-distilled-korquad-384 | question-answering | 5/9 |
| openai/clip-vit-base-patch16 | feature-extraction | 9/9 |
| openai/clip-vit-base-patch32 | feature-extraction | 8/9 |
| rizvandwiki/gender-classification | image-classification | 3/9 |
| sentence-transformers/all-MiniLM-L6-v2 | feature-extraction | 9/9 |
| sentence-transformers/all-MiniLM-L6-v2 | sentence-similarity | 9/9 |
| sentence-transformers/all-mpnet-base-v2 | feature-extraction | 7/9 |
| sentence-transformers/all-mpnet-base-v2 | fill-mask | 7/9 |
| sentence-transformers/all-mpnet-base-v2 | sentence-similarity | 7/9 |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | feature-extraction | 7/9 |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | fill-mask | 6/9 |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | sentence-similarity | 7/9 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-similarity | 8/9 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | sentence-similarity | 8/9 |
| valentinafeve/yolos-fashionpedia | object-detection | 7/9 |
| w11wo/indonesian-roberta-base-posp-tagger | token-classification | 2/9 |

---

## 4. w8a8 eval pass on ALL 3 NPU EPs

Subset of the target list where `*_w8a8_eval_result.json` exists in **every** NPU EP (QNN, OpenVINO, VitisAI).

Total: **21** (model, task) tuples.

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
| google-bert/bert-base-multilingual-uncased | fill-mask |
| google-bert/bert-base-uncased | fill-mask |
| google/vit-base-patch16-224-in21k | image-feature-extraction |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | feature-extraction |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | zero-shot-image-classification |
| microsoft/rad-dino | image-feature-extraction |
| openai/clip-vit-base-patch16 | feature-extraction |
| openai/clip-vit-base-patch32 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | sentence-similarity |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | sentence-similarity |

---

## 5. w8a16 eval pass on ALL 3 NPU EPs

Subset of the target list where `*_w8a16_eval_result.json` exists in **every** NPU EP (QNN, OpenVINO, VitisAI).

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
