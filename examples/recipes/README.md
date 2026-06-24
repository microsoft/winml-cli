# Built-in Model Recipes

Curated recipe configuration samples for **portable, high-performance, and high-quality** AI models on Windows ML, working consistently across supported EPs.

**Supported EPs:**

DML/GPU, MLAS/CPU, OpenVINO (CPU/GPU/NPU), QNN (GPU/NPU), VitisAI/NPU, NVIDIA TensorRT RTX/GPU

Each *(model, task)* includes:

- `fp16`
- `w8a8`
- `w8a16` quantized variants

## Models

Total: **75** (model, task) tuples that pass fp16 eval on all 10 (EP, device) buckets.

| Model | Task |
|---|---|
| BAAI/bge-base-en-v1.5 | feature-extraction |
| BAAI/bge-base-en-v1.5 | sentence-similarity |
| BAAI/bge-large-en-v1.5 | sentence-similarity |
| BAAI/bge-m3 | feature-extraction |
| BAAI/bge-m3 | sentence-similarity |
| alibaba-damo/mgp-str-base | image-to-text (scene-text-recognition; requires L1-light registration in `src/winml/modelkit/models/hf/mgp_str.py`) |
| BAAI/bge-small-en-v1.5 | feature-extraction |
| BAAI/bge-small-en-v1.5 | sentence-similarity |
| Babelscape/wikineural-multilingual-ner | token-classification |
| FacebookAI/roberta-base | fill-mask |
| FacebookAI/roberta-large | fill-mask |
| FacebookAI/xlm-roberta-base | fill-mask |
| Intel/bert-base-uncased-mrpc | feature-extraction |
| Intel/bert-base-uncased-mrpc | text-classification |
| Isotonic/distilbert_finetuned_ai4privacy_v2 | token-classification |
| ProsusAI/finbert | text-classification |
| Salesforce/blip-image-captioning-base | image-to-text |
| StanfordAIMI/dinov2-base-xray-224 | image-feature-extraction |
| ahotrod/electra_large_discriminator_squad2_512 | question-answering |
| apple/mobilevit-small | image-classification |
| cardiffnlp/twitter-roberta-base-sentiment-latest | text-classification |
| dbmdz/bert-large-cased-finetuned-conll03-english | token-classification |
| deepset/bert-large-uncased-whole-word-masking-squad2 | question-answering |
| deepset/roberta-base-squad2 | question-answering |
| deepset/tinyroberta-squad2 | question-answering |
| dima806/fairface_age_image_detection | image-classification |
| distilbert/distilbert-base-cased-distilled-squad | question-answering |
| distilbert/distilbert-base-uncased | fill-mask |
| distilbert/distilbert-base-uncased-distilled-squad | question-answering |
| distilbert/distilbert-base-uncased-finetuned-sst-2-english | text-classification |
| dslim/bert-base-NER | token-classification |
| facebook/convnext-tiny-224 | image-classification |
| facebook/dino-vitb16 | image-feature-extraction |
| facebook/dino-vits16 | image-feature-extraction |
| facebook/dinov2-base | image-feature-extraction |
| facebook/dinov2-large | image-feature-extraction |
| facebook/dinov2-small | image-feature-extraction |
| google-bert/bert-base-multilingual-cased | feature-extraction |
| google-bert/bert-base-multilingual-cased | fill-mask |
| google-bert/bert-base-multilingual-uncased | fill-mask |
| google-bert/bert-base-uncased | fill-mask |
| google-bert/bert-large-uncased-whole-word-masking-finetuned-squad | question-answering |
| google/vit-base-patch16-224 | image-classification |
| google/vit-base-patch16-224-in21k | image-feature-extraction |
| joeddav/xlm-roberta-large-xnli | zero-shot-classification |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | feature-extraction |
| mattmdjaga/segformer_b2_clothes | image-segmentation |
| microsoft/rad-dino | image-feature-extraction |
| microsoft/resnet-18 | image-classification |
| microsoft/resnet-50 | image-classification |
| microsoft/swin-large-patch4-window7-224 | image-classification |
| microsoft/swinv2-tiny-patch4-window16-256 | image-classification |
| microsoft/trocr-base-handwritten | image-to-text |
| microsoft/trocr-base-printed | image-to-text |
| microsoft/trocr-large-handwritten | image-to-text |
| microsoft/trocr-large-printed | image-to-text |
| monologg/koelectra-small-v2-distilled-korquad-384 | question-answering |
| nvidia/segformer-b1-finetuned-ade-512-512 | image-segmentation |
| nvidia/segformer-b2-finetuned-ade-512-512 | image-segmentation |
| nvidia/segformer-b5-finetuned-ade-640-640 | image-segmentation |
| openai/clip-vit-base-patch16 | feature-extraction |
| openai/clip-vit-base-patch16 | zero-shot-image-classification |
| openai/clip-vit-base-patch32 | feature-extraction |
| openai/clip-vit-large-patch14 | zero-shot-image-classification |
| openai/clip-vit-large-patch14-336 | zero-shot-image-classification |
| rizvandwiki/gender-classification | image-classification |
| sentence-transformers/all-MiniLM-L6-v2 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | sentence-similarity |
| sentence-transformers/all-mpnet-base-v2 | feature-extraction |
| sentence-transformers/all-mpnet-base-v2 | sentence-similarity |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | feature-extraction |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | sentence-similarity |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | feature-extraction |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | sentence-similarity |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | sentence-similarity |
| w11wo/indonesian-roberta-base-posp-tagger | token-classification |
