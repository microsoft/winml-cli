# Built-in Model Recipes

Curated recipe configuration samples for **portable, high-performance, and high-quality** AI models on Windows ML, working consistently across supported EPs.

**Supported EPs:**

DML/GPU, MLAS/CPU, OpenVINO (CPU/GPU/NPU), QNN (GPU/NPU), VitisAI/NPU, NVIDIA TensorRT RTX/GPU

Each *(model, task)* includes:

- `fp16`
- `w8a8`
- `w8a16` quantized variants

## Models

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
| nlpconnect/vit-gpt2-image-captioning | image-to-text |
| openai/clip-vit-base-patch16 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | sentence-similarity |
