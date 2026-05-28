# Built-in Model Recipes

These are curated recipe configs for models that **pass both perf and eval
on every supported (EP, device) bucket**: DML/GPU, MLAS/CPU,
OpenVINO/{CPU,GPU,NPU}, QNN/{GPU,NPU}, VitisAI/NPU, NVIDIA TensorRT RTX/GPU.

Layout: `examples/<model_dir>/<task>_<precision>_config.json`

Each (model, task) ships with `fp16` plus `w8a8` and `w8a16` quantized variants.

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
| openai/clip-vit-base-patch16 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | feature-extraction |
| sentence-transformers/all-MiniLM-L6-v2 | sentence-similarity |
