# Supported Models

winml-cli supports a wide range of model architectures and tasks. This page
lists what's validated and how to discover model support.

---

## Discovery Commands

```bash
# Browse the curated catalog (57 validated models)
uv run winml catalog

# Filter by task
uv run winml catalog -t image-classification

# Check if a specific model is supported
uv run winml inspect -m microsoft/resnet-50

# List all known tasks
uv run winml inspect --list-tasks
```

---

## Supported Tasks

winml-cli recognizes **35 task types** across vision, NLP, audio, and multimodal domains. Of these, 16 have dedicated inference classes; the remainder are supported via the generic task fallback.

### Vision

| Task | Example Models |
|------|----------------|
| `image-classification` | ResNet, ConvNeXt, ViT, Swin |
| `image-segmentation` | Segformer, Mask2Former |
| `semantic-segmentation` | Segformer |
| `object-detection` | DETR, YOLOS, Table-Transformer |
| `depth-estimation` | Depth Anything, ZoeDepth |
| `image-feature-extraction` | DINOv2, ViT |
| `zero-shot-image-classification` | CLIP, SigLIP |

### NLP

| Task | Example Models |
|------|----------------|
| `text-classification` | BERT, RoBERTa, XLM-RoBERTa |
| `token-classification` | BERT, RoBERTa (NER) |
| `question-answering` | BERT, RoBERTa |
| `fill-mask` | BERT, RoBERTa |
| `feature-extraction` | BGE, BERT, all-MiniLM |
| `text-generation` | Qwen3 (composite) |
| `text2text-generation` | T5, BART, Marian |

### Audio

| Task | Example Models |
|------|----------------|
| `automatic-speech-recognition` | Whisper |
| `audio-classification` | Wav2Vec2 |

### Multimodal

| Task | Example Models |
|------|----------------|
| `zero-shot-image-classification` | CLIP (text + vision) |
| `image-to-text` | VisionEncoderDecoder |
| `visual-question-answering` | BLIP |

---

## Validated Model Catalog

The following models have been validated end-to-end with EP compatibility
testing. Use `winml catalog` to browse the full list interactively.

### Image Classification

| Model | Architecture |
|-------|-------------|
| `AdamCodd/vit-base-nsfw-detector` | ViT |
| `Falconsai/nsfw_image_detection` | ViT |
| `amunchet/rorshark-vit-base` | ViT |
| `apple/mobilevit-small` | MobileViT |
| `dima806/fairface_age_image_detection` | ViT |
| `google/vit-base-patch16-224` | ViT |
| `microsoft/resnet-18` | ResNet |
| `rizvandwiki/gender-classification` | ViT |

### Image Feature Extraction

| Model | Architecture |
|-------|-------------|
| `facebook/dino-vitb16` | ViT |
| `facebook/dino-vits16` | ViT |
| `facebook/dinov2-base` | DINOv2 |
| `facebook/dinov2-large` | DINOv2 |
| `facebook/dinov2-small` | DINOv2 |
| `google/vit-base-patch16-224-in21k` | ViT |
| `microsoft/rad-dino` | DINOv2 |

### Feature Extraction (Text)

| Model | Architecture |
|-------|-------------|
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | CLIP |
| `openai/clip-vit-base-patch16` | CLIP |
| `openai/clip-vit-base-patch32` | CLIP |
| `sentence-transformers/all-MiniLM-L6-v2` | BERT |
| `sentence-transformers/all-mpnet-base-v2` | MPNet |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | MPNet |

### Sentence Similarity

| Model | Architecture |
|-------|-------------|
| `BAAI/bge-large-en-v1.5` | BERT |
| `BAAI/bge-small-en-v1.5` | BERT |
| `sentence-transformers/all-MiniLM-L6-v2` | BERT |
| `sentence-transformers/all-mpnet-base-v2` | MPNet |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | MPNet |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | BERT |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | XLM-RoBERTa |

### Fill-Mask

| Model | Architecture |
|-------|-------------|
| `FacebookAI/roberta-base` | RoBERTa |
| `FacebookAI/xlm-roberta-base` | XLM-RoBERTa |
| `distilbert/distilbert-base-uncased` | DistilBERT |
| `google-bert/bert-base-multilingual-cased` | BERT |
| `google-bert/bert-base-multilingual-uncased` | BERT |
| `google-bert/bert-base-uncased` | BERT |
| `sentence-transformers/all-mpnet-base-v2` | MPNet |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | MPNet |

### Text Classification

| Model | Architecture |
|-------|-------------|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | RoBERTa |
| `cross-encoder/ms-marco-MiniLM-L4-v2` | BERT |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | BERT |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | DistilBERT |

### Token Classification

| Model | Architecture |
|-------|-------------|
| `Isotonic/distilbert_finetuned_ai4privacy_v2` | DistilBERT |
| `Jean-Baptiste/camembert-ner-with-dates` | CamemBERT |
| `kredor/punctuate-all` | XLM-RoBERTa |
| `w11wo/indonesian-roberta-base-posp-tagger` | RoBERTa |

### Question Answering

| Model | Architecture |
|-------|-------------|
| `ahotrod/electra_large_discriminator_squad2_512` | Electra |
| `deepset/bert-large-uncased-whole-word-masking-squad2` | BERT |
| `deepset/roberta-base-squad2` | RoBERTa |
| `deepset/tinyroberta-squad2` | RoBERTa |
| `distilbert/distilbert-base-cased-distilled-squad` | DistilBERT |
| `distilbert/distilbert-base-uncased-distilled-squad` | DistilBERT |
| `monologg/koelectra-small-v2-distilled-korquad-384` | Electra |

### Zero-Shot Classification

| Model | Architecture |
|-------|-------------|
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | DistilBERT |

### Zero-Shot Image Classification

| Model | Architecture |
|-------|-------------|
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | CLIP |

### Object Detection

| Model | Architecture |
|-------|-------------|
| `hustvl/yolos-small` | YOLOS |
| `valentinafeve/yolos-fashionpedia` | YOLOS |

### Depth Estimation

| Model | Architecture |
|-------|-------------|
| `Intel/dpt-hybrid-midas` | DPT |

---

## Execution Provider Compatibility

Each validated model is tested against available EPs:

| EP | Alias | Devices | Notes |
|----|-------|---------|-------|
| NvTensorRTRTXExecutionProvider | `nvtensorrtrtx`, `nv_tensorrt_rtx` | GPU | NVIDIA TensorRT-RTX; NVIDIA GPU with TensorRT runtime |
| CUDAExecutionProvider | `cuda` | GPU | NVIDIA CUDA; any CUDA-capable GPU |
| MIGraphXExecutionProvider | `migraphx` | GPU | AMD ROCm MIGraphX |
| QNNExecutionProvider | `qnn` | NPU, GPU | Qualcomm Snapdragon; bundled in ORT |
| OpenVINOExecutionProvider | `openvino` | NPU, GPU, CPU | Intel hardware |
| DmlExecutionProvider | `dml` | GPU | DirectML; any DirectX 12 GPU |
| CPUExecutionProvider | `cpu` | CPU | Always available |
| VitisAIExecutionProvider | `vitisai` | NPU | AMD/Xilinx |

---

## Adding Unsupported Models

If your model architecture isn't in the catalog, winml-cli may still support it
through auto-detection:

```bash
# Try inspecting first
uv run winml inspect -m your-org/your-model

# If "Status: Supported", proceed normally
uv run winml build -m your-org/your-model -d auto -o output/
```

For truly custom architectures, use `--trust-remote-code` to allow execution of
model code from the Hugging Face Hub.

---

## See also

- [winml catalog](../commands/catalog.md) — browse validated models interactively
- [winml inspect](../commands/inspect.md) — check model compatibility
- [EP and Device](../concepts/eps-and-devices.md) — execution provider details
