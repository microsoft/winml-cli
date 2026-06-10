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

| Model | Architecture | Size |
|-------|-------------|------|
| `AdamCodd/vit-base-nsfw-detector` | ViT | 83.4 MB |
| `Falconsai/nsfw_image_detection` | ViT | 82.8 MB |
| `amunchet/rorshark-vit-base` | ViT | 82.8 MB |
| `apple/mobilevit-small` | MobileViT | 6.1 MB |
| `dima806/fairface_age_image_detection` | ViT | 82.8 MB |
| `google/vit-base-patch16-224` | ViT | 83.6 MB |
| `microsoft/resnet-18` | ResNet | 11.2 MB |
| `rizvandwiki/gender-classification` | ViT | 82.8 MB |

### Image Feature Extraction

| Model | Architecture | Size |
|-------|-------------|------|
| `facebook/dino-vitb16` | ViT | 83.4 MB |
| `facebook/dino-vits16` | ViT | 21.6 MB |
| `facebook/dinov2-base` | DINOv2 | 82.8 MB |
| `facebook/dinov2-large` | DINOv2 | 291.4 MB |
| `facebook/dinov2-small` | DINOv2 | 21.4 MB |
| `google/vit-base-patch16-224-in21k` | ViT | 83.4 MB |
| `microsoft/rad-dino` | DINOv2 | 84.4 MB |

### Feature Extraction (Text)

| Model | Architecture | Size |
|-------|-------------|------|
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | CLIP | 85.4 MB |
| `openai/clip-vit-base-patch16` | CLIP | 85.5 MB |
| `openai/clip-vit-base-patch32` | CLIP | 85.5 MB |
| `sentence-transformers/all-MiniLM-L6-v2` | BERT | 33.2 MB |
| `sentence-transformers/all-mpnet-base-v2` | MPNet | 133.4 MB |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | MPNet | 133.4 MB |

### Sentence Similarity

| Model | Architecture | Size |
|-------|-------------|------|
| `BAAI/bge-large-en-v1.5` | BERT | 351.8 MB |
| `BAAI/bge-small-en-v1.5` | BERT | 43.9 MB |
| `sentence-transformers/all-MiniLM-L6-v2` | BERT | 33.4 MB |
| `sentence-transformers/all-mpnet-base-v2` | MPNet | 134.0 MB |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | MPNet | 134.0 MB |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | BERT | 204.7 MB |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | XLM-RoBERTa | 450.3 MB |

### Fill-Mask

| Model | Architecture | Size |
|-------|-------------|------|
| `FacebookAI/roberta-base` | RoBERTa | 194.7 MB |
| `FacebookAI/xlm-roberta-base` | XLM-RoBERTa | 634.4 MB |
| `distilbert/distilbert-base-uncased` | DistilBERT | 109.5 MB |
| `google-bert/bert-base-multilingual-cased` | BERT | 346.5 MB |
| `google-bert/bert-base-multilingual-uncased` | BERT | 316.4 MB |
| `google-bert/bert-base-uncased` | BERT | 150.5 MB |
| `sentence-transformers/all-mpnet-base-v2` | MPNet | 156.5 MB |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | MPNet | 156.5 MB |

### Text Classification

| Model | Architecture | Size |
|-------|-------------|------|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | RoBERTa | 157.7 MB |
| `cross-encoder/ms-marco-MiniLM-L4-v2` | BERT | 29.9 MB |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | BERT | 33.4 MB |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | DistilBERT | 87.0 MB |

### Token Classification

| Model | Architecture | Size |
|-------|-------------|------|
| `Isotonic/distilbert_finetuned_ai4privacy_v2` | DistilBERT | 86.6 MB |
| `Jean-Baptiste/camembert-ner-with-dates` | CamemBERT | 130.4 MB |
| `kredor/punctuate-all` | XLM-RoBERTa | 449.7 MB |
| `w11wo/indonesian-roberta-base-posp-tagger` | RoBERTa | 157.2 MB |

### Question Answering

| Model | Architecture | Size |
|-------|-------------|------|
| `ahotrod/electra_large_discriminator_squad2_512` | Electra | 350.9 MB |
| `deepset/bert-large-uncased-whole-word-masking-squad2` | BERT | 350.9 MB |
| `deepset/roberta-base-squad2` | RoBERTa | 157.2 MB |
| `deepset/tinyroberta-squad2` | RoBERTa | 116.2 MB |
| `distilbert/distilbert-base-cased-distilled-squad` | DistilBERT | 84.2 MB |
| `distilbert/distilbert-base-uncased-distilled-squad` | DistilBERT | 86.5 MB |
| `monologg/koelectra-small-v2-distilled-korquad-384` | Electra | 17.7 MB |

### Zero-Shot Classification

| Model | Architecture | Size |
|-------|-------------|------|
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | DistilBERT | 217.5 MB |

### Zero-Shot Image Classification

| Model | Architecture | Size |
|-------|-------------|------|
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | CLIP | 170.1 MB |

### Object Detection

| Model | Architecture | Size |
|-------|-------------|------|
| `hustvl/yolos-small` | YOLOS | 38.1 MB |
| `valentinafeve/yolos-fashionpedia` | YOLOS | 38.1 MB |

### Depth Estimation

| Model | Architecture | Size |
|-------|-------------|------|
| `Intel/dpt-hybrid-midas` | DPT | 117.9 MB |

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
