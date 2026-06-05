# Supported Models

winml-cli supports a wide range of model architectures and tasks. This page
lists what's validated and how to discover model support.

---

## Discovery Commands

```bash
# Browse the curated catalog (57 validated models)
uv run winml catalog

# Filter by task
uv run winml catalog -k image-classification

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

The following architectures have been validated end-to-end with EP compatibility
testing. Use `winml catalog` to browse the full list interactively.

### Image Classification

| Model | Architecture | EPs Tested |
|-------|-------------|------------|
| `microsoft/resnet-50` | ResNet | CPU, QNN (GPU/NPU), OpenVINO |
| `facebook/convnext-tiny-224` | ConvNeXt | CPU, QNN (GPU/NPU), OpenVINO |
| `google/vit-base-patch16-224` | ViT | CPU, QNN (GPU/NPU), OpenVINO |

### Text Classification & NLU

| Model | Architecture | EPs Tested |
|-------|-------------|------------|
| `bert-base-uncased` | BERT | CPU, QNN (GPU/NPU), OpenVINO |
| `FacebookAI/roberta-base` | RoBERTa | CPU, QNN, OpenVINO |
| `FacebookAI/xlm-roberta-base` | XLM-RoBERTa | CPU, QNN, OpenVINO |

### Feature Extraction & Embeddings

| Model | Architecture | EPs Tested |
|-------|-------------|------------|
| `BAAI/bge-base-en-v1.5` | BERT | CPU, QNN (GPU/NPU), OpenVINO |
| `BAAI/bge-small-en-v1.5` | BERT | CPU, QNN (GPU/NPU), OpenVINO |
| `sentence-transformers/all-MiniLM-L6-v2` | BERT | CPU, QNN, OpenVINO |

### Vision-Language

| Model | Architecture | EPs Tested |
|-------|-------------|------------|
| `openai/clip-vit-base-patch32` | CLIP | CPU, QNN, OpenVINO |
| `openai/clip-vit-large-patch14` | CLIP | CPU, QNN, OpenVINO |

### Segmentation

| Model | Architecture | EPs Tested |
|-------|-------------|------------|
| `nvidia/segformer-b0-finetuned-ade-512-512` | Segformer | CPU, QNN, OpenVINO |
| `nvidia/segformer-b1-finetuned-cityscapes-1024-1024` | Segformer | CPU, QNN, OpenVINO |

### Object Detection

| Model | Architecture | EPs Tested |
|-------|-------------|------------|
| `microsoft/table-transformer-detection` | Table-Transformer | CPU, OpenVINO |

---

## Execution Provider Compatibility

Each validated model is tested against available EPs:

| EP | Alias | Devices | Notes |
|----|-------|---------|-------|
| NvTensorRTRTXExecutionProvider | `nvtensorrtrtx`, `nv_tensorrt_rtx` | GPU | NVIDIA TensorRT-RTX; NVIDIA GPU with TensorRT runtime |
| CUDAExecutionProvider | `cuda` | GPU | NVIDIA CUDA; any CUDA-capable GPU |
| MIGraphXExecutionProvider | `migraphx` | GPU | AMD ROCm MIGraphX |
| QNNExecutionProvider | `qnn` | NPU, GPU | Qualcomm Snapdragon; bundled in ORT (`--compiler qairt` needs QNN SDK) |
| OpenVINOExecutionProvider | `openvino` | NPU, GPU, CPU | Intel hardware; install with `--extra openvino` |
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
