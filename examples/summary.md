# Example Configs Test Summary

## Builtin Models

Models that have a config in every one of the 9 (EP, device) buckets, all configs have perf pass, and at least one config has eval pass.

Total: **28** (model, task) tuples.

| Model | Task |
|---|---|
| apple/mobilevit-small | image-classification |
| cardiffnlp/twitter-roberta-base-sentiment-latest | text-classification |
| deepset/roberta-base-squad2 | question-answering |
| deepset/tinyroberta-squad2 | question-answering |
| distilbert/distilbert-base-cased-distilled-squad | question-answering |
| distilbert/distilbert-base-uncased | fill-mask |
| distilbert/distilbert-base-uncased-distilled-squad | question-answering |
| distilbert/distilbert-base-uncased-finetuned-sst-2-english | text-classification |
| facebook/dino-vitb16 | image-feature-extraction |
| facebook/dinov2-base | image-feature-extraction |
| facebook/dinov2-small | image-feature-extraction |
| google-bert/bert-base-multilingual-cased | fill-mask |
| hustvl/yolos-small | object-detection |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | zero-shot-image-classification |
| lxyuan/distilbert-base-multilingual-cased-sentiments-student | zero-shot-classification |
| microsoft/resnet-18 | image-classification |
| monologg/koelectra-small-v2-distilled-korquad-384 | question-answering |
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
| valentinafeve/yolos-fashionpedia | object-detection |

## Overview

| EP | Models | Configs | Perf Pass | Eval Pass | Report |
|----|--------|---------|-----------|-----------|--------|
| AMD (VitisAI, NPU) - fp16 | 24 | 29 | 28/29 (97%) | 19/29 (66%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a16 | 48 | 56 | 30/56 (54%) | 26/56 (46%) | [Report](vitisai/npu/REPORT.md) |
| AMD (VitisAI, NPU) - w8a8 | 48 | 56 | 33/56 (59%) | 29/56 (52%) | [Report](vitisai/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - fp16 | 78 | 92 | 90/92 (98%) | 64/92 (70%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a16 | 55 | 63 | 63/63 (100%) | 53/63 (84%) | [Report](qnn/npu/REPORT.md) |
| QNN (Qualcomm, NPU) - w8a8 | 55 | 63 | 63/63 (100%) | 53/63 (84%) | [Report](qnn/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - fp16 | 71 | 85 | 84/85 (99%) | 49/85 (58%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a16 | 48 | 56 | 55/56 (98%) | 51/56 (91%) | [Report](openvino/npu/REPORT.md) |
| OpenVINO (Intel, NPU) - w8a8 | 48 | 56 | 55/56 (98%) | 51/56 (91%) | [Report](openvino/npu/REPORT.md) |
| QNN (Qualcomm, GPU) | 78 | 92 | 57/92 (62%) | 48/92 (52%) | [Report](qnn/gpu/REPORT.md) |
| OpenVINO (Intel, CPU) | 78 | 92 | 91/92 (99%) | 49/92 (53%) | [Report](openvino/cpu/REPORT.md) |
| OpenVINO (Intel, GPU) | 78 | 92 | 87/92 (95%) | 56/92 (61%) | [Report](openvino/gpu/REPORT.md) |
| DML (GPU) | 78 | 92 | 86/92 (93%) | 43/92 (47%) | [Report](dml/gpu/REPORT.md) |
| MLAS (CPU) | 78 | 92 | 89/92 (97%) | 54/92 (59%) | [Report](mlas/cpu/REPORT.md) |
| NVIDIA TensorRT RTX (GPU) | 78 | 92 | 80/92 (87%) | 57/92 (62%) | [Report](nv_tensorrt_rtx/gpu/REPORT.md) |
