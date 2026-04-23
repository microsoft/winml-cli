# Module: models
**Path**: `src/winml/modelkit/models/`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
The `models` module contains HuggingFace model wrappers (`hf/`) and WinML model classes (`winml/`) for loading, configuring, and running models. It is the primary integration point between HuggingFace Transformers and the WinML runtime.

## 2. Files Changed This Period
| File | PRs | Summary |
|------|-----|---------|
| `hf/sam.py` | #13, #212 | `do_pool` import refactored (#13); SAM2 model config improvements (+60/-5) (#212) |
| `hf/depth_pro.py` | #15 | New file (+79 lines) |
| `hf/segformer.py` | #15 | New file (+109 lines) |
| `hf/zoedepth.py` | #15 | New file (+74 lines) |
| `hf/vision_encoder_decoder.py` | #15 | New file (+17 lines) |
| `hf/roberta.py` | #15 | Updated (+40/-x) |
| `hf/__init__.py` | #15, #43 | Batch update; loader symbols added (#43) |
| `winml/feature_extraction.py` | #190 | New WinML model class for feature extraction (+57 lines) |
| `winml/object_detection.py` | #15 | New file (+95 lines) |
| `winml/__init__.py` | #15, #43, #44, #190 | Batch update; WinML class exports added (#43, #44); feature extraction class (#190) |
| `__init__.py` | #43, #44 | 5 WinML model classes re-exported at package level |
| `auto.py` | #198, #205 | Removed `_get_model_config()`; stale path fixes; task mapping comments |

## 3. Net Change Summary
- Four new HuggingFace model wrappers added in the #15 batch: DepthPro, Segformer, ZoeDepth, VisionEncoderDecoder.
- `WinMLModelForFeatureExtraction` added in PR #190 as a new WinML task class.
- Five WinML model classes (`WinMLModelForGenericTask`, `WinMLModelForImageSegmentation`, `WinMLModelForObjectDetection`, `WinMLModelForSemanticSegmentation`, `WinMLModelForSequenceClassification`) are now exported from `models/__init__.py` (PR #44).
- The `do_pool` function in `sam.py` is now imported from the HuggingFace Transformers library rather than copied locally.
- SAM2 model fixed to handle `facebook/sam2.1-hiera-small` configuration correctly.

## 4. New APIs/Functions Added
| Symbol | Description |
|--------|-------------|
| `WinMLModelForFeatureExtraction` | New WinML model class for feature extraction tasks (#190) |
| `WinMLModelForGenericTask` | Re-exported from `models/__init__.py` (#44) |
| `WinMLModelForImageSegmentation` | Re-exported from `models/__init__.py` (#44) |
| `WinMLModelForObjectDetection` | Re-exported from `models/__init__.py` (#44) |
| `WinMLModelForSemanticSegmentation` | Re-exported from `models/__init__.py` (#44) |
| `WinMLModelForSequenceClassification` | Re-exported from `models/__init__.py` (#44) |
| `models/hf/depth_pro.py` | New HuggingFace DepthPro wrapper |
| `models/hf/segformer.py` | New HuggingFace Segformer wrapper |
| `models/hf/zoedepth.py` | New HuggingFace ZoeDepth wrapper |
| `models/winml/object_detection.py` | New WinML object detection model class |
