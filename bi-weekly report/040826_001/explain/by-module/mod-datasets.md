# Module: datasets
**Path**: `src/winml/modelkit/datasets/`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
The `datasets` module provides dataset loading and configuration for evaluation tasks (image classification, image segmentation, object detection, random datasets). It is consumed by the `eval` module and the e2e evaluation scripts.

## 2. Files Changed This Period
| File | PRs | Summary |
|------|-----|---------|
| `__init__.py` | #15, #40, #190 | Batch update in #15; DatasetConfig and DEFAULT_OBJECT_DETECTION_SIZE exported (#40); feature extraction dataset export (#190) |
| `config.py` | #15 | New file added in batch update (+58 lines) |
| `image_segmentation.py` | #15 | Batch update (+54/-x) |

## 3. Net Change Summary
- `DatasetConfig` and `DEFAULT_OBJECT_DETECTION_SIZE` were added to `datasets/__init__.py` in PR #40, eliminating internal submodule imports in test code.
- Feature extraction dataset support symbols exported in PR #190 to support `FeatureExtractionEvaluator`.
- `datasets/config.py` was introduced as part of the #15 batch sync, consolidating dataset configuration data classes.

## 4. New APIs/Functions Added
| Symbol | Description |
|--------|-------------|
| `DatasetConfig` | Exported from `datasets/__init__.py` (#40) |
| `DEFAULT_OBJECT_DETECTION_SIZE` | Exported from `datasets/__init__.py` (#40) |
