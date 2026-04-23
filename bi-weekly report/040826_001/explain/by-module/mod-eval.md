# Module: eval
**Path**: `src/winml/modelkit/eval/`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
The `eval` module provides the evaluator framework for comparing WinML model outputs against PyTorch baselines. It includes a base evaluator, task-specific evaluators, and accuracy metrics.

## 2. Files Changed This Period
| File | PRs | Summary |
|------|-----|---------|
| `__init__.py` | #15, #42, #190 | Framework exported in #15; public API expanded in #42; feature extraction symbols in #190 |
| `base_evaluator.py` | #15, #190 | New in #15 (+203 lines); minor update in #190 (+8/-8) |
| `evaluate.py` | #15, #190 | New in #15 (+166 lines); feature extraction dispatch added in #190 (+18) |
| `image_segmentation_evaluator.py` | #15 | New file (+202 lines) |
| `object_detection_evaluator.py` | #15 | New file (+239 lines) |
| `text_classification_evaluator.py` | #15 | New file (+54 lines) |
| `token_classification_evaluator.py` | #15 | New file (+58 lines) |
| `feature_extraction_evaluator.py` | #190 | New file (+149 lines) |
| `config.py` | #15 | New file (+99 lines) |
| `metrics/__init__.py` | #15, #42, #190 | Metric symbols exported; SpearmanCorrelation added (#190) |
| `metrics/mean_average_precision.py` | #15 | New file (+155 lines) |
| `metrics/mean_iou.py` | #15 | New file (+103 lines) |
| `metrics/spearman_correlation.py` | #190 | New file (+57 lines) |

## 3. Net Change Summary
- The entire `eval` module framework was introduced in the #15 batch sync with a base evaluator, 4 task-specific evaluators, and 2 metrics.
- PR #42 expanded the public API: `WinMLEvaluator`, evaluator subclasses, `MAPMetric`, `MeanIoUMetric`, and `IGNORE_INDEX` are now exported from `eval/__init__.py`.
- PR #190 added the `FeatureExtractionEvaluator` using Spearman rank correlation and the `SpearmanCorrelation` metric class.

## 4. New APIs/Functions Added
| Symbol | Description |
|--------|-------------|
| `WinMLEvaluator` | Base evaluator class, exported from `eval/__init__.py` (#42) |
| `ImageSegmentationEvaluator` | Task evaluator for segmentation, exported (#42) |
| `ObjectDetectionEvaluator` | Task evaluator for object detection, exported (#42) |
| `TextClassificationEvaluator` | Task evaluator for text classification, exported (#42) |
| `TokenClassificationEvaluator` | Task evaluator for token classification, exported (#42) |
| `FeatureExtractionEvaluator` | New evaluator using Spearman correlation (#190) |
| `MAPMetric` | Mean Average Precision metric, exported (#42) |
| `MeanIoUMetric` | Mean IoU metric, exported (#42) |
| `SpearmanCorrelation` | New Spearman rank correlation metric (#190) |
| `IGNORE_INDEX` | Constant exported from `eval/__init__.py` (#42) |
