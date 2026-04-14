# Object Detection Evaluator — Design

**Version**: 2.0
**Date**: 2026-03-16
**Status**: Implemented

---

## 1. Overview

The object detection evaluator measures how well an ONNX object detection model localizes and classifies objects in images. Given an image, the model produces a set of bounding box predictions — each with a class label and confidence score — and the evaluator compares these against ground truth annotations from a dataset.

The primary metric is **Mean Average Precision (mAP)**, the COCO-standard metric that combines both classification accuracy and localization quality into a single number. The evaluator computes mAP across multiple Intersection over Union (IoU) thresholds to assess how tightly the predicted boxes align with ground truth.

This evaluator extends the existing `wmk eval` framework (see [3_design.md](3_design.md)) with object-detection-specific logic for box matching, format conversion, and metric computation via `torchmetrics.detection.MeanAveragePrecision`.

---

## 2. Average Precision Metric

### 2.1 Intersection over Union (IoU)

IoU measures how well a predicted bounding box overlaps with a ground truth box. It is the ratio of the overlapping area to the total combined area of both boxes.

- IoU = 1.0 — the predicted box perfectly overlaps with the ground truth
- IoU = 0.0 — no overlap at all
- A threshold (e.g., 0.5) determines whether a detection "matches" a ground truth

### 2.2 Matching Predictions to Ground Truth

A prediction matches a ground truth box when:
- The $\text{IoU}$ between them $\geq$ the threshold (e.g., 0.5)
- The predicted class matches the ground truth class
- The ground truth box hasn't already been matched

From matching, each prediction is classified as:
- **True Positive (TP)** — correctly matched to a ground truth object
- **False Positive (FP)** — no matching ground truth (wrong location or wrong class)
- **False Negative (FN)** — a ground truth object that no prediction matched (missed detection)

### 2.3 Mean Average Precision (mAP)

**Average Precision (AP)** summarizes both the accuracy and completeness of detections for a single class at a single IoU threshold. It covers:

- **Precision** — "Of all detections the model made, how many were correct?"
- **Recall** — "Of all ground truth objects, how many did the model find?"

AP combines both into a single number (0 to 1) that balances finding all objects with making accurate detections. A perfect model has AP = 1.0.

**Mean Average Precision (mAP)** averages AP across all classes and multiple IoU thresholds. The primary COCO mAP averages over 10 IoU thresholds (0.50 to 0.95 in steps of 0.05), rewarding models that produce tightly localized boxes, not just roughly correct ones. This is the COCO standard metric used across virtually all object detection research and benchmarks.

The detailed computation is delegated to `torchmetrics.detection.MeanAveragePrecision`.

---

## 3. Schemas

### 3.1 I/O Schema

#### Input — Column Mapping

Object detection datasets have diverse column structures. The evaluator uses `columns_mapping` to locate annotation fields within the dataset, with sensible defaults for COCO-format datasets.

| Key | Default | Description |
|---|---|---|
| `annotation_column` | `"objects"` | Column name containing nested annotations (boxes + labels) |
| `bbox_key` | `"bbox"` | Key within annotation for bounding boxes |
| `category_key` | `"category"` | Key within annotation for category/class labels |
| `box_format` | `"xywh"` | Box format: `"xywh"` (top-left + size) or `"xyxy"` (two corners) |
| `box_coords` | `"absolute"` | Coordinate system: `"absolute"` (pixels) or `"normalized"` (0–1 range) |

**CLI usage**:
```bash
wmk eval -m model.onnx --model-id facebook/detr-resnet-50 \
    --dataset detection-datasets/coco \
    --column annotation_column=objects \
    --column bbox_key=bbox \
    --column category_key=category \
    --column box_format=xywh
```

#### Output — Evaluation Result

The evaluator returns a JSON object containing model metadata, dataset configuration, and COCO-standard metrics:

```json
{
  "model_id": "facebook/detr-resnet-50",
  "model_path": "temp/detr-resnet-50.onnx",
  "task": "object-detection",
  "device": "cpu",
  "dataset": {
    "path": "detection-datasets/coco",
    "split": "val",
    "samples": 100,
    "shuffle": true,
    "seed": 42,
    "columns_mapping": {
      "annotation_column": "objects",
      "bbox_key": "bbox",
      "category_key": "category",
      "box_format": "xywh"
    }
  },
  "output_path": "temp/od_eval_output.json",
  "metrics": {
    "map": 0.456,
    "map_50": 0.623,
    "map_75": 0.489,
    "map_small": 0.234,
    "map_medium": 0.412,
    "map_large": 0.567,
    "mar_1": 0.312,
    "mar_10": 0.487,
    "mar_100": 0.521,
    "num_predictions": 1523,
    "num_ground_truths": 2891,
    "num_images": 100
  }
}
```

| Metric Key | IoU Threshold | Description |
|---|---|---|
| `map` | 0.50:0.05:0.95 | **Primary metric** — average over 10 IoU thresholds |
| `map_50` | 0.50 | Loose matching (classic VOC metric) |
| `map_75` | 0.75 | Strict matching — requires tight localization |
| `map_small` | 0.50:0.95, area < 32² px | Small objects only |
| `map_medium` | 0.50:0.95, 32² < area < 96² px | Medium objects only |
| `map_large` | 0.50:0.95, area > 96² px | Large objects only |
| `mar_1` | 0.50:0.95, max 1 det/image | Average recall with at most 1 detection |
| `mar_10` | 0.50:0.95, max 10 det/image | Average recall with at most 10 detections |
| `mar_100` | 0.50:0.95, max 100 det/image | Average recall with at most 100 detections |
| `num_predictions` | — | Total predictions across all images |
| `num_ground_truths` | — | Total ground truth objects |
| `num_images` | — | Number of images evaluated |

### 3.2 Dataset Ground Truth Schema

The dataset provides per-image annotations as a nested structure under the annotation column. Example using COCO format (`detection-datasets/coco`) — an image containing a cat and a dog:

```python
# One dataset row
{
    "image": <PIL.Image>,          # The input image (an animal photo)
    "objects": {                    # annotation_column
        "bbox": [                   # bbox_key — list of bounding boxes
            [100.0, 150.0, 200.0, 180.0],   # category 15 → "cat" (xywh)
            [400.0, 50.0, 120.0, 200.0],    # category 17 → "dog" (xywh)
        ],
        "category": [15, 17],       
    }
}
```

| Field | Type | Description |
|---|---|---|
| `{annotation_column}.{bbox_key}` | `list[list[float]]` | Bounding boxes, format determined by `box_format` |
| `{annotation_column}.{category_key}` | `list[int]` | Class labels as `ClassLabel` integers (mapped to names via `ClassLabel.names`) |

The evaluator converts ground truth boxes to `[xmin, ymin, xmax, ymax]` absolute pixels internally, regardless of the source format.

### 3.3 Model Output Schema

The HuggingFace `object-detection` pipeline returns a list of detections per image. For the same animal image above:

```python
# Model predictions for the same image
[
    {
        "score": 0.95,
        "label": "cat",       # matches GT category 15 → "cat"
        "box": {"xmin": 105, "ymin": 148, "xmax": 295, "ymax": 330},
    },
    {
        "score": 0.87,
        "label": "dog",       # matches GT category 17 → "dog"
        "box": {"xmin": 410, "ymin": 55, "xmax": 515, "ymax": 245},
    },
]
```

| Field | Type | Description |
|---|---|---|
| `score` | `float` | Confidence score (0–1) |
| `label` | `str` | Class name from `model.config.id2label` |
| `box` | `dict` | `{xmin, ymin, xmax, ymax}` — always xyxy, absolute pixels |


---

## 4. Design Details

### 4.1 Evaluation Flow

The object detection evaluator follows a four-step flow:

```
1. Load Dataset    →  Load HF dataset, validate schema
2. Run Predictions →  Run model inference via HF pipeline, collect detections
3. Prepare Targets →  Convert ground truth to metric input format, align labels
4. Compute Metrics →  Delegate to MAPMetric (torchmetrics) for COCO mAP
```

Steps 2 and 3 happen per-sample in a single loop. After iterating through all samples, the collected predictions and targets are passed to `MAPMetric.compute()`.

### 4.2 Load Dataset

After loading the dataset via `load_dataset()`, the evaluator validates the dataset schema to ensure it contains the required annotation fields.

**Validation checks**:
1. The `annotation_column` (default: `"objects"`) exists in the dataset
2. The `bbox_key` (default: `"bbox"`) exists within the annotation column
3. The `category_key` (default: `"category"`) exists within the annotation column

If any field is missing, a `ValueError` is raised with context about what was expected and what was found. This catches dataset configuration errors early before any inference runs.

### 4.3 Compute Metrics

The evaluator delegates metric computation to `MAPMetric`, which wraps `torchmetrics.detection.MeanAveragePrecision`.

**`MAPMetric.compute()` signature**:

```python
def compute(
    self,
    predictions: list[dict[str, list]],
    references: list[dict[str, list]],
    box_format: str = "xywh",
    box_coords: str = "absolute",
) -> dict[str, Any]
```

**Input — `predictions`** (one dict per image):

```python
{
    "boxes": [[105, 148, 295, 330], [410, 55, 515, 245]],  # xyxy absolute pixels
    "scores": [0.95, 0.87],                                 # confidence floats (required for PR curve)
    "labels": [17, 20],                                      # model label IDs
}
```

**Input — `references`** (one dict per image):

```python
{
    "boxes": [[100.0, 150.0, 200.0, 180.0], [400.0, 50.0, 120.0, 200.0]],  # xywh
    "labels": [17, 20],        # model label IDs (-1 = excluded from evaluation)
}
```

**Output**: dict with `map`, `map_50`, `map_75`, `num_predictions`, `num_ground_truths`, `num_images`, and additional torchmetrics scalars.

#### torchmetrics Integration

`MAPMetric` converts the plain Python list inputs to the tensor format required by `torchmetrics.detection.MeanAveragePrecision`:

```python
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

# torchmetrics expects per-image dicts with torch tensors:
# predictions: {"boxes": Tensor(N,4), "scores": Tensor(N,), "labels": Tensor(N,)}
# targets:     {"boxes": Tensor(M,4), "labels": Tensor(M,)}

metric.update(preds=pred_list, target=target_list)
results = metric.compute()
```

**torchmetrics input requirements**:

| Input | Field | Type | Description |
|---|---|---|---|
| Prediction | `boxes` | `Tensor(N, 4)` float32 | Boxes in xyxy format, absolute pixels |
| Prediction | `scores` | `Tensor(N,)` float32 | Confidence scores — required for building the PR curve |
| Prediction | `labels` | `Tensor(N,)` int64 | Model label integer IDs |
| Target | `boxes` | `Tensor(M, 4)` float32 | Boxes in xyxy format, absolute pixels |
| Target | `labels` | `Tensor(M,)` int64 | Model label integer IDs |

`MAPMetric` handles the conversion internally:
- **`_convert_target()`**: Converts reference boxes from xywh → xyxy (if needed), denormalizes normalized coordinates, and filters out labels marked as `-1` (unmapped classes)
- **`_convert_prediction()`**: Converts plain Python lists to torch tensors

### 4.4 Run Predictions

The `WinMLModelForObjectDetection` class provides ONNX inference for object detection models.

**`forward()` signature**:

```python
def forward(
    self,
    pixel_values: torch.Tensor | np.ndarray,
    pixel_mask: torch.Tensor | np.ndarray | None = None,
    **kwargs: Any,
) -> ObjectDetectionOutput
```

**`ObjectDetectionOutput`** fields:

| Field | Type | Description |
|---|---|---|
| `logits` | `torch.FloatTensor` | Shape `[B, num_queries, num_classes+1]` — class logits |
| `pred_boxes` | `torch.FloatTensor` | Shape `[B, num_queries, 4]` — predicted boxes |

This output is compatible with `image_processor.post_process_object_detection()`, which the HF pipeline calls internally to convert raw model outputs into the pipeline detection format (list of `{score, label, box}` dicts described in section 3.3).

**Pipeline configuration**: The evaluator configures the HF pipeline before running inference:

1. **Image processor size from ONNX input shape** — Extracts `[B, C, H, W]` from the model's ONNX input shapes and sets `image_processor.size = {"height": H, "width": W}` to ensure preprocessing matches the model's expected input dimensions.
2. **Padding disabled** — Sets `image_processor.do_pad = False` when the attribute exists. Padding shifts bounding box coordinates relative to the image, which would produce incorrect metric results.
3. **Detection threshold set to 0.0** — Calls `pipe(image, threshold=0.0)` to include all predictions. This is required by the COCO evaluation protocol — the PR curve must be constructed from all detections sorted by confidence (see DD-004).

For each detection, the `label` string is converted to a model integer ID via `model.config.label2id`.

### 4.5 Prepare Targets

For each dataset sample, the ground truth is converted to the `MAPMetric` reference schema:

1. **Extract annotations**: Read `boxes` and `labels` from the annotation column
2. **Map labels**: Convert dataset `ClassLabel` integer IDs to model label IDs by matching class names:
   - Dataset integer → `ClassLabel.names[id]` → class name string → `model.config.label2id[name]` → model integer
   - If a class name is not found in `model.config.label2id`, the label is set to `-1`
   - Labels marked `-1` are filtered out by `MAPMetric._convert_target()` (our code) before passing to torchmetrics — these annotations are excluded from evaluation entirely (they don't count as FN)
3. **Fallback — no ClassLabel**: If the dataset's category feature is not a `ClassLabel` type (i.e., plain integers with no class name metadata), raw integer IDs are used directly without mapping. This assumes the dataset and model share the same integer label space. If they don't, metrics will be incorrect.
4. **Build reference dict**: `{"boxes": raw_boxes, "labels": mapped_labels}`

Box format conversion (xywh → xyxy) and coordinate denormalization are handled later by `MAPMetric._convert_target()` during metric computation.

---

## 5. Design Decisions

### DD-001: Use COCO-Standard Evaluation Protocol

**Decision**: Adopt the COCO evaluation protocol (mAP averaged over IoU thresholds 0.50:0.05:0.95) as the standard for object detection metrics.

**Rationale**: COCO-style mAP is the de facto standard in object detection research. Nearly every object detection paper reports COCO mAP as the primary metric. This means:
- Our evaluation results are directly comparable to published benchmarks
- Users can verify their ONNX model quality against known reference numbers
- The dataset is expected to provide bounding boxes and category labels in a structured annotation column (e.g., `objects.bbox`, `objects.category`)

**Alternatives considered**: Pascal VOC (single-threshold mAP@50 only — too lenient, largely superseded).

### DD-002: Use `torchmetrics.detection.MeanAveragePrecision` as Metric Backend

**Decision**: Use `torchmetrics[detection]` to compute COCO-standard mAP metrics, wrapped by our `MAPMetric` class.

**Rationale**: We evaluated three options:

| Option | Pros | Cons |
|---|---|---|
| **torchmetrics** (chosen) | PyTorch-native tensor API, COCO-standard results, wraps pycocotools for correctness | Extra dependency (but torch already required) |
| **pycocotools** (direct) | Gold-standard reference implementation | File/JSON-based API, C extension (historically tricky on Windows) |
| **Custom implementation** | No deps, full control | Significant effort to match COCO evaluation exactly, subtle bugs likely |

`torchmetrics` wraps `pycocotools` under the hood, giving reference-implementation correctness with a clean tensor-based API. Since we already depend on `torch`, this adds minimal dependency surface.

### DD-003: Label ID Mapping Between Dataset and Model

**Decision**: Build a `dataset_id → model_id` mapping by matching class name strings. Labels present in only one side are excluded from evaluation.

**Rationale**:

(a) **Mapping is required** — No HuggingFace object detection dataset was found where integer label IDs perfectly match the model's label IDs. For example, COCO datasets on HF use 0-indexed `ClassLabel` integers (0 = "person", 1 = "bicycle", ...), while models like DETR use a different scheme where `id2label[0] = "N/A"` (a reserved "no object" slot) and `id2label[1] = "person"`. Direct integer comparison would produce incorrect matches.

(b) **Map by class name** — Class name string matching is the only universal convention available. Both HF datasets (via `ClassLabel.names`) and HF models (via `model.config.id2label`) expose human-readable class names. There is no other shared identifier across the HF ecosystem.

(c) **Unmapped labels produce a warning** — If a dataset class name is not found in `model.config.label2id`, the label is set to `-1` and a warning is logged. These annotations are filtered out by `MAPMetric._convert_target()` before reaching torchmetrics. This ensures unmapped classes don't produce false negatives. Similarly, predictions with labels not found in the dataset's class set don't produce false positives.

### DD-004: Use `threshold=0.0` During Evaluation

**Decision**: Override the HF pipeline's default `threshold=0.5` to `threshold=0.0` during evaluation.

**Rationale**: The COCO evaluation protocol requires **all** predictions to be included. The PR curve is constructed by sorting all predictions by confidence score — pre-filtering by score would truncate the curve and produce incorrect mAP numbers. With `threshold=0.0`, the pipeline returns all model queries (e.g., DETR returns 100 detections per image = `num_queries`), compared to only a handful with the default `threshold=0.5`. This is explicitly hardcoded in the evaluator and is not user-configurable.
