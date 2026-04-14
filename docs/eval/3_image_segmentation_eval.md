# Image Segmentation Evaluator Design

## Overview

The image segmentation evaluator measures model quality using **mean Intersection-over-Union (mIoU)** — the standard benchmark metric for semantic segmentation. It compares model predictions against pixel-level ground truth annotations.

## CLI Usage

```bash
# Evaluate a HuggingFace model
wmk eval -m nvidia/segformer-b0-finetuned-ade-512-512 \
    --task image-segmentation \
    --dataset danjacobellis/scene_parse_150 \
    --label-mapping scripts/e2e_eval/ade20k_gt_to_model_label.json \
    --column annotation_column=annotation \
    --samples 1000

# Evaluate an ONNX model
wmk eval -m model.onnx \
    --model-id nvidia/segformer-b0-finetuned-ade-512-512 \
    --task image-segmentation \
    --dataset danjacobellis/scene_parse_150 \
    --label-mapping scripts/e2e_eval/ade20k_gt_to_model_label.json \
    --column annotation_column=annotation \
    --samples 1000
```

### Key Options

| Option | Description |
|--------|-------------|
| `--dataset` | HF Hub dataset ID (e.g., `danjacobellis/scene_parse_150`) or local path |
| `--label-mapping` | JSON file mapping dataset GT pixel values → model class IDs |
| `--column annotation_column=<name>` | Name of the annotation column in the dataset |
| `--samples` | Number of images to evaluate |
| `--task image-segmentation` | Required if not auto-detected |

## Dataset Schema

The evaluator expects a HuggingFace `Dataset` with two columns:

| Column | Type | Description |
|--------|------|-------------|
| `image` | `Image` (PIL RGB) | Input image, any resolution |
| `annotation` | `Image` (PIL L or RGB) | Per-pixel class ID annotation, same resolution as `image` |

The annotation column name defaults to `annotation` but can be overridden via `--column annotation_column=<name>`.

> **Note:** The original `zhoubolei/scene_parse_150` uses a deprecated HF script loader that is no longer supported by the `datasets` library. Use `danjacobellis/scene_parse_150` instead, which provides the same data in standard Parquet format.

## Model Output Format

The HF `image-segmentation` pipeline returns a **list of per-class binary masks**:

```python
[
    {"label": "sky",      "score": None, "mask": <PIL.Image mode=L>},
    {"label": "building", "score": None, "mask": <PIL.Image mode=L>},
    ...
]
```

- **`label`**: Human-readable class name from `model.config.id2label`
- **`score`**: Always `None` for semantic segmentation (hard assignments, no confidence)
- **`mask`**: Binary PIL Image — nonzero pixels belong to this class, zero pixels don't. Same size as the input image regardless of internal processing resolution.

### Prediction Conversion (`prepare_prediction`)

The evaluator stacks binary masks into a single label map:

```
Pipeline layers:                         Combined label map:
"sky" (class 2):    [[255,255,0,0],      [[2, 2, 1, 1],
                     [  0,  0,0,0]]       [4, 4, 4, 4]]
"building" (cls 1): [[  0,  0,255,255],
                     [  0,  0,  0,  0]]
"tree" (class 4):   [[  0,  0,  0,  0],
                     [255,255,255,255]]
```

Each pixel gets the class ID of whichever mask claims it. Masks are mutually exclusive in semantic segmentation.

## Ground Truth Format

A single-channel image where each **pixel value = class ID**:

```
[[3, 3, 3, 3],    ← sky=3
 [2, 2, 5, 5],    ← building=2, tree=5
 [5, 5, 5, 5]]    ← tree=5
```

### Reference Conversion (`prepare_reference`)

1. If annotation is RGB (e.g., Cityscapes stores `R=G=B=label_id`), extract the first channel
2. If `label_mapping` is provided, remap pixel values. Unmapped pixels → `IGNORE_INDEX` (-1)
3. If no `label_mapping`, use raw pixel values as class IDs directly

## Label Alignment

### When It's Needed

Label alignment is required when dataset GT pixel values differ from model class IDs. This is configured via a JSON label mapping file.

| Scenario | GT Pixel Values | Model Class IDs | Mapping Needed |
|----------|:-:|:-:|:-:|
| **ADE20K** | 0-150 (1-indexed, 0=background) | 0-149 | Yes: `{1→0, 2→1, ..., 150→149}` |
| **Cityscapes** | 0-33 (sparse label IDs) | 0-18 (train IDs) | Yes: `{7→0, 8→1, ..., 33→18}` |
| **ATR (clothes)** | 0-17 | 0-17 | No (identity) |

### How It Works

The mapping file is a JSON dict: `{"gt_pixel_value": model_class_id}`:

```json
{"1": 0, "2": 1, "3": 2, ..., "150": 149}
```

- Pixels **in** the mapping → remapped to the model class ID
- Pixels **not in** the mapping → set to `IGNORE_INDEX` (-1) → excluded from metric
- This handles background/void/unlabeled pixels automatically

### Label Compatibility Verification

Each dataset-model pair was verified:

| Dataset | Model | Classes Match | Unmapped Pixels |
|---------|-------|:-:|-------|
| ADE20K | SegFormer-ADE (150) | Yes (150/150 classes, offset by 1) | GT pixel 0 = "other objects" (official: excluded from eval) |
| Cityscapes | SegFormer-CS (19) | Yes (19/19 classes) | 10 void classes (unlabeled, ego vehicle, etc.) |
| ATR | segformer_b2_clothes (18) | Exact match (18/18) | None |

## Metric: Mean IoU (mIoU)

### Definition

Intersection-over-Union (IoU) measures how well a predicted segmentation mask overlaps with the ground truth mask for a given class. It is the ratio of the overlap area (intersection) to the total area covered by either the prediction or the ground truth (union). A perfect prediction gives IoU = 1.0; no overlap gives IoU = 0.0.

```
IoU(class_c) = TP_c / (TP_c + FP_c + FN_c)

  where:
    TP_c = pixels correctly predicted as class c
    FP_c = pixels incorrectly predicted as class c (actually another class)
    FN_c = pixels that are class c but predicted as something else

mIoU = mean(IoU over all classes present)
```

mIoU averages IoU across all classes, giving equal weight to each class regardless of how many pixels it covers. This makes it robust to class imbalance — a rare class matters as much as a dominant one.

### Implementation

Uses `torchmetrics.classification.MulticlassJaccardIndex` with incremental `update()`/`compute()` pattern:

```python
metric = MeanIoUMetric(num_classes=150, ignore_index=IGNORE_INDEX)

for pred, ref in images:
    metric.update(pred, ref)   # accumulates confusion matrix

result = metric.compute()      # final aggregation
# → {"mean_iou": float, "overall_accuracy": float, "per_category_iou": list}
```

### Returned Metrics

| Key | Type | Description |
|-----|------|-------------|
| `mean_iou` | `float` | Mean IoU across all classes (primary metric) |
| `overall_accuracy` | `float` | Pixel-level accuracy across all classes |
| `per_category_iou` | `list[float]` | Per-class IoU values (for debugging) |

### `IGNORE_INDEX`

A constant (`-1`) defined in `modelkit.eval.metrics.mean_iou`. Pixels with this value in either prediction or reference are excluded from the confusion matrix. Used for:
- Background/void pixels after label remapping
- Any pixel that has no corresponding model class

## Image Size Handling

### The Problem

Dataset images have **varying resolutions** (e.g., 683×512, 400×437, 300×225). The ONNX model has a **fixed input shape** (e.g., 224×224 or 512×512). The GT annotation has the **same resolution as the input image**.

### The Solution

The HF pipeline handles all resize logic internally. No manual resizing is needed.

```
Dataset image (683×512)
    │
    ├── Pipeline preprocess: saves target_size=(512, 683)
    │   └── Image processor squashes to model size (e.g., 224×224)
    │       └── No padding — SegFormer uses resize, not pad
    │
    ├── Model inference: logits at 1/4 resolution (56×56)
    │
    └── Pipeline postprocess: upsamples masks back to target_size=(512, 683)
        └── Output masks match original image size
```

### Key Properties

| Property | Behavior |
|----------|----------|
| **Dataset image size ≠ model input size** | Always true. Pipeline resizes internally. |
| **Resize method** | Squash (no padding). Aspect ratio is distorted internally but corrected on upsample. |
| **Output mask size** | Always matches original image size — verified across 10+ image sizes |
| **GT annotation size** | Always matches original image size — same resolution as `image` column |
| **Prediction vs GT shape** | Always identical — guaranteed by pipeline `target_size` mechanism |

### Evaluator's `prepare_pipeline()`

The evaluator overrides the image processor size to match the ONNX model's fixed input shape:

```python
io_config = self.model.io_config or {}
input_shapes = io_config.get("input_shapes", [])
if input_shapes and len(input_shapes[0]) == 4:
    _, _, h, w = input_shapes[0]
    pipe.image_processor.size = {"height": h, "width": w}
```

This ensures the processor sends the correct dimensions to the ONNX model. The pipeline still remembers the original image size and upsamples masks back to it.

### Important: Do NOT resize images or annotations manually

- Pre-resizing input images would change the output mask size, causing shape mismatch with GT
- Resizing annotation images **corrupts class IDs** — bilinear/bicubic interpolation creates invalid pixel values

### Note: Resize applies to both PyTorch and ONNX

The `SegformerImageProcessor` has `do_resize=True` and no `do_pad` option. This applies equally to both PyTorch and ONNX inference — the same processor is used.

Segmentation processors work differently:

- **SegFormer**: Has no `do_pad` attribute at all. Resize-only.
- **Mask2Former**: Has `pad_size` (not `do_pad`). Padding is **batch-level** — it happens in `encode_inputs()` to make all images in a batch the same size. Since the HF pipeline processes one image at a time, batch padding is a no-op. And since we set `size` to the exact ONNX dimensions, every image is already the correct shape after resize.

Therefore, no `do_pad = False` guard is needed for segmentation. The evaluator correctly relies on the processor's own resize behavior and the ONNX input shape override alone.

## File Inventory

| File | Purpose |
|------|---------|
| `modelkit/eval/image_segmentation_evaluator.py` | Main evaluator class |
| `modelkit/eval/metrics/mean_iou.py` | `MeanIoUMetric` wrapper + `IGNORE_INDEX` constant |
| `modelkit/eval/evaluate.py` | Registry entry for `image-segmentation` task |
| `scripts/e2e_eval/ade20k_gt_to_model_label.json` | ADE20K pixel → model class mapping (150 entries) |
| `scripts/e2e_eval/cityscapes_label_to_train_id.json` | Cityscapes label_id → train_id mapping (19 entries) |
| `scripts/e2e_eval/model_with_acc.json` | Model configs with dataset + label mapping references |
| `tests/eval/test_image_segmentation_evaluator.py` | 21 unit tests |

## Baseline Results (1000 samples, danjacobellis/scene_parse_150)

### nvidia/segformer-b0-finetuned-ade-512-512

| Model | mIoU | Pixel Accuracy | Input Resolution |
|-------|:-:|:-:|:-:|
| PyTorch | 0.3556 | 0.7762 | 512×512 |
| ONNX | 0.2390 | 0.7002 | 224×224 |

- PyTorch mIoU (0.3556) is consistent with the paper's 37.4 (single-scale, subset eval)
- ONNX accuracy gap is entirely from export resolution (224 vs 512) — verified: ONNX@512 = PyTorch@512 exactly
- Known issue: `wmk export` uses `config.image_size` (224, from backbone pretrain) instead of the image processor's size (512)
