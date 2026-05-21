# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Object detection evaluator using COCO-standard metrics.

Computes mAP, mAP@50, mAP@75 via MAPMetric (wrapping torchmetrics).
HF evaluate library has no object-detection evaluator, so this class
uses MAPMetric instead.

Label mapping: dataset label IDs are converted to model label IDs
via model.config.label2id[ClassLabel.names[dataset_id]].
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..utils.eval_utils import DatasetValidationError
from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..datasets.config import DatasetConfig
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

logger = logging.getLogger(__name__)


class WinMLObjectDetectionEvaluator(WinMLEvaluator):
    """Evaluator for object detection using COCO-standard mAP metrics."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        # Read column config BEFORE super().__init__() since prepare_data() needs them
        from ..utils.eval_utils import get_default

        mapping = config.dataset.columns_mapping
        task = "object-detection"
        self._image_col = mapping.get("input_column", get_default(task, "input_column"))
        self._annotation_col = mapping.get(
            "annotation_column", get_default(task, "annotation_column"),
        )
        self._bbox_key = mapping.get("bbox_key", get_default(task, "bbox_key"))
        self._category_key = mapping.get("category_key", get_default(task, "category_key"))
        self._box_format = mapping.get("box_format", get_default(task, "box_format"))
        self._box_coords = mapping.get("box_coords", get_default(task, "box_coords"))

        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline and match image processor size to ONNX input shape."""
        pipe = super().prepare_pipeline()

        io_config = getattr(self.model, "io_config", None) or {}
        input_shapes = io_config.get("input_shapes", [[]])
        if input_shapes and len(input_shapes[0]) == 4:
            _, _, h, w = input_shapes[0]
            pipe.image_processor.size = {"height": h, "width": w}
            if hasattr(pipe.image_processor, "do_pad"):
                pipe.image_processor.do_pad = False

        return pipe

    def align_labels(
        self,
        dataset: Dataset,
        ds_config: DatasetConfig,
    ) -> Dataset:
        """Remap ground truth category IDs from dataset space to model space."""
        from datasets import ClassLabel, Sequence

        self._validate_schema(dataset)

        label2id = ds_config.label_mapping or getattr(self.model.config, "label2id", None)
        if not label2id:
            logger.warning("No label2id found; alignment skipped. mAP may be incorrect.")
            return dataset

        ann_feature = dataset.features[self._annotation_col]
        cat_feature = ann_feature[self._category_key]
        if isinstance(cat_feature, Sequence):
            cat_feature = cat_feature.feature
        if not isinstance(cat_feature, ClassLabel):
            logger.warning("Category is not ClassLabel; alignment skipped. mAP may be incorrect.")
            return dataset

        ds_class_names = cat_feature.names

        # Build dataset_id → model_id mapping
        id_map = {}
        for ds_id, name in enumerate(ds_class_names):
            if name not in label2id:
                raise DatasetValidationError(
                    f"Dataset label '{name}' not in model's label2id. "
                    f"Model labels: {list(label2id.keys())}",
                )
            id_map[ds_id] = int(label2id[name])

        # Skip if already aligned
        if all(ds_id == model_id for ds_id, model_id in id_map.items()):
            logger.info("Labels already aligned for %s.", ds_config.path)
            return dataset

        ann_col = self._annotation_col
        cat_key = self._category_key

        # Update features for the aligned label space
        from datasets import Value

        new_features = dataset.features.copy()
        ann_feat = new_features[ann_col].copy()
        ann_feat[cat_key] = Sequence(Value("int64"))
        new_features[ann_col] = ann_feat

        def remap(sample):
            ann = sample[ann_col]
            ann[cat_key] = [id_map[lbl] for lbl in ann[cat_key]]
            return sample

        dataset = dataset.map(remap, features=new_features)
        logger.info(
            "Labels aligned for %s (%d classes remapped).",
            ds_config.path,
            len(id_map),
        )
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run object detection evaluation and return COCO metrics."""
        from .metrics import MAPMetric

        label2id = getattr(self.model.config, "label2id", {})

        predictions = []
        references = []

        for i, sample in enumerate(self.data):
            # --- Ground truth ---
            annotations = sample[self._annotation_col]

            ref: dict[str, Any] = {
                "boxes": annotations[self._bbox_key],
                "labels": [int(lbl) for lbl in annotations[self._category_key]],
            }
            if self._box_coords == "normalized":
                image = sample.get(self._image_col)
                if image is not None:
                    ref["image_size"] = image.size  # (width, height)
            references.append(ref)

            # --- Predictions ---
            image = sample.get(self._image_col)
            if image is None:
                predictions.append({"boxes": [], "scores": [], "labels": []})
                continue

            detections = self.pipe(image, threshold=0.0)
            if not detections:
                predictions.append({"boxes": [], "scores": [], "labels": []})
                continue

            predictions.append(
                {
                    "boxes": [
                        [d["box"]["xmin"], d["box"]["ymin"], d["box"]["xmax"], d["box"]["ymax"]]
                        for d in detections
                    ],
                    "scores": [d["score"] for d in detections],
                    "labels": [label2id.get(d["label"], -1) for d in detections],
                }
            )

            if (i + 1) % 10 == 0:
                logger.info("Processed %d / %d images...", i + 1, len(self.data))

        metric = MAPMetric()
        return metric.compute(
            predictions=predictions,
            references=references,
            box_format=self._box_format,
            box_coords=self._box_coords,
        )

    def _validate_schema(self, dataset: Dataset) -> None:
        """Check dataset has required annotation structure."""
        ann = dataset.features.get(self._annotation_col)
        if ann is None:
            raise DatasetValidationError(
                f"No column '{self._annotation_col}'. "
                f"Available: {list(dataset.features.keys())}.",
            )
        sub = ann
        if hasattr(ann, "feature") and isinstance(ann.feature, dict):
            sub = ann.feature
        for key in (self._bbox_key, self._category_key):
            if key not in sub:
                avail = list(sub.keys()) if isinstance(sub, dict) else []
                raise DatasetValidationError(
                    f"'{self._annotation_col}' has no key '{key}'. Available: {avail}.",
                )
