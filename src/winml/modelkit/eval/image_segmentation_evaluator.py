# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Semantic segmentation evaluator using mIoU metric.

Computes mean IoU, pixel accuracy via MeanIoUMetric (wrapping torchmetrics).
HF evaluate library has no image-segmentation evaluator, so this class
handles pipeline output conversion and metric computation manually.

Pipeline output: list of {"label": str, "mask": PIL.Image(0/255)} per image.
Ground truth: single-channel annotation image, pixel values = class IDs.

Label alignment: When dataset GT pixel values differ from model class IDs,
provide a label_mapping (via --label-mapping or label_mapping_file in config)
to remap GT pixels. Unmapped pixels are set to -1 (ignored).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .base_evaluator import WinMLEvaluator
from .metrics import IGNORE_INDEX


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..datasets.config import DatasetConfig
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

logger = logging.getLogger(__name__)


class WinMLImageSegmentationEvaluator(WinMLEvaluator):
    """Evaluator for semantic segmentation using mIoU metrics."""

    @classmethod
    def schema_info(cls) -> list:
        """Return expected dataset schema for image segmentation."""
        from .config import SchemaColumn

        return [
            SchemaColumn("image", "Image", "input_column", description="PIL Image"),
            SchemaColumn(
                "annotation",
                "Image",
                "annotation_column",
                description="Single-channel annotation image (pixel value = class ID)",
            ),
        ]

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        ds = config.dataset
        self._annotation_col = ds.columns_mapping.get("annotation_column", "annotation")
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline and match image processor size to ONNX input shape."""
        pipe = super().prepare_pipeline()

        io_config = getattr(self.model, "io_config", None) or {}
        input_shapes = io_config.get("input_shapes", [])
        if input_shapes and len(input_shapes[0]) == 4:
            _, _, h, w = input_shapes[0]
            pipe.image_processor.size = {"height": h, "width": w}

        return pipe

    def align_labels(
        self,
        dataset: Dataset,
        ds_config: DatasetConfig,
    ) -> Dataset:
        """Validate schema and log label remapping status.

        For segmentation, label alignment is pixel-level: each pixel value in
        the annotation image is remapped from dataset IDs to model class IDs.
        This is done per-sample in compute() using ds_config.label_mapping.
        """
        self._validate_schema(dataset)
        if ds_config.label_mapping:
            logger.info(
                "Label mapping provided (%d entries). "
                "GT pixels will be remapped during evaluation.",
                len(ds_config.label_mapping),
            )
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run segmentation evaluation and return mIoU metrics."""
        from .metrics import MeanIoUMetric

        num_labels = getattr(self.model.config, "num_labels", None)
        if num_labels is None:
            raise ValueError("model.config.num_labels is required for segmentation evaluation.")
        label2id = getattr(self.model.config, "label2id", {})
        label_mapping = self.config.dataset.label_mapping

        metric = MeanIoUMetric(
            num_classes=num_labels,
            ignore_index=IGNORE_INDEX,
        )

        for i, sample in enumerate(self.data):
            image = sample.get("image")
            annotation = sample.get(self._annotation_col)

            if image is None or annotation is None:
                logger.warning("Skipping sample %d: missing image or annotation.", i)
                continue

            result = self.pipe(image)
            pred_label_map = self.prepare_prediction(result, label2id, image.size)
            gt_label_map = self.prepare_reference(annotation, label_mapping)

            metric.update(pred_label_map, gt_label_map)

            if (i + 1) % 10 == 0:
                logger.info("Processed %d / %d images...", i + 1, len(self.data))

        return metric.compute()

    @staticmethod
    def prepare_prediction(
        pipeline_result: list[dict[str, Any]],
        label2id: dict[str, int],
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Convert pipeline binary masks into a single label map.

        Args:
            pipeline_result: Pipeline output, list of {"label": str, "mask": PIL.Image}.
            label2id: Model's label name → class ID mapping.
            image_size: (width, height) from PIL Image.

        Returns:
            (H, W) int64 array with class IDs per pixel (IGNORE_INDEX for uncovered).
        """
        height, width = image_size[1], image_size[0]
        label_map = np.full((height, width), IGNORE_INDEX, dtype=np.int64)

        for item in pipeline_result:
            class_id = label2id.get(item["label"], -1)
            if class_id < 0:
                continue
            mask = np.array(item["mask"])
            label_map[mask > 0] = int(class_id)

        return label_map

    @staticmethod
    def prepare_reference(
        annotation: Any,
        label_mapping: dict[str, int] | None,
    ) -> np.ndarray:
        """Convert annotation image to a label map, applying remapping if provided.

        Args:
            annotation: PIL Image with pixel values as class IDs.
            label_mapping: Optional mapping from GT pixel values to model class IDs.
                Unmapped pixels are set to IGNORE_INDEX.

        Returns:
            (H, W) int64 array with model class IDs (or IGNORE_INDEX for unmapped).
        """
        gt_label_map = np.array(annotation)
        # Handle RGB annotations (e.g., Cityscapes: R=G=B=label_id)
        if gt_label_map.ndim == 3:
            gt_label_map = gt_label_map[:, :, 0]

        if label_mapping:
            remapped = np.full_like(gt_label_map, IGNORE_INDEX, dtype=np.int64)
            for src, dst in label_mapping.items():
                remapped[gt_label_map == int(src)] = int(dst)
            gt_label_map = remapped

        return gt_label_map

    def _validate_schema(self, dataset: Dataset) -> None:
        """Check dataset has required columns."""
        if "image" not in dataset.column_names:
            raise ValueError(
                f"Dataset missing 'image' column. Available: {list(dataset.column_names)}."
            )
        if self._annotation_col not in dataset.column_names:
            raise ValueError(
                f"Dataset missing annotation column '{self._annotation_col}'. "
                f"Available: {list(dataset.column_names)}. "
                f"Set annotation_column in columns_mapping."
            )
