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

from ..utils.eval_utils import DatasetValidationError
from .base_evaluator import WinMLEvaluator
from .metrics import IGNORE_INDEX


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..models.winml.base import WinMLPreTrainedModel
    from .config import DatasetConfig, WinMLEvaluationConfig

logger = logging.getLogger(__name__)


class WinMLImageSegmentationEvaluator(WinMLEvaluator):
    """Evaluator for semantic segmentation using mIoU metrics."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        from ..utils.eval_utils import get_default

        mapping = config.dataset.columns_mapping
        task = "image-segmentation"
        self._image_col = mapping.get("input_column", get_default(task, "input_column"))
        self._annotation_col = mapping.get(
            "annotation_column", get_default(task, "annotation_column"),
        )
        self._binary_foreground = self._uses_binary_foreground_contract(model)
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline and match image processor size to ONNX input shape."""
        if getattr(self, "_binary_foreground", False):
            return None  # type: ignore[return-value]
        pipe = super().prepare_pipeline()

        io_config = getattr(self.model, "io_config", None) or {}
        input_shapes = io_config.get("input_shapes", [])
        if pipe.image_processor is not None and input_shapes and len(input_shapes[0]) == 4:
            _, _, h, w = input_shapes[0]
            # Runtime-settable processor attribute; not on the base class.
            pipe.image_processor.size = {"height": h, "width": w}  # type: ignore[attr-defined]

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
        if getattr(self, "_binary_foreground", False):
            from datasets import Image

            dataset = dataset.cast_column(self._image_col, Image(decode=False))
            dataset = dataset.cast_column(self._annotation_col, Image(decode=False))
        if ds_config.label_mapping:
            logger.info(
                "Label mapping provided (%d entries). "
                "GT pixels will be remapped during evaluation.",
                len(ds_config.label_mapping),
            )
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run segmentation evaluation and return mIoU metrics."""
        if getattr(self, "_binary_foreground", False):
            return self._compute_binary_foreground()

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
            image = sample.get(self._image_col)
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

    def _compute_binary_foreground(self) -> dict[str, Any]:
        """Evaluate a one-logit foreground mask without semantic-class decoding."""
        import io
        from pathlib import Path

        import torch
        from PIL import Image

        from .metrics import BinarySegmentationMetric

        metric = BinarySegmentationMetric()
        skipped = 0
        requested = len(self.data)
        input_shapes = self.model.io_config.get("input_shapes", [])
        if not input_shapes or len(input_shapes[0]) != 4:
            raise ValueError("Binary image segmentation requires a four-dimensional input shape.")
        _, channels, height, width = input_shapes[0]
        if channels != 3 or not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(
                "Binary image segmentation requires a static [batch, 3, height, width] input."
            )

        def open_image(value: Any, mode: str) -> Image.Image:
            if isinstance(value, Image.Image):
                return value.convert(mode)
            if isinstance(value, dict):
                if value.get("bytes") is not None:
                    return Image.open(io.BytesIO(value["bytes"])).convert(mode)
                if value.get("path") is not None:
                    return Image.open(value["path"]).convert(mode)
            if isinstance(value, bytes):
                return Image.open(io.BytesIO(value)).convert(mode)
            if isinstance(value, (str, Path)):
                return Image.open(value).convert(mode)
            raise TypeError(f"Unsupported image value: {type(value).__name__}")

        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

        for index, sample in enumerate(self.data):
            image_value = sample.get(self._image_col)
            annotation_value = sample.get(self._annotation_col)
            if image_value is None or annotation_value is None:
                logger.warning("Skipping sample %d: missing image or annotation.", index)
                skipped += 1
                continue

            image = open_image(image_value, "RGB")
            annotation = open_image(annotation_value, "L")
            resized = image.resize((width, height), Image.Resampling.BILINEAR)
            image_array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            normalized = (image_array - mean) / std
            output = self.model(pixel_values=torch.from_numpy(normalized[None]))
            logits = getattr(output, "logits", None)
            if logits is None:
                raise ValueError("Binary image-segmentation output is missing foreground logits.")
            if isinstance(logits, torch.Tensor):
                logits = logits.detach().cpu().numpy()
            logits_array = np.asarray(logits, dtype=np.float32)
            if logits_array.ndim != 4 or logits_array.shape[:2] != (1, 1):
                raise ValueError(
                    "Binary foreground logits must have shape [1, 1, height, width], "
                    f"got {logits_array.shape}."
                )
            stable_logits = np.clip(logits_array[0, 0], -88.0, 88.0)
            probabilities = 1.0 / (1.0 + np.exp(-stable_logits))
            prediction = Image.fromarray((probabilities >= 0.5).astype(np.uint8) * 255)
            prediction = prediction.resize(annotation.size, Image.Resampling.NEAREST)
            metric.update(np.asarray(prediction) > 0, np.asarray(annotation) > 0)

        result = metric.compute()
        result["num_skipped"] += skipped
        result["requested_samples"] = requested
        result["processed_samples"] = result["num_samples"]
        if result["num_samples"] == 0:
            raise DatasetValidationError(
                "Binary image-segmentation evaluation processed no non-empty masks."
            )
        return result

    @staticmethod
    def _uses_binary_foreground_contract(model: WinMLPreTrainedModel) -> bool:
        """Return whether metadata and ONNX I/O declare binary foreground masks."""
        config = getattr(model, "config", None)
        architectures = set(getattr(config, "architectures", None) or [])
        auto_map = getattr(config, "auto_map", None)
        if not architectures or not isinstance(auto_map, dict):
            return False
        remote_reference = auto_map.get("AutoModelForImageSegmentation")
        if not isinstance(remote_reference, str):
            return False
        if remote_reference.rsplit(".", 1)[-1] not in architectures:
            return False

        io_config = getattr(model, "io_config", None) or {}
        input_names = io_config.get("input_names", [])
        output_names = io_config.get("output_names", [])
        output_shapes = io_config.get("output_shapes", [])
        return (
            input_names == ["x"]
            and output_names == ["logits"]
            and len(output_shapes) == 1
            and len(output_shapes[0]) == 4
            and output_shapes[0][1] == 1
        )

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
        if self._image_col not in dataset.column_names:
            raise DatasetValidationError(
                f"Dataset missing image column '{self._image_col}'. "
                f"Available: {list(dataset.column_names)}.",
            )
        if self._annotation_col not in dataset.column_names:
            raise DatasetValidationError(
                f"Dataset missing annotation column '{self._annotation_col}'. "
                f"Available: {list(dataset.column_names)}. "
                f"Set annotation_column in columns_mapping.",
            )
