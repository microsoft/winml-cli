# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Keypoint detection (human pose) evaluator using COCO OKS-based AP.

ViTPose is top-down: it predicts keypoints inside a given person box, and
transformers exposes no ``keypoint-detection`` pipeline. So this evaluator
drives the image processor and ONNX model directly — for each ground-truth
person box it runs ``processor.preprocess -> model -> post_process_pose_estimation``
— and scores the predictions against ground truth with ``KeypointAPMetric``.

Using ground-truth person boxes isolates pose accuracy from detection quality,
which is the standard COCO top-down evaluation protocol.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from transformers.image_processing_utils import BaseImageProcessor

    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

logger = logging.getLogger(__name__)


class WinMLKeypointDetectionEvaluator(WinMLEvaluator):
    """Evaluator for keypoint detection using COCO OKS-based AP."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        from ..utils.eval_utils import get_default

        mapping = config.dataset.columns_mapping
        task = "keypoint-detection"
        self._image_col = mapping.get("input_column", get_default(task, "input_column"))
        ann_col = mapping.get("annotation_column", get_default(task, "annotation_column"))
        keypoints_key = mapping.get("keypoints_key", get_default(task, "keypoints_key"))
        bbox_key = mapping.get("bbox_key", get_default(task, "bbox_key"))
        area_key = mapping.get("area_key", get_default(task, "area_key"))
        box_format = mapping.get("box_format", get_default(task, "box_format"))
        assert ann_col is not None, "annotation_column has no default for keypoint-detection"
        assert keypoints_key is not None, "keypoints_key has no default for keypoint-detection"
        assert bbox_key is not None, "bbox_key has no default for keypoint-detection"
        assert area_key is not None, "area_key has no default for keypoint-detection"
        assert box_format is not None, "box_format has no default for keypoint-detection"
        self._annotation_col: str = ann_col
        self._keypoints_key: str = keypoints_key
        self._bbox_key: str = bbox_key
        self._area_key: str = area_key
        self._box_format: str = box_format

        # Optional non-COCO keypoint layout: a model with a different keypoint
        # set (e.g. SynthPose's 52 anatomical markers) can be scored by this
        # same evaluator by supplying matching OKS sigmas and keypoint names
        # through the dataset config. Absent -> the metric's COCO 17 defaults.
        raw_sigmas = mapping.get("sigmas")
        raw_names = mapping.get("keypoint_names")
        self._sigmas: tuple[float, ...] | None = (
            tuple(float(s) for s in self._as_list(raw_sigmas)) if raw_sigmas else None
        )
        self._keypoint_names: tuple[str, ...] | None = (
            tuple(str(n) for n in self._as_list(raw_names)) if raw_names else None
        )

        super().__init__(config, model)

    def prepare_pipeline(self) -> BaseImageProcessor:
        """Load the image processor (no HF pipeline exists for this task).

        The processor size is forced to the exported ONNX input shape so the
        preprocessed crops match the static model input.
        """
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(self.config.model_id)

        io_config = getattr(self.model, "io_config", None) or {}
        input_shapes = io_config.get("input_shapes", [])
        if input_shapes and len(input_shapes[0]) == 4:
            _, _, h, w = input_shapes[0]
            processor.size = {"height": h, "width": w}  # type: ignore[attr-defined]

        return processor

    def compute(self) -> dict[str, Any]:
        """Run keypoint evaluation over all samples and return COCO AP/AR."""
        from tqdm import tqdm

        from .metrics import KeypointAPMetric

        processor = self.pipe
        predictions: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        skipped = 0

        for image_id, sample in enumerate(tqdm(self.data, desc="Evaluating keypoints")):
            image = sample.get(self._image_col)
            annotation = sample.get(self._annotation_col)
            if image is None or not annotation:
                skipped += 1
                continue

            boxes = [self._to_xywh(b) for b in annotation[self._bbox_key]]
            gt_keypoints = annotation[self._keypoints_key]
            areas = annotation[self._area_key]
            if not boxes:
                skipped += 1
                continue

            pose_results = self._predict_poses(processor, image, boxes)

            for person_idx, pose in enumerate(pose_results):
                predictions.append(
                    {
                        "image_id": image_id,
                        "keypoints": self._flatten_prediction(pose),
                        "score": self._person_score(pose),
                    }
                )
                references.append(
                    {
                        "image_id": image_id,
                        "keypoints": list(gt_keypoints[person_idx]),
                        "bbox": boxes[person_idx],
                        "area": float(areas[person_idx]),
                    }
                )

        if skipped:
            logger.warning("Skipped %d samples with missing image or annotations.", skipped)

        metric_kwargs: dict[str, Any] = {}
        if self._sigmas is not None:
            metric_kwargs["sigmas"] = self._sigmas
        if self._keypoint_names is not None:
            metric_kwargs["keypoint_names"] = self._keypoint_names
        return KeypointAPMetric().compute(
            predictions=predictions, references=references, **metric_kwargs
        )

    def _predict_poses(
        self,
        processor: BaseImageProcessor,
        image: Any,
        boxes: list[list[float]],
    ) -> list[dict[str, Any]]:
        """Run preprocess -> model -> post_process for one image's person boxes.

        ViTPose is exported with a static batch size of 1, so each person crop
        is run separately and the resulting heatmaps are stacked back into one
        ``(num_persons, ...)`` batch for post-processing.
        """
        import torch

        inputs = processor.preprocess(images=image, boxes=[boxes], return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        heatmaps = []
        for i in range(pixel_values.shape[0]):
            outputs = self.model(pixel_values=pixel_values[i : i + 1])
            heatmaps.append(self._extract_heatmaps(outputs))

        wrapped = SimpleNamespace(heatmaps=torch.cat(heatmaps, dim=0))
        # post_process returns one list per image; we pass a single image.
        return processor.post_process_pose_estimation(wrapped, boxes=[boxes])[0]

    @staticmethod
    def _extract_heatmaps(outputs: Any) -> Any:
        """Pull the heatmap tensor from the model output.

        Falls back to the first output when the name differs, so the evaluator
        does not depend on a specific ONNX output tensor name.
        """
        if not isinstance(outputs, dict):
            return outputs.heatmaps
        heatmaps = outputs.get("heatmaps")
        if heatmaps is None:
            heatmaps = next(iter(outputs.values()))
        return heatmaps

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """Coerce a comma-separated string or an existing sequence into a list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)

    def _to_xywh(self, box: Any) -> list[float]:
        """Normalize a person box to COCO ``[x, y, w, h]``."""
        x0, y0, a, b = (float(v) for v in box)
        if self._box_format == "xyxy":
            return [x0, y0, a - x0, b - y0]
        return [x0, y0, a, b]

    @staticmethod
    def _flatten_prediction(pose: dict[str, Any]) -> list[float]:
        """Interleave predicted ``(x, y)`` and per-keypoint score to ``[x, y, s, ...]``."""
        keypoints = pose["keypoints"].cpu().numpy()
        scores = pose["scores"].cpu().numpy()
        flat: list[float] = []
        for (x, y), score in zip(keypoints, scores, strict=False):
            flat.extend([float(x), float(y), float(score)])
        return flat

    @staticmethod
    def _person_score(pose: dict[str, Any]) -> float:
        """Overall person confidence: mean of per-keypoint scores."""
        scores = pose["scores"].cpu().numpy()
        return float(scores.mean()) if scores.size else 0.0
