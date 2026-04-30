# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Monocular depth estimation evaluator.

HF ``evaluate`` has no depth-estimation evaluator, so we run the metric
loop manually against ``DepthMetric`` (AbsRel, RMSE, delta1).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

from .base_evaluator import WinMLEvaluator
from .metrics import DepthMetric


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..datasets.config import DatasetConfig
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

logger = logging.getLogger(__name__)


class WinMLDepthEstimationEvaluator(WinMLEvaluator):
    """Evaluator for monocular depth estimation."""

    @classmethod
    def schema_info(cls) -> list:
        """Return expected dataset schema for depth estimation."""
        from .config import SchemaColumn

        return [
            SchemaColumn("image", "Image", "input_column", description="PIL Image"),
            SchemaColumn(
                "depth_map",
                "Image",
                "depth_column",
                description=(
                    "single-channel depth image; any unit when align=median, "
                    "otherwise must match the model's predicted unit (e.g. metres)"
                ),
            ),
            SchemaColumn(
                "median",
                "option",
                "align",
                required=False,
                description=(
                    'per-image scaling: "median" (relative-depth models) '
                    'or "none" (metric models like ZoeDepth/DepthPro)'
                ),
            ),
            SchemaColumn(
                "1e-3",
                "float",
                "min_depth",
                required=False,
                description=(
                    "GT pixels at or below this are excluded "
                    "(sensor noise floor; NYU default)"
                ),
            ),
            SchemaColumn(
                "10.0",
                "float | none",
                "max_depth",
                required=False,
                description='GT pixels above this are excluded (NYU=10, KITTI=80, "none" disables)',
            ),
        ]

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        mapping = config.dataset.columns_mapping
        self._input_col = mapping.get("input_column", "image")
        self._depth_col = mapping.get("depth_column", "depth_map")
        self._align = mapping.get("align", "median")
        self._min_depth = float(mapping.get("min_depth", 1e-3))
        max_depth_raw = mapping.get("max_depth", 10.0)
        self._max_depth: float | None
        if isinstance(max_depth_raw, str) and max_depth_raw.lower() == "none":
            self._max_depth = None
        else:
            self._max_depth = float(max_depth_raw)
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline and match image processor size to ONNX input shape."""
        pipe = super().prepare_pipeline()

        io_config = getattr(self.model, "io_config", None) or {}
        input_shapes = io_config.get("input_shapes", [])
        if input_shapes and len(input_shapes[0]) == 4 and pipe.image_processor is not None:
            _, _, h, w = input_shapes[0]
            pipe.image_processor.size = {"height": h, "width": w}

        return pipe

    def align_labels(
        self,
        dataset: Dataset,
        ds_config: DatasetConfig,
    ) -> Dataset:
        """Validate input and depth columns; no class-label mapping for depth."""
        self._validate_schema(dataset)
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run depth evaluation over all samples."""
        metric = DepthMetric(
            align=self._align,
            min_depth=self._min_depth,
            max_depth=self._max_depth,
        )

        skipped = 0
        for sample in tqdm(self.data, desc="Evaluating depth"):
            image = sample.get(self._input_col)
            depth = sample.get(self._depth_col)
            if image is None or depth is None:
                skipped += 1
                continue

            result = self.pipe(image)
            pred = self._extract_predicted_depth(result)
            gt = np.asarray(depth, dtype=np.float32)
            metric.update(pred, gt)

        if skipped:
            logger.warning("Skipped %d samples with missing image or depth.", skipped)

        return metric.compute()

    @staticmethod
    def _extract_predicted_depth(result: Any) -> np.ndarray:
        """Pull the numeric depth tensor out of an HF pipeline result."""
        if not isinstance(result, dict):
            raise TypeError(
                f"Unexpected pipeline output type: {type(result).__name__}; expected dict.",
            )

        predicted = result.get("predicted_depth")
        if predicted is None:
            raise ValueError(
                f"Pipeline output missing 'predicted_depth'; got keys {list(result)}.",
            )
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.detach().cpu().numpy()
        return np.asarray(predicted, dtype=np.float32).squeeze()

    def _validate_schema(self, dataset: Dataset) -> None:
        """Check dataset has required image and depth columns."""
        col_names = list(dataset.column_names)
        if self._input_col not in col_names:
            raise ValueError(
                f"Dataset missing input column '{self._input_col}'. "
                f"Available: {col_names}. Set input_column in columns_mapping.",
            )
        if self._depth_col not in col_names:
            raise ValueError(
                f"Dataset missing depth column '{self._depth_col}'. "
                f"Available: {col_names}. Set depth_column in columns_mapping.",
            )
