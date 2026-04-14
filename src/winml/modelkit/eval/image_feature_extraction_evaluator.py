# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Image feature extraction evaluator using kNN classification accuracy.

Evaluates image embedding models (e.g. DINOv2, DINO, ViT-in21k) by:
  1. Extracting the CLS token embedding for each image via the pipeline.
  2. Running a leave-one-out k-Nearest Neighbor classifier.
  3. Reporting kNN top-1 and top-5 accuracy.

Pipeline output contract (HF image-feature-extraction):
    pipe(image) -> [[[float, ...]]]   shape: [1, num_tokens, hidden_dim]
    The first token (index 0) is the CLS token — the image-level embedding.

Ground-truth dataset (default: timm/mini-imagenet):
    {"image": PIL.Image, "label": ClassLabel}
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..datasets.config import DatasetConfig
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

logger = logging.getLogger(__name__)


class WinMLImageFeatureExtractionEvaluator(WinMLEvaluator):
    """Evaluator for image feature extraction using kNN classification accuracy."""

    @classmethod
    def schema_info(cls) -> list:
        """Return expected dataset schema for image feature extraction evaluation."""
        from .config import SchemaColumn

        return [
            SchemaColumn("image", "Image", "input_column", description="PIL Image"),
            SchemaColumn(
                "label", "ClassLabel", "label_column",
                description="integer class label",
            ),
        ]

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        mapping = config.dataset.columns_mapping
        self._label_col = mapping.get("label_column", "label")
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

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """No-op: kNN uses dataset labels directly, no model-side label mapping."""
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run kNN evaluation and return accuracy metrics."""
        from .metrics.knn_accuracy import KNNAccuracyMetric

        embeddings: list[np.ndarray] = []
        labels: list[int] = []

        for i, sample in enumerate(self.data):
            image = sample.get("image")
            label = sample.get(self._label_col)

            if image is None:
                logger.warning("Skipping sample %d: missing image.", i)
                continue

            raw = self.pipe(image)  # [[[float, ...]]]
            tokens = np.array(raw[0])
            embeddings.append(tokens[0] if tokens.ndim > 1 else tokens)
            labels.append(int(label))

            if (i + 1) % 10 == 0:
                total = len(self.data) if hasattr(self.data, "__len__") else "?"
                logger.info("Embedded %d / %s images...", i + 1, total)

        if len(embeddings) < 2:
            raise ValueError(
                f"Need at least 2 valid samples for kNN evaluation, got {len(embeddings)}."
            )

        embeddings_array = np.array(embeddings)
        labels_array = np.array(labels)

        metric = KNNAccuracyMetric(k=10)
        return metric.compute(embeddings_array, labels_array)
