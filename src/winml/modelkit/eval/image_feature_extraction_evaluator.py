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

from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..models.winml.base import WinMLPreTrainedModel
    from .config import DatasetConfig, WinMLEvaluationConfig


class WinMLImageFeatureExtractionEvaluator(WinMLEvaluator):
    """Evaluator for image feature extraction using kNN classification accuracy."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        from ..utils.eval_utils import get_default

        mapping = config.dataset.columns_mapping
        task = "image-feature-extraction"
        self._image_col = mapping.get("input_column", get_default(task, "input_column"))
        self._label_col = mapping.get("label_column", get_default(task, "label_column"))
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

        for sample in tqdm(self.data, desc="Embedding images", unit="img"):
            image = sample.get(self._image_col)
            label = sample.get(self._label_col)

            if image is None or label is None:
                continue

            raw = self.pipe(image)
            embeddings.append(self._extract_image_embedding(raw))
            labels.append(int(label))

        if len(embeddings) < 2:
            raise ValueError(
                f"Need at least 2 valid samples for kNN evaluation, got {len(embeddings)}."
            )

        embeddings_array = np.array(embeddings)
        labels_array = np.array(labels)

        metric = KNNAccuracyMetric(k=10)
        return metric.compute(embeddings_array, labels_array)

    @staticmethod
    def _extract_image_embedding(raw: Any) -> np.ndarray:
        """Reduce a pipeline output to a single 1D image-level embedding vector.

        Supports the two output shapes produced by HF ``image-feature-extraction``
        for transformer vision encoders (ViT / DINOv2 / DINO / BEiT / CLIP-ViT):
          - ``[1, num_tokens, hidden]`` (default, ``pool=False``): take CLS
            token at index 0 — the canonical image-level embedding.
          - ``[1, hidden]`` (``pool=True`` or a model with a projection head):
            use as-is.
        """
        tokens = np.asarray(raw[0])
        if tokens.ndim == 1:
            return tokens
        if tokens.ndim == 2:
            # CLS token (index 0) — standard image-level embedding for ViT/DINOv2.
            return tokens[0]
        raise ValueError(
            f"Unsupported image-feature-extraction output shape: {np.asarray(raw).shape}. "
            "Expected [1, hidden] (pooled) or [1, num_tokens, hidden] (token sequence)."
        )
