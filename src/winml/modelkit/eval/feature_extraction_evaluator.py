# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Feature extraction evaluator using Spearman correlation on STS-B.

Evaluates sentence embedding models (e.g. sentence-transformers) by:
  1. Encoding each sentence in a pair via the feature-extraction pipeline.
  2. Mean-pooling token embeddings into a single sentence vector.
  3. Computing cosine similarity between the two vectors.
  4. Reporting Spearman rank correlation with ground-truth similarity scores.

Pipeline output contract (HF feature-extraction):
    pipe(text) -> [[[float, ...]]]   shape: [1, seq_len, hidden_dim]
    Mean-pool over seq_len -> [hidden_dim] sentence embedding.

Ground-truth dataset (default: mteb/stsbenchmark-sts):
    {"sentence1": str, "sentence2": str, "score": float}  score in [0, 5]
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


class WinMLFeatureExtractionEvaluator(WinMLEvaluator):
    """Evaluator for text feature extraction using Spearman correlation."""

    @classmethod
    def schema_info(cls) -> list:
        """Return expected dataset schema for sentence similarity evaluation."""
        from .config import SchemaColumn

        return [
            SchemaColumn(
                "sentence1",
                "Value(string)",
                "input_column_1",
                description="first sentence of the pair",
            ),
            SchemaColumn(
                "sentence2",
                "Value(string)",
                "input_column_2",
                description="second sentence of the pair",
            ),
            SchemaColumn(
                "score",
                "Value(float64)",
                "score_column",
                description="ground-truth similarity score (e.g. [0, 5] for STS-B)",
            ),
        ]

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        mapping = config.dataset.columns_mapping
        self._input_col_1 = mapping.get("input_column_1", "sentence1")
        self._input_col_2 = mapping.get("input_column_2", "sentence2")
        self._score_col = mapping.get("score_column", "score")
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline and set tokenizer padding for fixed-shape ONNX."""
        pipe = super().prepare_pipeline()

        if pipe.tokenizer is not None:
            io_config = getattr(self.model, "io_config", None) or {}
            shapes = io_config.get("input_shapes", [[]])
            if shapes and len(shapes[0]) > 1 and isinstance(shapes[0][1], int):
                pipe._preprocess_params.setdefault("padding", "max_length")
                pipe._preprocess_params.setdefault("max_length", shapes[0][1])
                pipe._preprocess_params.setdefault("truncation", True)

        return pipe

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """No-op: STS-B has no class labels to align."""
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run evaluation and return Spearman correlation."""
        from .metrics.spearman_correlation import SpearmanCorrelationMetric

        cosine_sims: list[float] = []
        gt_scores: list[float] = []

        for i, sample in enumerate(self.data):
            s1 = sample[self._input_col_1]
            s2 = sample[self._input_col_2]
            gt_score = sample[self._score_col]

            emb1 = self._embed(s1)
            emb2 = self._embed(s2)

            denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
            sim = float(np.dot(emb1, emb2) / max(denom, 1e-9))
            cosine_sims.append(sim)
            gt_scores.append(float(gt_score))

            if (i + 1) % 10 == 0:
                total = len(self.data) if hasattr(self.data, "__len__") else "?"
                logger.info("Processed %d / %s samples...", i + 1, total)

        return SpearmanCorrelationMetric().compute(cosine_sims, gt_scores)

    def _embed(self, text: str) -> np.ndarray:
        """Encode text and mean-pool token embeddings into a sentence vector.

        The HF feature-extraction pipeline returns a nested list:
            [[[float, ...]]]  shape: [batch=1, seq_len, hidden_dim]

        Uses attention-mask-weighted mean pooling to exclude padding tokens.
        This is critical for ONNX models that pad to a fixed sequence length
        (e.g. 512): without masking, 98%+ of the mean is over padding embeddings.
        Falls back to simple mean when no tokenizer is available.
        """
        from ..inference.tasks import _masked_mean_pool

        raw = self.pipe(text)  # [[[float, ...]]]
        token_embeddings = np.array(raw[0])  # [seq_len, hidden_dim]

        tokenizer = getattr(self.pipe, "tokenizer", None)
        if tokenizer is not None:
            params = self.pipe._preprocess_params
            enc = tokenizer(
                text,
                padding=params.get("padding", False),
                max_length=params.get("max_length", None),
                truncation=params.get("truncation", False),
                return_tensors="np",
            )
            return _masked_mean_pool(token_embeddings, enc["attention_mask"][0])

        return _masked_mean_pool(token_embeddings)
