# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Image feature extraction evaluator using kNN classification accuracy.

Evaluates image embedding models (e.g. DINOv2, DINO, ViT-in21k) by:
  1. Extracting the CLS token embedding for each image via the pipeline.
  2. Running a leave-one-out k-Nearest Neighbor classifier.
  3. Reporting kNN top-1 and top-5 accuracy alongside standard retrieval
     metrics (Recall@K and MRR) computed on the same cosine ranking --
     the numbers the SSL / embedding-quality literature (DINO, DINOv2,
     MoCo, MAE) actually reports.

Pipeline output contract (HF image-feature-extraction):
    pipe(image) -> [[[float, ...]]]   shape: [1, num_tokens, hidden_dim]
    The first token (index 0) is the CLS token — the image-level embedding.

Ground-truth dataset (default: timm/mini-imagenet):
    {"image": PIL.Image, "label": ClassLabel}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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
        if pipe.image_processor is not None and input_shapes and len(input_shapes[0]) == 4:
            _, _, h, w = input_shapes[0]
            # Runtime-settable processor attribute; not on the base class.
            pipe.image_processor.size = {"height": h, "width": w}  # type: ignore[attr-defined]

        return pipe

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """No-op: kNN uses dataset labels directly, no model-side label mapping."""
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run kNN evaluation and return accuracy + retrieval metrics.

        Returns:
            ``knn_top1_accuracy`` and ``knn_top5_accuracy`` (classification
            accuracy via distance-weighted kNN majority vote), plus
            ``recall_at_1`` / ``recall_at_5`` / ``recall_at_10`` (fraction
            of queries whose top-K cosine neighbours contain a same-class
            item) and ``mrr`` (mean reciprocal rank of the first same-class
            neighbour).  All accuracy figures are percentages in
            ``[0, 100]``; recall and MRR are in ``[0, 1]``.
        """
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

        knn_result = KNNAccuracyMetric(k=10).compute(embeddings_array, labels_array)
        retrieval_result = self._compute_retrieval_metrics(embeddings_array, labels_array)
        return {**knn_result, **retrieval_result}

    @staticmethod
    def _compute_retrieval_metrics(
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        """Compute Recall@{1, 5, 10} + MRR on the leave-one-out cosine ranking.

        Uses the same L2-normalisation and self-exclusion as
        :class:`~winml.modelkit.eval.metrics.KNNAccuracyMetric` so the
        rankings driving the two report families are consistent -- the
        retrieval numbers describe *the same neighbour ordering* the kNN
        classifier voted on.

        For every query, a same-class neighbour is treated as the single
        relevant match (hit@K / rank-of-first-hit).  This matches the
        classification-as-retrieval convention used across DINO, DINOv2,
        MoCo and MAE evaluations.
        """
        from .metrics.mean_reciprocal_rank import MeanReciprocalRankMetric
        from .metrics.recall_at_k import RecallAtKMetric

        # L2-normalise with an eps floor to guard degenerate embeddings.
        # Matches KNNAccuracyMetric so the ranking is bit-identical.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        normalized = embeddings / norms

        similarity = normalized @ normalized.T
        np.fill_diagonal(similarity, -np.inf)  # exclude self

        # Full descending sort so MRR can find hits at arbitrary rank.
        # Self naturally sits at the last position (its -inf became +inf
        # under negation) so slicing off the tail drops the self entry.
        ranked_indices = np.argsort(-similarity, axis=1)[:, :-1]
        ranked_labels = labels[ranked_indices]

        recall = RecallAtKMetric(k_values=(1, 5, 10))
        mrr = MeanReciprocalRankMetric()
        for i in range(len(labels)):
            query_label = int(labels[i])
            recall.update(ranked_labels[i], query_label)
            mrr.update(ranked_labels[i], query_label)

        # Merge into a flat dict; drop the duplicate ``n_samples`` keys
        # (both metrics report the same count -- the outer evaluator
        # already reports it via KNNAccuracyMetric-adjacent bookkeeping).
        result = recall.compute()
        result.pop("n_samples", None)
        mrr_result = mrr.compute()
        mrr_result.pop("n_samples", None)
        result.update(mrr_result)
        return result

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
            return cast("np.ndarray", tokens[0])
        raise ValueError(
            f"Unsupported image-feature-extraction output shape: {np.asarray(raw).shape}. "
            "Expected [1, hidden] (pooled) or [1, num_tokens, hidden] (token sequence)."
        )
