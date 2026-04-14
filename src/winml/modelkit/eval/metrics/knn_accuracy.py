# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""k-Nearest Neighbor accuracy metric for image feature extraction.

Evaluates embedding quality by using a leave-one-out kNN classifier:
  1. For each sample, find its k nearest neighbors by cosine similarity.
  2. Predict label via distance-weighted majority vote among neighbors.
  3. Report top-1 and top-5 kNN classification accuracy.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_DEFAULT_K = 10


class KNNAccuracyMetric:
    """k-Nearest Neighbor classification accuracy on embeddings.

    Typical usage::

        metric = KNNAccuracyMetric(k=10)
        result = metric.compute(embeddings, labels)
        # {"knn_top1_accuracy": 72.5, "knn_top5_accuracy": 91.3}
    """

    def __init__(self, k: int = _DEFAULT_K) -> None:
        self.k = k

    def compute(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        """Compute kNN accuracy.

        Args:
            embeddings: (N, D) float array of L2-normalized embeddings.
            labels: (N,) int array of ground-truth class labels.

        Returns:
            Dict with ``knn_top1_accuracy`` and ``knn_top5_accuracy``
            as percentages in [0, 100].

        Raises:
            ValueError: If fewer than 2 samples or k < 1.
        """
        n = len(embeddings)
        if n < 2:
            raise ValueError(f"At least 2 samples required for kNN, got {n}.")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}.")

        k = min(self.k, n - 1)
        top1_predictions, top5_predictions = self._predict_labels(
            embeddings, labels, k,
        )
        return self._compute_accuracy(top1_predictions, top5_predictions, labels)

    def _predict_labels(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Predict labels via kNN weighted voting.

        Args:
            embeddings: (N, D) float array.
            labels: (N,) int array of ground-truth class labels (used as
                neighbor labels for voting, not for accuracy).
            k: Number of neighbors to use.

        Returns:
            Tuple of (top1_predictions, top5_predictions):
                - top1_predictions: (N,) int array of predicted labels.
                - top5_predictions: list of N lists, each containing up to 5
                  class labels ranked by vote weight (descending).
        """
        # L2-normalize (no-op if already normalized, safe either way)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        embeddings = embeddings / norms

        # Cosine similarity matrix: (N, N)
        similarity = embeddings @ embeddings.T

        # Exclude self-similarity
        np.fill_diagonal(similarity, -np.inf)

        # Top-k neighbor indices per sample
        # argpartition is O(N) per row vs O(N log N) for full sort
        top_k_indices = np.argpartition(similarity, -k, axis=1)[:, -k:]

        n = len(embeddings)
        top5_k = min(5, k)
        top1_predictions = np.empty(n, dtype=np.int64)
        top5_predictions: list[list[int]] = []

        for i in range(n):
            neighbor_idx = top_k_indices[i]
            neighbor_sims = similarity[i, neighbor_idx]
            neighbor_labels = labels[neighbor_idx]

            # Sort neighbors by similarity (descending)
            sorted_order = np.argsort(-neighbor_sims)
            sorted_labels = neighbor_labels[sorted_order]
            sorted_sims = neighbor_sims[sorted_order]

            # Weighted vote
            vote_weights: dict[int, float] = {}
            for label, sim in zip(sorted_labels, sorted_sims, strict=True):
                label_int = int(label)
                vote_weights[label_int] = vote_weights.get(label_int, 0.0) + float(sim)

            ranked = sorted(vote_weights, key=lambda c: vote_weights[c], reverse=True)
            top1_predictions[i] = ranked[0]
            top5_predictions.append(ranked[:top5_k])

        return top1_predictions, top5_predictions

    @staticmethod
    def _compute_accuracy(
        top1_predictions: np.ndarray,
        top5_predictions: list[list[int]],
        labels: np.ndarray,
    ) -> dict[str, Any]:
        """Compute top-1 and top-5 accuracy from predictions.

        Args:
            top1_predictions: (N,) int array of predicted labels.
            top5_predictions: list of N lists of up to 5 candidate labels.
            labels: (N,) int array of ground-truth labels.

        Returns:
            Dict with ``knn_top1_accuracy`` and ``knn_top5_accuracy``
            as percentages in [0, 100].
        """
        n = len(labels)
        top1_correct = int(np.sum(top1_predictions == labels))
        top5_correct = sum(
            int(labels[i]) in top5_predictions[i] for i in range(n)
        )

        top1_acc = round(top1_correct / n * 100, 4)
        top5_acc = round(top5_correct / n * 100, 4)

        return {"knn_top1_accuracy": top1_acc, "knn_top5_accuracy": top5_acc}
