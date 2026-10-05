# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Recall@K metric for retrieval and ranked-neighbor evaluations.

For each query, given a ranked list of predicted item IDs (or labels)
sorted by descending score and one or more ground-truth relevant IDs,
Recall@K reports the fraction of relevant items retrieved in the top K::

    recall@k(query) = |relevant ∩ ranked[:k]| / |relevant|

Two input shapes are supported:

* **Single-relevant** -- pass an ``int`` ground truth.  ``recall@k`` is
  ``1.0`` if the ground-truth ID appears in ``ranked[:k]`` else ``0.0``.
  This matches the classification-as-retrieval convention used in the SSL
  embedding literature (DINO, DINOv2, MoCo, MAE): for each query, count a
  hit if *any* neighbor with the correct label falls in the top K.
* **Multi-relevant** -- pass a 1-D array/tuple/set of relevant IDs.
  ``recall@k`` becomes the classical retrieval recall.

The metric is fully model-agnostic: it consumes ranked ID lists produced
by any similarity computation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_DEFAULT_K_VALUES: tuple[int, ...] = (1, 5, 10)


class RecallAtKMetric:
    """Streaming Recall@K over ranked prediction lists.

    Typical usage::

        metric = RecallAtKMetric(k_values=(1, 5, 10))
        for ranked, gt in per_query:
            metric.update(ranked, gt)
        result = metric.compute()
        # {"recall_at_1": 0.42, "recall_at_5": 0.71, "recall_at_10": 0.83,
        #  "n_samples": 100}
    """

    def __init__(self, k_values: tuple[int, ...] = _DEFAULT_K_VALUES) -> None:
        """Initialize the metric with the K values to report.

        Args:
            k_values: Non-empty iterable of positive integers.  Duplicates are
                collapsed and values are sorted ascending.  Defaults to
                ``(1, 5, 10)`` matching the SSL embedding-evaluation convention.

        Raises:
            ValueError: If ``k_values`` is empty or contains a non-positive
                value.
        """
        ks = tuple(sorted({int(k) for k in k_values}))
        if not ks:
            raise ValueError("k_values must be a non-empty iterable of positive ints")
        if any(k < 1 for k in ks):
            raise ValueError(f"k_values must all be >= 1, got {sorted(k_values)}")
        self._k_values: tuple[int, ...] = ks
        # Sum of per-query recall values, one running sum per K.
        self._recall_sums: dict[int, float] = dict.fromkeys(ks, 0.0)
        self._count = 0

    def update(
        self,
        ranked_predictions: np.ndarray,
        ground_truth: int | np.integer | np.ndarray | list[int] | tuple[int, ...] | set[int],
    ) -> None:
        """Record one query's ranked prediction list.

        Args:
            ranked_predictions: 1-D array of predicted item IDs (or labels),
                sorted in descending score order.  Anything that
                ``np.asarray`` accepts as a 1-D integer array works.
            ground_truth: A single relevant ID (``int``) or a collection of
                relevant IDs.  Passing an empty collection is rejected --
                a query with zero relevant items has undefined Recall@K.

        Raises:
            ValueError: If ``ranked_predictions`` is not 1-D or the
                ground-truth collection is empty.
        """
        ranked = np.asarray(ranked_predictions).ravel()
        if ranked.size == 0:
            raise ValueError("ranked_predictions cannot be empty")

        # Normalize ground truth to a set of ints for uniform handling.
        if isinstance(ground_truth, (int, np.integer)):
            relevant: set[int] = {int(ground_truth)}
        else:
            relevant = {int(x) for x in np.asarray(list(ground_truth)).ravel()}
        if not relevant:
            raise ValueError("ground_truth must contain at least one relevant ID")

        total_relevant = len(relevant)
        for k in self._k_values:
            top_k = ranked[:k]
            hits = sum(1 for item in top_k if int(item) in relevant)
            self._recall_sums[k] += hits / total_relevant
        self._count += 1

    def compute(self) -> dict[str, Any]:
        """Return mean Recall@K for every configured K, plus ``n_samples``.

        Returns:
            Dictionary with keys ``recall_at_{k}`` (rounded to 4 decimals) for
            every ``k`` in ``k_values``, plus ``n_samples``.  Every recall
            value is ``None`` when no samples have been recorded.
        """
        if self._count == 0:
            result: dict[str, Any] = {f"recall_at_{k}": None for k in self._k_values}
            result["n_samples"] = 0
            return result
        result = {
            f"recall_at_{k}": round(self._recall_sums[k] / self._count, 4) for k in self._k_values
        }
        result["n_samples"] = self._count
        return result

    def reset(self) -> None:
        """Clear all accumulated recall sums."""
        self._recall_sums = dict.fromkeys(self._k_values, 0.0)
        self._count = 0
