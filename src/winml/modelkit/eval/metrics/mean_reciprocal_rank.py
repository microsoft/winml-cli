# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Mean Reciprocal Rank (MRR) metric for retrieval and ranked-neighbor evaluation.

MRR reports the mean of the reciprocal of the rank at which the first
relevant item appears in each query's ranked list::

    MRR = (1/N) * Σᵢ 1 / rank_first_relevant(qᵢ)

Queries whose ranked list contains no relevant item contribute ``0`` to
the mean (equivalent to treating the rank of the first relevant item as
infinity).

Compared with :class:`~winml.modelkit.eval.metrics.RecallAtKMetric` --
which reports whether a relevant item is anywhere in the top K -- MRR is
sensitive to the *position* of the first hit and rewards ranking a
correct answer higher.  The two metrics are complementary and are
usually reported together in retrieval evaluations.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MeanReciprocalRankMetric:
    """Streaming Mean Reciprocal Rank over ranked prediction lists.

    Typical usage::

        metric = MeanReciprocalRankMetric()
        for ranked, gt in per_query:
            metric.update(ranked, gt)
        result = metric.compute()
        # {"mrr": 0.623, "n_samples": 100}

    ``update`` accepts the same input shape as
    :class:`~winml.modelkit.eval.metrics.RecallAtKMetric` so that a single
    ``(ranked_predictions, ground_truth)`` stream can drive both metrics.
    """

    def __init__(self) -> None:
        self._rr_sum = 0.0
        self._count = 0

    def update(
        self,
        ranked_predictions: np.ndarray,
        ground_truth: int | np.integer | np.ndarray | list[int] | tuple[int, ...] | set[int],
    ) -> None:
        """Record one query's ranked prediction list.

        Args:
            ranked_predictions: 1-D array of predicted item IDs (or labels),
                sorted in descending score order.
            ground_truth: A single relevant ID (``int``) or a collection of
                relevant IDs.  Empty collections are rejected -- a query
                with zero relevant items has undefined MRR.

        Raises:
            ValueError: If ``ranked_predictions`` is empty or the
                ground-truth collection is empty.
        """
        ranked = np.asarray(ranked_predictions).ravel()
        if ranked.size == 0:
            raise ValueError("ranked_predictions cannot be empty")

        if isinstance(ground_truth, (int, np.integer)):
            relevant: set[int] = {int(ground_truth)}
        else:
            relevant = {int(x) for x in np.asarray(list(ground_truth)).ravel()}
        if not relevant:
            raise ValueError("ground_truth must contain at least one relevant ID")

        # First-hit rank is 1-indexed.  A query with no hit contributes 0
        # (equivalent to 1/infinity) and does not raise -- callers reason
        # about MRR = 0 vs None distinctly.
        rr = 0.0
        for position, item in enumerate(ranked, start=1):
            if int(item) in relevant:
                rr = 1.0 / position
                break
        self._rr_sum += rr
        self._count += 1

    def compute(self) -> dict[str, Any]:
        """Return the mean reciprocal rank and the sample count.

        Returns:
            Dictionary with ``mrr`` (rounded to 4 decimals) and
            ``n_samples``.  ``mrr`` is ``None`` when no samples have been
            recorded.
        """
        if self._count == 0:
            return {"mrr": None, "n_samples": 0}
        return {
            "mrr": round(self._rr_sum / self._count, 4),
            "n_samples": self._count,
        }

    def reset(self) -> None:
        """Clear the accumulated reciprocal-rank sum."""
        self._rr_sum = 0.0
        self._count = 0
