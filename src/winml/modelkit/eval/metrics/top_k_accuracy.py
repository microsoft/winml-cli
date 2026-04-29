# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Top-k accuracy metric for zero-shot image classification.

Computes top-k accuracy by comparing a ranked list of predicted labels
against a single ground-truth label. The values of ``k`` are configurable
via the constructor.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


class TopKAccuracyMetric:
    """Incremental top-k accuracy metric.

    Tracks correct predictions over multiple update() calls and returns
    accuracy percentages on compute() for each configured ``k``.

    Typical usage::

        metric = TopKAccuracyMetric(ks=(1, 5))
        for pred_labels, gt_label in samples:
            metric.update(pred_labels, gt_label)
        result = metric.compute()
        # {"top1_accuracy": 91.0, "top5_accuracy": 99.0}
    """

    def __init__(self, ks: Iterable[int] = (1, 5)) -> None:
        """Create a metric that tracks top-k accuracy for each k in ``ks``.

        Args:
            ks: Iterable of positive ``k`` values to track (e.g. ``[1]`` or
                ``[1, 5]``). Defaults to ``(1, 5)``.

        Raises:
            ValueError: If ``ks`` is empty or contains a non-positive value.
        """
        ks_tuple = tuple(ks)
        if not ks_tuple:
            raise ValueError("ks must contain at least one value.")
        if any(k <= 0 for k in ks_tuple):
            raise ValueError(f"ks must all be positive; got {ks_tuple!r}.")
        self._ks: tuple[int, ...] = ks_tuple
        self._correct: dict[int, int] = dict.fromkeys(ks_tuple, 0)
        self._total = 0

    def update(self, predicted_labels: list[str], ground_truth: str) -> None:
        """Record one sample.

        Args:
            predicted_labels: Labels ranked by score descending (best first).
            ground_truth: The true label string.
        """
        self._total += 1
        for k in self._ks:
            if ground_truth in predicted_labels[:k]:
                self._correct[k] += 1

    def compute(self) -> dict[str, Any]:
        """Return accuracy as percentages keyed by ``"top{k}_accuracy"``.

        Raises:
            ValueError: If no samples have been recorded.
        """
        if self._total == 0:
            raise ValueError("No samples recorded. Call update() before compute().")

        return {
            f"top{k}_accuracy": round(self._correct[k] / self._total * 100, 4)
            for k in self._ks
        }
