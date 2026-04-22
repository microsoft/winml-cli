# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Top-k accuracy metric for zero-shot image classification.

Computes top-1 and top-5 accuracy by comparing a ranked list of predicted
labels against a single ground-truth label.
"""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


class TopKAccuracyMetric:
    """Incremental top-k accuracy metric.

    Tracks correct predictions over multiple update() calls and returns
    top-1 and top-5 accuracy percentages on compute().

    Typical usage::

        metric = TopKAccuracyMetric()
        for pred_labels, gt_label in samples:
            metric.update(pred_labels, gt_label)
        result = metric.compute()
        # {"top1_accuracy": 91.0, "top5_accuracy": 99.0}
    """

    def __init__(self) -> None:
        self._correct_top1 = 0
        self._correct_top5 = 0
        self._total = 0

    def update(self, predicted_labels: list[str], ground_truth: str) -> None:
        """Record one sample.

        Args:
            predicted_labels: Labels ranked by score descending (best first).
            ground_truth: The true label string.
        """
        self._total += 1
        if predicted_labels and predicted_labels[0] == ground_truth:
            self._correct_top1 += 1
        if ground_truth in predicted_labels[:5]:
            self._correct_top5 += 1

    def compute(self) -> dict[str, Any]:
        """Return top-1 and top-5 accuracy as percentages.

        Raises:
            ValueError: If no samples have been recorded.
        """
        if self._total == 0:
            raise ValueError("No samples recorded. Call update() before compute().")

        return {
            "top1_accuracy": round(self._correct_top1 / self._total * 100, 4),
            "top5_accuracy": round(self._correct_top5 / self._total * 100, 4),
        }
