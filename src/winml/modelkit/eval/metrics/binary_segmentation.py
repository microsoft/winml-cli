# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Binary-segmentation metrics for promptable mask-generation.

Unlike :class:`MeanIoUMetric` (semantic, multi-class, pixel-level), this
metric operates on a *per-instance* binary prediction / GT pair (each
sample is one prompted mask).  It computes:

* **mIoU**: arithmetic mean of per-sample Intersection-over-Union.
* **Dice**: arithmetic mean of per-sample Dice coefficient
  (2 * |P ∩ G| / (|P| + |G|)).

Both are dataset-level (macro) means -- one number per sample, then
averaged.  This matches the canonical SAM / mask-generation reporting
convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class BinarySegmentationMetric:
    """Per-instance mIoU + Dice for promptable mask generation.

    Incremental: call :meth:`update` per sample, then :meth:`compute` once.
    Stores only running sums (O(1) memory per added sample).
    """

    def __init__(self) -> None:
        self._iou_sum = 0.0
        self._dice_sum = 0.0
        self._count = 0
        # Track empty-GT samples separately so a degenerate dataset doesn't
        # silently inflate the score.  An empty GT can only score IoU/Dice
        # of either 0 (any positive prediction) or undefined (empty pred),
        # so we exclude them and surface the skip count to the caller.
        self._skipped = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Add one (pred, gt) pair to the running totals.

        Both must be 2-D and the same shape.  Any nonzero value is treated
        as foreground.  Empty-GT samples are counted in ``skipped`` and do
        not contribute to mIoU / Dice.
        """
        if pred.shape != gt.shape:
            raise ValueError(
                f"pred shape {pred.shape} != gt shape {gt.shape}",
            )
        pred_b = pred.astype(bool)
        gt_b = gt.astype(bool)
        gt_pos = int(gt_b.sum())
        if gt_pos == 0:
            self._skipped += 1
            return

        inter = int(np.logical_and(pred_b, gt_b).sum())
        union = int(np.logical_or(pred_b, gt_b).sum())
        pred_pos = int(pred_b.sum())

        iou = inter / union if union > 0 else 0.0
        dice = (2.0 * inter) / (pred_pos + gt_pos) if (pred_pos + gt_pos) > 0 else 0.0

        self._iou_sum += iou
        self._dice_sum += dice
        self._count += 1

    def compute(self) -> dict[str, Any]:
        """Return the aggregated metrics.

        Always includes ``num_samples`` and ``num_skipped`` so the caller
        can detect when too many samples were filtered out.
        """
        if self._count == 0:
            return {
                "mIoU": 0.0,
                "dice": 0.0,
                "num_samples": 0,
                "num_skipped": self._skipped,
            }
        return {
            "mIoU": self._iou_sum / self._count,
            "dice": self._dice_sum / self._count,
            "num_samples": self._count,
            "num_skipped": self._skipped,
        }
