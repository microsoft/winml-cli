# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Mean IoU metric for semantic segmentation.

Wraps torchmetrics.classification.MulticlassJaccardIndex for incremental
mIoU computation from per-image label maps.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Sentinel value for pixels excluded from metric computation.
# Must be outside any valid class ID range (0..num_classes-1).
IGNORE_INDEX = -1


class MeanIoUMetric:
    """Mean Intersection-over-Union metric for semantic segmentation.

    Uses torchmetrics MulticlassJaccardIndex (Jaccard = IoU) with
    incremental update/compute pattern for memory efficiency.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        """Initialize mIoU metric.

        Args:
            num_classes: Number of semantic classes (excluding background).
            ignore_index: Label index to ignore in metric computation.
        """
        from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

        self._num_classes = num_classes
        self._ignore_index = ignore_index

        self._iou_macro = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
            ignore_index=ignore_index,
        )
        self._iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="none",
            ignore_index=ignore_index,
        )
        self._accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="micro",
            ignore_index=ignore_index,
        )

    def update(
        self,
        prediction: np.ndarray,
        reference: np.ndarray,
    ) -> None:
        """Add one image's prediction and reference label maps.

        Args:
            prediction: (H, W) int array, 0-indexed class IDs.
            reference: (H, W) int array, 0-indexed class IDs.
                Pixels with ignore_index value are excluded from metric.
        """
        import torch

        pred_t = torch.from_numpy(prediction.astype(np.int64)).unsqueeze(0)
        ref_t = torch.from_numpy(reference.astype(np.int64)).unsqueeze(0)

        self._iou_macro.update(pred_t, ref_t)
        self._iou_per_class.update(pred_t, ref_t)
        self._accuracy.update(pred_t, ref_t)

    def compute(self) -> dict[str, Any]:
        """Compute final metrics.

        Returns:
            Dict with mean_iou, overall_accuracy, and per_category_iou.
        """
        mean_iou = self._iou_macro.compute().item()
        per_class_iou = self._iou_per_class.compute().numpy()
        overall_accuracy = self._accuracy.compute().item()

        return {
            "mean_iou": mean_iou,
            "overall_accuracy": overall_accuracy,
            "per_category_iou": per_class_iou.tolist(),
        }

    def reset(self) -> None:
        """Reset all accumulated state for a fresh evaluation."""
        self._iou_macro.reset()
        self._iou_per_class.reset()
        self._accuracy.reset()
