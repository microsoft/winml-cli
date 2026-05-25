# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Mean Average Precision metric for object detection.

Wraps torchmetrics.detection.MeanAveragePrecision for use with
plain Python lists of boxes, scores, and labels. Handles tensor
conversion and box format normalization internally.
"""

from __future__ import annotations

from typing import Any

import torch


class MAPMetric:
    """COCO-standard mAP metric wrapping torchmetrics MeanAveragePrecision.

    Accepts raw Python lists for predictions and references,
    handles all tensor conversion and box format conversion internally.
    """

    def compute(
        self,
        predictions: list[dict[str, list]],
        references: list[dict[str, list]],
        box_format: str = "xywh",
        box_coords: str = "absolute",
    ) -> dict[str, Any]:
        """Compute COCO mAP metrics.

        Args:
            predictions: Per-image predictions. Each dict has:
                - "boxes": list of [xmin, ymin, xmax, ymax] (always xyxy)
                - "scores": list of confidence floats
                - "labels": list of integer label IDs
            references: Per-image ground truth. Each dict has:
                - "boxes": list of [x, y, w, h] or [x1, y1, x2, y2]
                - "labels": list of integer label IDs (-1 = excluded)
                - "image_size": (width, height) tuple, required if
                  box_coords is "normalized"
            box_format: Format of reference boxes — "xywh" or "xyxy".
                Prediction boxes are always xyxy.
            box_coords: Coordinate system of reference boxes —
                "absolute" (pixels) or "normalized" (0-1).

        Returns:
            Dict with keys: map, map_50, map_75, num_predictions,
            num_ground_truths, num_images, plus additional scalar
            metrics from torchmetrics.
        """
        from torchmetrics.detection import MeanAveragePrecision

        target_list = [self._convert_target(ref, box_format, box_coords) for ref in references]
        pred_list = [self._convert_prediction(pred) for pred in predictions]

        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        metric.update(preds=pred_list, target=target_list)
        results = metric.compute()

        metrics: dict[str, Any] = {
            "map": results["map"].item(),
            "map_50": results["map_50"].item(),
            "map_75": results["map_75"].item(),
            "num_predictions": sum(p["boxes"].shape[0] for p in pred_list),
            "num_ground_truths": sum(t["boxes"].shape[0] for t in target_list),
            "num_images": len(references),
        }
        for key, value in results.items():
            if key not in metrics and isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[key] = value.item()

        return metrics

    @staticmethod
    def _convert_target(
        ref: dict[str, Any],
        box_format: str,
        box_coords: str,
    ) -> dict[str, torch.Tensor]:
        """Convert one image's ground truth to torchmetrics format."""
        raw_boxes = ref.get("boxes", [])
        raw_labels = ref.get("labels", [])

        if not raw_boxes:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        # Filter out unmapped labels (-1)
        filtered = [
            (box, lbl) for box, lbl in zip(raw_boxes, raw_labels, strict=False) if lbl != -1
        ]

        if not filtered:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        filtered_boxes, filtered_labels = zip(*filtered, strict=False)
        boxes = torch.tensor(list(filtered_boxes), dtype=torch.float32)
        labels = torch.tensor(filtered_labels, dtype=torch.int64)

        if box_format == "xywh":
            boxes = _xywh_to_xyxy(boxes)

        if box_coords == "normalized":
            image_size = ref.get("image_size")
            if image_size is not None:
                w, h = image_size
                boxes = _denormalize_boxes(boxes, w, h)

        return {"boxes": boxes, "labels": labels}

    @staticmethod
    def _convert_prediction(pred: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert one image's predictions to torchmetrics format."""
        raw_boxes = pred.get("boxes", [])
        raw_scores = pred.get("scores", [])
        raw_labels = pred.get("labels", [])

        if not raw_boxes:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        return {
            "boxes": torch.tensor(raw_boxes, dtype=torch.float32),
            "scores": torch.tensor(raw_scores, dtype=torch.float32),
            "labels": torch.tensor(raw_labels, dtype=torch.int64),
        }


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x, y, width, height] to [xmin, ymin, xmax, ymax]."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def _denormalize_boxes(
    boxes: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Convert normalized [0,1] boxes to absolute pixel coordinates."""
    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
    return boxes * scale
