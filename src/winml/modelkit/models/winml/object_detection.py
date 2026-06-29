# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""WinML Object Detection Model.

Contains WinMLModelForObjectDetection for DETR-style object detection models.
Returns ObjectDetectionOutput with logits and pred_boxes, compatible with
HF image_processor.post_process_object_detection().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers.utils.generic import ModelOutput

from .base import WinMLPreTrainedModel


if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ObjectDetectionOutput(ModelOutput):
    """Output for object detection models.

    Compatible with image_processor.post_process_object_detection(),
    which reads:
        outputs.logits      — [B, num_queries, num_classes+1]
        outputs.pred_boxes  — [B, num_queries, 4]
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    pred_boxes: torch.Tensor | None = None


class WinMLModelForObjectDetection(WinMLPreTrainedModel):
    """WinML model for object detection.

    Mirrors HuggingFace AutoModelForObjectDetection.
    Returns ObjectDetectionOutput with logits and pred_boxes
    so that image_processor.post_process_object_detection() works.
    """

    def forward(  # type: ignore[override]  # HF-pipeline base uses generic **kwargs; task-specific signature
        self,
        pixel_values: torch.Tensor | np.ndarray,
        pixel_mask: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ObjectDetectionOutput:
        """Run object detection inference."""
        inputs: dict[str, Any] = {"pixel_values": pixel_values}
        # Only include pixel_mask if the ONNX model accepts it
        if pixel_mask is not None:
            accepted_inputs = set(self.io_config.get("input_names", []))
            if "pixel_mask" in accepted_inputs:
                inputs["pixel_mask"] = pixel_mask

        formatted = self._format_inputs(**inputs)
        outputs = self._run_inference(formatted)

        return ObjectDetectionOutput(
            logits=outputs.get("logits"),
            pred_boxes=outputs.get("pred_boxes"),
        )

    @property
    def num_labels(self) -> int:
        """Number of detection classes."""
        if self.config is not None:
            return getattr(self.config, "num_labels", 91)
        return 91

    @property
    def id2label(self) -> dict[int, str]:
        """Mapping from label ID to label name."""
        if self.config is not None:
            return getattr(self.config, "id2label", {})
        return {}

    @property
    def label2id(self) -> dict[str, int]:
        """Mapping from label name to label ID."""
        if self.config is not None:
            return getattr(self.config, "label2id", {})
        return {}
