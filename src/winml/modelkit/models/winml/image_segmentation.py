# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Image Segmentation Models.

Contains:
- WinMLModelForImageSegmentation: Panoptic/instance segmentation (DETR-style, accepts pixel_mask)
- WinMLModelForSemanticSegmentation: Pixel-level semantic segmentation (SegFormer/BEiT/DPT)

These mirror the HuggingFace distinction:
- AutoModelForImageSegmentation (panoptic, DETR) — needs logits + pred_masks + pred_boxes
- AutoModelForSemanticSegmentation (pixel-level, SegFormer/BEiT/DPT) — needs logits only

Pipeline execution (export/optimize/compile) is done by WinMLAutoModel factory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.utils.generic import ModelOutput

from .base import WinMLPreTrainedModel


if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImageSegmentationOutput(ModelOutput):
    """Output for panoptic/instance segmentation models (DETR-style).

    Compatible with image_processor.post_process_panoptic_segmentation()
    and post_process_instance_segmentation(), which read:
        outputs.logits      — [B, num_queries, num_classes+1]
        outputs.pred_masks  — [B, num_queries, H, W]

    Also compatible with post_process_object_detection() which reads:
        outputs.logits      — [B, num_queries, num_classes+1]
        outputs.pred_boxes  — [B, num_queries, 4]
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    pred_boxes: torch.Tensor | None = None
    pred_masks: torch.Tensor | None = None


class WinMLModelForImageSegmentation(WinMLPreTrainedModel):
    """WinML model for image segmentation (panoptic/instance, DETR-style).

    Mirrors HuggingFace AutoModelForImageSegmentation.
    Returns ImageSegmentationOutput with logits, pred_masks, and pred_boxes
    so that image_processor.post_process_panoptic_segmentation() works.

    Thin wrapper - only handles inference I/O.
    Pipeline execution is done by WinMLAutoModel factory.
    """

    def forward(  # type: ignore[override]  # HF-pipeline base uses generic **kwargs; task-specific signature
        self,
        pixel_values: torch.Tensor | np.ndarray,
        pixel_mask: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ImageSegmentationOutput:
        """Run panoptic/instance segmentation inference.

        Args:
            pixel_values: Image tensor (B, C, H, W)
            pixel_mask: Optional pixel mask for variable-size images (B, H, W)
            **kwargs: Additional arguments (ignored, for HF pipeline compatibility)

        Returns:
            ImageSegmentationOutput with logits, pred_masks, and pred_boxes
        """
        # Build inputs dict - only include non-None values
        inputs: dict[str, Any] = {"pixel_values": pixel_values}
        if pixel_mask is not None:
            inputs["pixel_mask"] = pixel_mask

        # Use base class helpers for validation, formatting, and inference
        formatted = self._format_inputs(**inputs)
        outputs = self._run_inference(formatted)

        return ImageSegmentationOutput(
            logits=outputs.get("logits"),
            pred_boxes=outputs.get("pred_boxes"),
            pred_masks=outputs.get("pred_masks"),
        )

    @property
    def num_labels(self) -> int:
        """Number of segmentation labels."""
        if self.config is not None:
            return getattr(self.config, "num_labels", 150)
        return 150

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


class WinMLModelForSemanticSegmentation(WinMLPreTrainedModel):
    """WinML model for semantic segmentation.

    For pixel-level semantic segmentation models (SegFormer, BEiT, DPT).
    Does NOT accept pixel_mask (that is a DETR/panoptic concept for ImageSegmentation).

    Mirrors HuggingFace AutoModelForSemanticSegmentation which is a completely
    separate class from AutoModelForImageSegmentation with zero model overlap.

    Thin wrapper - only handles inference I/O.
    Pipeline execution is done by WinMLAutoModel factory.
    """

    def forward(  # type: ignore[override]  # HF-pipeline base uses generic **kwargs; task-specific signature
        self,
        pixel_values: torch.Tensor | np.ndarray,
        **kwargs: Any,
    ) -> SemanticSegmenterOutput:
        """Run semantic segmentation inference.

        Args:
            pixel_values: Image tensor (B, C, H, W)
            **kwargs: Additional arguments (ignored, for HF pipeline compatibility)

        Returns:
            SemanticSegmenterOutput with logits of shape (B, num_labels, H, W)
        """
        # Use base class helpers for validation, formatting, and inference
        formatted = self._format_inputs(pixel_values=pixel_values)
        outputs = self._run_inference(formatted)

        # Get logits (by name or first output)
        logits = outputs.get("logits", next(iter(outputs.values())))

        # transformers' Output fields are annotated FloatTensor (legacy, over-narrow);
        # the ONNX session returns a real float Tensor.
        return SemanticSegmenterOutput(logits=cast("torch.FloatTensor", logits))

    @property
    def num_labels(self) -> int:
        """Number of segmentation labels."""
        if self.config is not None:
            return getattr(self.config, "num_labels", 150)
        return 150

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
