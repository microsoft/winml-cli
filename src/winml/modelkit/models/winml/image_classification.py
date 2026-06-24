# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLModelForImageClassification.

Thin wrapper for image classification inference.
Pipeline execution (export/optimize/compile) is done by WinMLAutoModel factory.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from transformers.modeling_outputs import ImageClassifierOutput

from .base import WinMLPreTrainedModel


if TYPE_CHECKING:
    import numpy as np
    import torch

logger = logging.getLogger(__name__)


class WinMLModelForImageClassification(WinMLPreTrainedModel):
    """WinML model for image classification.

    Thin wrapper - only handles inference I/O.
    Pipeline execution is done by WinMLAutoModel factory.
    """

    def forward(  # type: ignore[override]  # HF-pipeline base uses generic **kwargs; task-specific signature
        self,
        pixel_values: torch.Tensor | np.ndarray,
        **kwargs: Any,
    ) -> ImageClassifierOutput:
        """Run image classification inference.

        Args:
            pixel_values: Image tensor (B, C, H, W)
            **kwargs: Additional arguments (ignored, for HF pipeline compatibility)

        Returns:
            ImageClassifierOutput with logits
        """
        # Use base class helpers for validation, formatting, and inference
        inputs = self._format_inputs(pixel_values=pixel_values)
        outputs = self._run_inference(inputs)

        # Get logits (by name or first output)
        logits = outputs.get("logits", next(iter(outputs.values())))

        # transformers' Output fields are annotated FloatTensor (legacy, over-narrow);
        # the ONNX session returns a real float Tensor.
        return ImageClassifierOutput(logits=cast("torch.FloatTensor", logits))

    @property
    def num_labels(self) -> int:
        """Number of classification labels."""
        if self.config is not None:
            return getattr(self.config, "num_labels", 1000)
        return 1000

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
