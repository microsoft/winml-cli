# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLModelForImageToImage.

Thin wrapper for image-to-image inference (super-resolution, denoising, etc.).
Pipeline execution (export/optimize/compile) is done by WinMLAutoModel factory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers.utils import ModelOutput

from .base import WinMLPreTrainedModel


if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImageReconstructionOutput(ModelOutput):
    """Output for image-to-image models (super-resolution, denoising, etc.).

    Compatible with HF ImageToImagePipeline which reads outputs.reconstruction.
    """

    loss: torch.FloatTensor | None = None
    reconstruction: torch.FloatTensor | None = None


class WinMLModelForImageToImage(WinMLPreTrainedModel):
    """WinML model for image-to-image tasks.

    Covers: super-resolution, denoising, JPEG artifact removal, etc.
    Thin wrapper - only handles inference I/O.
    Pipeline execution is done by WinMLAutoModel factory.
    """

    def forward(
        self,
        pixel_values: torch.Tensor | np.ndarray,
        **kwargs: Any,
    ) -> ImageReconstructionOutput:
        """Run image-to-image inference.

        Args:
            pixel_values: Image tensor (B, C, H, W)
            **kwargs: Additional arguments (ignored, for HF pipeline compatibility)

        Returns:
            ImageReconstructionOutput with reconstruction tensor
        """
        inputs = self._format_inputs(pixel_values=pixel_values)
        outputs = self._run_inference(inputs)

        reconstruction = outputs.get("reconstruction", next(iter(outputs.values())))

        return ImageReconstructionOutput(reconstruction=reconstruction)
