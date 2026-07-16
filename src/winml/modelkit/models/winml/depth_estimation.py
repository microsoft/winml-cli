# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""WinML Model for Depth Estimation.

Thin wrapper for monocular depth estimation inference.
Pipeline execution (export/optimize/compile) is done by WinMLAutoModel factory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers.modeling_outputs import DepthEstimatorOutput

from .base import WinMLPreTrainedModel


logger = logging.getLogger(__name__)


@dataclass
class WinMLDepthEstimatorOutput(DepthEstimatorOutput):
    """Depth output with optional architecture-specific camera metadata."""

    field_of_view: torch.FloatTensor | None = None


class WinMLModelForDepthEstimation(WinMLPreTrainedModel):
    """WinML model for monocular depth estimation.

    Returns ``DepthEstimatorOutput`` with ``predicted_depth`` so HF's
    depth-estimation pipeline can run post-processing via
    ``image_processor.post_process_depth_estimation()``.
    """

    def forward(self, **kwargs: Any) -> WinMLDepthEstimatorOutput:
        """Run depth estimation inference.

        Accepts all processor outputs via ``**kwargs`` and passes them
        directly to the ONNX session, keeping the implementation
        architecture-agnostic.

        Returns:
            Depth output with ``predicted_depth`` and available camera metadata.
        """
        formatted = self._format_inputs(**kwargs)
        outputs = self._run_inference(formatted)

        predicted_depth = outputs.get("predicted_depth")
        if predicted_depth is None:
            # Fall back to first output for non-standard output names.
            predicted_depth = next(iter(outputs.values()))

        # transformers' Output fields are annotated FloatTensor (legacy, over-narrow);
        # the ONNX session returns a real float Tensor.
        depth: torch.FloatTensor = cast("torch.FloatTensor", predicted_depth)
        field_of_view: torch.FloatTensor | None = cast(
            "torch.FloatTensor | None", outputs.get("field_of_view")
        )
        return WinMLDepthEstimatorOutput(
            predicted_depth=depth,
            field_of_view=field_of_view,
        )
