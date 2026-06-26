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
from typing import TYPE_CHECKING, Any, cast

from transformers.modeling_outputs import DepthEstimatorOutput

from .base import WinMLPreTrainedModel


if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class WinMLModelForDepthEstimation(WinMLPreTrainedModel):
    """WinML model for monocular depth estimation.

    Returns ``DepthEstimatorOutput`` with ``predicted_depth`` so HF's
    depth-estimation pipeline can run post-processing via
    ``image_processor.post_process_depth_estimation()``.
    """

    def forward(self, **kwargs: Any) -> DepthEstimatorOutput:
        """Run depth estimation inference.

        Accepts all processor outputs via ``**kwargs`` and passes them
        directly to the ONNX session, keeping the implementation
        architecture-agnostic.

        Returns:
            DepthEstimatorOutput with the ``predicted_depth`` tensor populated.
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
        return DepthEstimatorOutput(predicted_depth=depth)
