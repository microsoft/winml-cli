# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""WinML keypoint-detection model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers.utils.generic import ModelOutput

from .base import WinMLPreTrainedModel


if TYPE_CHECKING:
    import numpy as np


@dataclass
class KeypointDetectionOutput(ModelOutput):
    """Output for ViTPose-style keypoint detection models."""

    loss: torch.Tensor | None = None
    heatmaps: torch.Tensor | None = None


class WinMLModelForKeypointDetection(WinMLPreTrainedModel):
    """WinML model for top-down pose estimation.

    Returns ``KeypointDetectionOutput`` with ``heatmaps`` so ViTPose image
    processors can run ``post_process_pose_estimation``.
    """

    def forward(  # type: ignore[override]
        self,
        pixel_values: torch.Tensor | np.ndarray,
        dataset_index: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> KeypointDetectionOutput:
        inputs: dict[str, Any] = {"pixel_values": pixel_values}
        accepted_inputs = set(self.io_config.get("input_names", []))
        if dataset_index is not None and "dataset_index" in accepted_inputs:
            inputs["dataset_index"] = dataset_index
        for name, value in kwargs.items():
            if value is not None and name in accepted_inputs:
                inputs[name] = value

        formatted = self._format_inputs(**inputs)
        outputs = self._run_inference(formatted)

        heatmaps = outputs.get("heatmaps")
        if heatmaps is None:
            heatmaps = next(iter(outputs.values()))
        return KeypointDetectionOutput(heatmaps=heatmaps)
