# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""DepthPro HuggingFace Model Configuration.

DepthPro uses multiple DINOv2 backbones (image, patch, fov) with multi-scale
patch processing for metric monocular depth estimation.

This module provides:
- DepthProIOConfig: ONNX export config for depth-estimation task

DepthPro requires a minimum input image size of patch_size / min(scaled_images_ratios)
(e.g., 384 / 0.25 = 1536). This is expressed as a computed property on a
NormalizedConfig subclass, so the standard DummyVisionInputGenerator picks it up
without needing a custom generator.

Note:
    No model-specific build config needed. The analyzer autoconf loop discovers
    optimization flags automatically. See issue #232.
"""

from __future__ import annotations

from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator

from ...export import register_onnx_overwrite


class _DepthProNormalizedConfig(NormalizedConfig):
    """Normalized config for DepthPro with computed image_size.

    image_size is derived from patch_size / min(scaled_images_ratios),
    since DepthPro's multi-scale processing requires a minimum input size
    that isn't stored as a single config field.
    """

    NUM_CHANNELS = "image_model_config.num_channels"

    @property
    def image_size(self) -> int:
        """Compute minimum valid input size from multi-scale parameters."""
        return int(self.config.patch_size / min(self.config.scaled_images_ratios))


@register_onnx_overwrite("depth_pro", "depth-estimation", library_name="transformers")
class DepthProIOConfig(OnnxConfig):
    """ONNX config for DepthPro depth estimation.

    Model: apple/DepthPro-hf
    model.config.model_type = "depth_pro"

    Inputs:
        - pixel_values: {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}

    Outputs:
        - predicted_depth: {0: "batch_size", 1: "height", 2: "width"}
        - field_of_view: {0: "batch_size"}
    """

    NORMALIZED_CONFIG_CLASS = _DepthProNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return input tensors for depth estimation."""
        return {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return output tensors for depth estimation."""
        return {
            "predicted_depth": {0: "batch_size", 1: "height", 2: "width"},
            "field_of_view": {0: "batch_size"},
        }
