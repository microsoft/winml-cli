"""Depth Anything HuggingFace Model Configuration.

Depth Anything V2 uses a DINOv2 backbone with a DPT-style decoder head
for monocular depth estimation.

This module provides:
- DepthAnythingIOConfig: ONNX export config for depth-estimation task

The config resolves image_size and num_channels from the nested backbone_config
(DINOv2) using NormalizedConfig dotted paths.
"""

from __future__ import annotations

from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator

from ...export import register_onnx_overwrite


@register_onnx_overwrite("depth_anything", "depth-estimation", library_name="transformers")
class DepthAnythingIOConfig(OnnxConfig):
    """ONNX config for Depth Anything depth estimation.

    Model: depth-anything/Depth-Anything-V2-Small-hf
    model.config.model_type = "depth_anything"

    Inputs:
        - pixel_values: {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}

    Outputs:
        - predicted_depth: {0: "batch_size", 1: "height", 2: "width"}
    """

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        image_size="backbone_config.image_size",
        num_channels="backbone_config.num_channels",
        allow_new=True,
    )
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
        }
