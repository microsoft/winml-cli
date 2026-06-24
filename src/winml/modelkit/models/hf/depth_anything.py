# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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
from optimum.utils import DEFAULT_DUMMY_SHAPES, NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator

from ...export import register_onnx_overwrite


class _DepthAnythingVisionInputGenerator(DummyVisionInputGenerator):  # type: ignore[misc]  # optimum/transformers base is untyped
    """Vision input generator that lets explicit height/width override config.image_size.

    Optimum's DummyVisionInputGenerator prioritizes normalized_config.image_size
    (resolved here from backbone_config.image_size) over explicit height/width
    kwargs. When the user supplies a non-default shape via --shape-config (e.g.
    to match a non-square dataset), this subclass restores the override behavior
    so user kwargs take precedence. Mirrors the pattern used in
    `_SegformerVisionInputGenerator`.
    """

    def __init__(
        self,
        task: str,
        normalized_config,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        **kwargs,
    ):
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            num_channels=num_channels,
            width=width,
            height=height,
            **kwargs,
        )
        # If caller passed non-default height/width (e.g. from --shape-config),
        # use those instead of the backbone config's pretraining resolution.
        if height != DEFAULT_DUMMY_SHAPES["height"] or width != DEFAULT_DUMMY_SHAPES["width"]:
            self.height = height
            self.width = width
            self.image_size = (height, width)


@register_onnx_overwrite("depth_anything", "depth-estimation", library_name="transformers")
class DepthAnythingIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum/transformers base is untyped
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
    DUMMY_INPUT_GENERATOR_CLASSES = (_DepthAnythingVisionInputGenerator,)

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
