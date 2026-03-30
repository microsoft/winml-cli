# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""ZoeDepth HuggingFace Model Configuration.

ZoeDepth uses a BEiT backbone with a DPT-style decoder head for metric
depth estimation. Supports NYU and KITTI depth ranges via domain_logits.

This module provides:
- ZoeDepthIOConfig: ONNX export config for depth-estimation task

The config resolves image_size and num_channels from the nested backbone_config
(BEiT) using NormalizedConfig dotted paths.

Note:
    No model-specific build config needed. The analyzer autoconf loop discovers
    optimization flags automatically. See issue #232.
"""

from __future__ import annotations

from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator

from ...export import register_onnx_overwrite


@register_onnx_overwrite("zoedepth", "depth-estimation", library_name="transformers")
class ZoeDepthIOConfig(OnnxConfig):
    """ONNX config for ZoeDepth depth estimation.

    Model: Intel/zoedepth-nyu-kitti
    model.config.model_type = "zoedepth"

    Inputs:
        - pixel_values: {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}

    Outputs:
        - predicted_depth: {0: "batch_size", 1: "height", 2: "width"}
        - domain_logits: {0: "batch_size"}
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
        """Return output tensors for depth estimation.

        Includes domain_logits only when the model has multiple bin
        configurations (e.g. NYU + KITTI).  Single-domain models do
        not produce domain_logits.
        """
        out: dict[str, dict[int, str]] = {
            "predicted_depth": {0: "batch_size", 1: "height", 2: "width"},
        }
        bin_cfgs = getattr(self._config, "bin_configurations", [])
        if len(bin_cfgs) > 1:
            out["domain_logits"] = {0: "batch_size"}
        return out
