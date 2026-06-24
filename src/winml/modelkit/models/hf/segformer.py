# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Segformer HuggingFace Model Configuration.

Segformer uses a hierarchical Transformer encoder with lightweight MLP decoder
for semantic segmentation.

This module provides:
- SegformerIOConfig: ONNX export config for image-segmentation task
- MODEL_CLASS_MAPPING: Routes image-segmentation to AutoModelForSemanticSegmentation

"""

from __future__ import annotations

from optimum.exporters.onnx import OnnxConfig
from optimum.utils import DEFAULT_DUMMY_SHAPES, NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator
from transformers import AutoModelForSemanticSegmentation

from ...export import register_onnx_overwrite


# Segformer is registered under AutoModelForSemanticSegmentation, not
# AutoModelForImageSegmentation. Without this mapping, TasksManager defaults
# to AutoModelForImageSegmentation which doesn't recognize SegformerConfig.
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("segformer", "image-segmentation"): AutoModelForSemanticSegmentation,
}


class _SegformerVisionInputGenerator(DummyVisionInputGenerator):  # type: ignore[misc]  # optimum base is untyped
    """Vision input generator that uses preprocessor resolution over config.image_size.

    Optimum's DummyVisionInputGenerator prioritizes normalized_config.image_size
    (from config.json) over explicit height/width kwargs. For most models these
    agree, but Segformer's config.image_size=224 is the backbone pretraining
    resolution, not the finetuned inference resolution. The correct size comes
    from preprocessor_config.json (e.g. 512x512 for ADE20K finetuned models),
    which is passed as height/width kwargs by _populate_image_size_from_preprocessor().

    This subclass overrides the priority so that preprocessor-derived height/width
    take precedence when present.
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
        # Override: if caller passed non-default height/width (from preprocessor),
        # use those instead of config.image_size which is the backbone resolution.
        if height != DEFAULT_DUMMY_SHAPES["height"] or width != DEFAULT_DUMMY_SHAPES["width"]:
            self.height = height
            self.width = width
            self.image_size = (height, width)


@register_onnx_overwrite("segformer", "image-segmentation", library_name="transformers")
class SegformerIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for Segformer semantic segmentation.

    Model: nvidia/segformer-b0-finetuned-ade-512-512
    model.config.model_type = "segformer"

    Inputs:
        - pixel_values: {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}

    Outputs:
        - logits: {0: "batch_size", 1: "num_labels", 2: "height", 3: "width"}
    """

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        image_size="image_size",
        num_channels="num_channels",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (_SegformerVisionInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return input tensors for semantic segmentation."""
        return {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return output tensors for semantic segmentation."""
        return {
            "logits": {0: "batch_size", 1: "num_labels", 2: "height", 3: "width"},
        }
