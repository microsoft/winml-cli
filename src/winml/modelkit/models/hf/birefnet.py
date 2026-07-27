# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""BiRefNet HuggingFace model configuration.

BiRefNet is published as a custom ``AutoModelForImageSegmentation`` model for
dichotomous image segmentation. The checkpoint config exposes a non-standard
``model_type`` value (``SegformerForSemanticSegmentation``), so WinML needs an
explicit class mapping and ONNX I/O config for the image-segmentation task.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import DEFAULT_DUMMY_SHAPES, NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator
from transformers import AutoModelForImageSegmentation

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...export.config import WinMLExportConfig
from ...optim.config import WinMLOptimizationConfig


BIREFNET_MODEL_TYPE = "segformerforsemanticsegmentation"
BIREFNET_DEFAULT_IMAGE_SIZE = 1024


class BiRefNetImageSegmentationWrapper(nn.Module):
    """Export adapter for BiRefNet image segmentation.

    The remote model's public ``forward`` argument is named ``x`` and returns a
    list of scale predictions. WinML recipes expose the conventional
    ``pixel_values`` input and a single final ``pred_masks`` output.
    """

    def __init__(self, model: nn.Module, config: Any) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.config.model_type = BIREFNET_MODEL_TYPE

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs: Any
    ) -> BiRefNetImageSegmentationWrapper:
        """Load BiRefNet and wrap it for stable ONNX export."""
        kwargs.setdefault("torch_dtype", torch.float32)
        model = AutoModelForImageSegmentation.from_pretrained(model_name_or_path, **kwargs)
        model.float()
        wrapper = cls(model, model.config)
        wrapper.eval()
        return wrapper

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return the final-scale segmentation mask logits."""
        outputs = self.model(pixel_values)
        return cast("torch.Tensor", outputs[-1])


MODEL_CLASS_MAPPING: dict[tuple[str, str | None], type] = {
    (BIREFNET_MODEL_TYPE, "image-segmentation"): BiRefNetImageSegmentationWrapper,
    (BIREFNET_MODEL_TYPE, None): BiRefNetImageSegmentationWrapper,
}


BIREFNET_CONFIG = WinMLBuildConfig(
    export=WinMLExportConfig(dynamo=True, opset_version=19),
    optim=WinMLOptimizationConfig(),
)


class _BiRefNetVisionInputGenerator(DummyVisionInputGenerator):  # type: ignore[misc]  # optimum base is untyped
    """Vision input generator using BiRefNet's documented 1024x1024 shape.

    The HF checkpoint does not include ``preprocessor_config.json`` and its
    custom config has no ``image_size`` field. The model card's inference sample
    resizes images to 1024x1024 before calling ``BiRefNet.forward``.
    """

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = BIREFNET_DEFAULT_IMAGE_SIZE,
        height: int = BIREFNET_DEFAULT_IMAGE_SIZE,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            num_channels=num_channels,
            width=width,
            height=height,
            **kwargs,
        )
        self.height = height
        self.width = width
        self.image_size = (height, width)


@register_onnx_overwrite(BIREFNET_MODEL_TYPE, "image-segmentation", library_name="transformers")
class BiRefNetIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for BiRefNet dichotomous image segmentation.

    Inputs:
        - pixel_values: {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}

    Outputs:
        - pred_masks: final binary mask logits from the last decoder scale.
    """

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="num_channels",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (_BiRefNetVisionInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return BiRefNet input tensors."""
        return {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return BiRefNet output tensors."""
        return {
            "pred_masks": {0: "batch_size", 2: "height", 3: "width"},
        }
