# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""DETR HuggingFace Model Configuration.

DETR (DEtection TRansformer) uses a ResNet backbone with BatchNorm layers.
After ONNX export, BN is folded into Mul+Add pairs. These can be further
absorbed into Conv weights via conv_bn/conv_mul/conv_add fusion, reducing
node count by ~15% (874 → 746 for DETR-ResNet-50).

Note:
    Autoconf discovers gelu_fusion, matmul_add_fusion, layer_norm_fusion.
    Conv fusions (conv_bn, conv_mul, conv_add) are NOT autoconf-discoverable
    and must be explicitly configured here. See issue #232.

Image size:
    DETR's preprocessor uses shortest_edge=800. The export pipeline reads
    this from preprocessor_config.json via _populate_image_size_from_preprocessor.

ONNX export:
    Optimum's built-in DETR/Table Transformer OnnxConfig exports only
    ``pixel_values``. For non-square images this causes large accuracy
    regressions because padded regions cannot be masked during inference.
    This module registers an override that adds optional ``pixel_mask``
    input and generates matching dummy input tensors during export.

Exports:
    DETR_CONFIG: WinMLBuildConfig with conv fusion flags
"""

from __future__ import annotations

from optimum.exporters.onnx.model_configs import DetrOnnxConfig, TableTransformerOnnxConfig
from optimum.utils.input_generators import DummyVisionInputGenerator

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...optim import WinMLOptimizationConfig


# DETR config: conv fusions for ResNet backbone BN folding.
# Autoconf handles gelu/layernorm/matmul_add; these are not autoconf-discoverable.
DETR_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        conv_bn_fusion=True,
        conv_mul_fusion=True,
        conv_add_fusion=True,
    ),
)


class _DetrPixelMaskMixin:
    """Shared pixel_mask input override for DETR-family ONNX export configs."""

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return input tensors including optional pixel_mask."""
        return {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "pixel_mask": {0: "batch_size", 1: "height", 2: "width"},
        }


class PixelMaskInputGenerator(DummyVisionInputGenerator):
    """Generate all-ones pixel masks with DETR-compatible int64 dtype."""

    SUPPORTED_INPUT_NAMES = ("pixel_mask",)

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        """Generate an all-ones int64 pixel mask for the current image batch."""
        del input_name, int_dtype, float_dtype
        return self.random_int_tensor(
            shape=[self.batch_size, self.height, self.width],
            min_value=1,
            max_value=2,
            framework=framework,
            dtype="int64",
        )


@register_onnx_overwrite(
    "detr",
    "feature-extraction",
    "object-detection",
    "image-segmentation",
    library_name="transformers",
)
class DetrIOConfig(_DetrPixelMaskMixin, DetrOnnxConfig):
    """DETR ONNX config override that adds optional pixel_mask input."""

    DUMMY_INPUT_GENERATOR_CLASSES = (
        PixelMaskInputGenerator,
        *DetrOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES,
    )


@register_onnx_overwrite(
    "table-transformer",
    "feature-extraction",
    "object-detection",
    library_name="transformers",
)
class TableTransformerIOConfig(_DetrPixelMaskMixin, TableTransformerOnnxConfig):
    """Table Transformer ONNX config override that adds optional pixel_mask input."""

    DUMMY_INPUT_GENERATOR_CLASSES = (
        PixelMaskInputGenerator,
        *TableTransformerOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES,
    )
