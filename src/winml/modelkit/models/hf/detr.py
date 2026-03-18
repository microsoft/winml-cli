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
    this from preprocessor_config.json via _populate_image_size_from_preprocessor,
    so no custom OnnxConfig is needed. Override via --shape-config if desired.

Exports:
    DETR_CONFIG: WinMLBuildConfig with conv fusion flags
"""

from __future__ import annotations

from ...config import WinMLBuildConfig
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
