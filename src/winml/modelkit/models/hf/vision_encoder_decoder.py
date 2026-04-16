# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ...config import WinMLBuildConfig
from ...optim import WinMLOptimizationConfig
from ...quant import WinMLQuantizationConfig


# =============================================================================
# WinML Build Config
# =============================================================================

# Quantization op list: all standard ONNX compute ops plus Gelu.
# LayerNormalization is excluded because quantizing it causes all-zero
# outputs and DEVICE_LOST on OpenVINO NPU for large encoder-decoder
# graphs.
_QUANT_OPS = [
    "MatMul",
    "Gemm",
    "Conv",
    "ConvTranspose",
    "Add",
    "Mul",
    "Softmax",
    "Div",
    "Concat",
    "Slice",
    "Pad",
    "Transpose",
    "Reshape",
    "Gather",
    "Gelu",
]

VISION_ENCODER_DECODER_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        reshape_mergedreshape=True,
    ),
    quant=WinMLQuantizationConfig(
        op_types_to_quantize=_QUANT_OPS,
    ),
)
