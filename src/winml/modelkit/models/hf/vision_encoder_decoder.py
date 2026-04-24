# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from optimum.exporters.onnx.model_configs import VisionEncoderDecoderOnnxConfig

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...optim import WinMLOptimizationConfig


# =============================================================================
# WinML Build Config
# =============================================================================
VISION_ENCODER_DECODER_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        reshape_mergedreshape=True,
    ),
)


# =============================================================================
# ONNX task registration
# =============================================================================


@register_onnx_overwrite(
    "vision-encoder-decoder", "document-question-answering", library_name="transformers"
)
class VisionEncoderDecoderDocQAOnnxConfig(VisionEncoderDecoderOnnxConfig):
    """ONNX config for vision-encoder-decoder document-question-answering.

    Donut and similar seq2seq document models share the same graph for both
    image-to-text and document-question-answering: pixel_values as encoder input
    and decoder_input_ids as the prompt. The question text is embedded into
    decoder_input_ids by the processor at runtime, so the ONNX inputs are
    identical to the image-to-text configuration.
    """
