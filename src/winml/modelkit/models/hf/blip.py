# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""BLIP HuggingFace Model Configuration.

BLIP (Bootstrapping Language-Image Pre-training) configuration for image captioning.

Key specs:
- Vision: ViT-B/16, 384x384 input, 577 sequence (1 CLS + 576 patches)
- Text: BERT-based decoder, vocab 30524, max 512 positions

Optimization settings match WMK_blip production pipeline:
- GELU fusion enabled
- LayerNorm fusion enabled
- MatMul+Add fusion enabled (GEMM)
- Attention fusion disabled (NPU compatibility)

This module provides:
- BLIP_CONFIG: WinML build configuration with optimizations
- BlipCaptioningIOConfig: ONNX config for BLIP captioning tasks

Registered tasks: image-to-text, image-text-to-text
Both tasks use BlipForConditionalGeneration and share the same ONNX graph.
ONNX export traces forward() which always requires input_ids for the decoder,
even for pure captioning (image-to-text). The difference between the two tasks
is only in what the caller feeds as input_ids at inference time (BOS token vs
a text prompt).

Not supported:
- visual-question-answering: forward returns only vision encoder outputs, no decoder logits
- zero-shot-image-classification: BlipModel is deprecated by transformers

Todo:
    - KV cache support for autoregressive decoder inference
"""

from __future__ import annotations

from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator

from ...config import WinMLBuildConfig
from ...export import (
    MaxLengthTextInputGenerator,
    register_onnx_overwrite,
)
from ...optim import WinMLOptimizationConfig


# =============================================================================
# WinML Build Config
# =============================================================================

BLIP_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        gelu_fusion=True,
        layer_norm_fusion=True,
        matmul_add_fusion=True,
        clip_constant_values=True,
    ),
)


# =============================================================================
# Optimum ONNX Export Config Registration
# =============================================================================


@register_onnx_overwrite("blip", "image-to-text", library_name="transformers")
@register_onnx_overwrite("blip", "image-text-to-text", library_name="transformers")
class BlipCaptioningIOConfig(OnnxConfig):
    """ONNX config for BLIP captioning (model_type='blip').

    Optimum has no built-in BLIP support. This config provides:
    - NormalizedConfig with dotted paths to traverse nested sub-configs
    - Vision + text dummy input generators
    - Logits-only output (filters out image_embeds, last_hidden_state)

    BLIP has nested sub-configs (vision_config, text_config). NormalizedConfig
    supports dotted paths (e.g., "vision_config.image_size") to traverse them.

    Inputs:
        - pixel_values: {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}
        - input_ids: {0: "batch_size", 1: "sequence_length"}
        - attention_mask: {0: "batch_size", 1: "sequence_length"}

    Outputs:
        - logits: {0: "batch_size", 1: "sequence_length"}
    """

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="vision_config.num_channels",
        image_size="vision_config.image_size",
        vocab_size="text_config.vocab_size",
        sequence_length="text_config.max_position_embeddings",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyVisionInputGenerator,
        MaxLengthTextInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Vision + decoder text inputs for ONNX export tracing."""
        return {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Decoder logits output."""
        return {
            "logits": {0: "batch_size", 1: "sequence_length"},
        }
