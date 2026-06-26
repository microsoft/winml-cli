# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLIP HuggingFace Model Configuration.

Provides WinML build config and ONNX export configs for CLIP models.
CLIP benefits from GELU, LayerNorm, and GEMM fusion for optimal performance.

This module registers ONNX export configs with Optimum's TasksManager for:
- CLIPTextModelWithProjection (clip_text_model)
- CLIPVisionModelWithProjection (clip_vision_model)

Key Features:
- MODEL_CLASS_MAPPING: HuggingFace model class overrides for CLIP.
  TasksManager normalizes "image-feature-extraction" -> "feature-extraction",
  returning CLIPModel (combined). We override to get separate encoders.
- CLIPTextModelIOConfig: Uses max_position_embeddings as sequence_length (77 for CLIP).
- CLIPVisionModelIOConfig: Outputs image_embeds (projected) instead of pooler_output.

Exports:
    CLIP_CONFIG: WinMLBuildConfig for CLIP model optimization.
    MODEL_CLASS_MAPPING: Dict mapping (model_type, task) to HF model class name.
    CLIPTextModelIOConfig: ONNX config for CLIPTextModelWithProjection.
    CLIPVisionModelIOConfig: ONNX config for CLIPVisionModelWithProjection.
"""

from __future__ import annotations

from optimum.exporters.onnx.model_configs import (
    CLIPTextWithProjectionOnnxConfig,
    CLIPVisionModelOnnxConfig,
)
from optimum.utils import NormalizedTextConfig
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from ...config import WinMLBuildConfig
from ...export import MaxLengthTextInputGenerator, register_onnx_overwrite
from ...optim import WinMLOptimizationConfig


# =============================================================================
# HuggingFace Model Class Mapping
# =============================================================================

# HuggingFace model class mapping for CLIP
# (model_type, task) -> HuggingFace model class
#
# Why CLIP needs class mapping:
# CLIP sub-model task mappings.
# - "feature-extraction"       → CLIPTextModelWithProjection (text encoder)
# - "image-feature-extraction" → CLIPVisionModelWithProjection (vision encoder)
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("clip", "feature-extraction"): CLIPTextModelWithProjection,
    ("clip", "image-feature-extraction"): CLIPVisionModelWithProjection,
}


# =============================================================================
# WinML Build Config
# =============================================================================
CLIP_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        gelu_fusion=True,
        layer_norm_fusion=True,
        matmul_add_fusion=True,
        clamp_constant_values=True,
    ),
)


# =============================================================================
# Optimum ONNX Export Config Registrations
# =============================================================================
@register_onnx_overwrite("clip_text_model", "feature-extraction", library_name="transformers")
class CLIPTextModelIOConfig(CLIPTextWithProjectionOnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for CLIPTextModelWithProjection from transformers.

    Model: openai/clip-vit-base-patch32 (text encoder only)
    model.config.model_type = "clip_text_model"

    Inputs:
        - input_ids: {0: "batch_size", 1: "sequence_length"}
        - attention_mask: {0: "batch_size", 1: "sequence_length"}

    Outputs:
        - text_embeds: {0: "batch_size"}
        - last_hidden_state: {0: "batch_size", 1: "sequence_length"}

    Key difference from Optimum's default:
        - sequence_length = max_position_embeddings (77 for CLIP)
        - Instead of hardcoded 16
    """

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        sequence_length="max_position_embeddings",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (MaxLengthTextInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return input tensors with attention_mask exposed."""
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }


@register_onnx_overwrite("clip_vision_model", "feature-extraction", library_name="transformers")
class CLIPVisionModelIOConfig(CLIPVisionModelOnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for CLIPVisionModelWithProjection from transformers.

    Model: openai/clip-vit-base-patch32 (vision encoder only)
    model.config.model_type = "clip_vision_model"

    Inputs:
        - pixel_values: {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}

    Outputs:
        - image_embeds: {0: "batch_size"} (projected embeddings)
        - last_hidden_state: {0: "batch_size"}
    """

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return output tensors for CLIPVisionModelWithProjection.

        CLIPVisionModelWithProjection outputs image_embeds (projected)
        instead of pooler_output.
        """
        return {
            "image_embeds": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
        }


__all__ = [
    "CLIP_CONFIG",
    "MODEL_CLASS_MAPPING",
    "CLIPTextModelIOConfig",
    "CLIPVisionModelIOConfig",
]
