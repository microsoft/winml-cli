# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""SigLIP HuggingFace Model Configuration.

Mirrors CLIP's split-export configuration so SigLIP zero-shot-image-classification
models can be built as two separate encoders (vision + text). The composite
model class ``WinMLModelForZeroShotImageClassification`` (in ``models/winml/``)
handles both CLIP and SigLIP uniformly; this file only holds the SigLIP-specific
Optimum configs needed to produce those split encoder ONNX files.

SigLIP specifics vs. CLIP:
- text sequence_length = 64 (SigLIP default, CLIP is 77). Resolved via
  ``max_position_embeddings``.
- No ``SiglipTextModelWithProjection`` / ``SiglipVisionModelWithProjection``
  in ``transformers`` — SigLIP's projection is baked into the base classes.
  The projected embedding is surfaced as ``pooler_output`` in the exported
  ONNX; the composite model's output-key resolver handles this alongside
  CLIP's ``image_embeds`` / ``text_embeds`` naming.
"""

from __future__ import annotations

from optimum.exporters.onnx.model_configs import (
    SiglipTextOnnxConfig,
    SiglipVisionModelOnnxConfig,
)
from optimum.utils import NormalizedTextConfig
from transformers import SiglipTextModel, SiglipVisionModel

from ...config import WinMLBuildConfig
from ...export import MaxLengthTextInputGenerator, register_onnx_overwrite
from ...optim import WinMLOptimizationConfig


# =============================================================================
# HuggingFace Model Class Mapping
# =============================================================================
#
# TasksManager doesn't know to pick sub-encoder classes for SigLIP; these
# overrides tell our loader to use the split text/vision classes when the
# composite pipeline requests one sub-task at a time.
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("siglip", "feature-extraction"): SiglipTextModel,
    ("siglip", "image-feature-extraction"): SiglipVisionModel,
}


# =============================================================================
# WinML Build Config
# =============================================================================
SIGLIP_CONFIG = WinMLBuildConfig(
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
@register_onnx_overwrite("siglip_text_model", "feature-extraction", library_name="transformers")
class SiglipTextModelIOConfig(SiglipTextOnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for SiglipTextModel (text encoder only).

    Uses ``max_position_embeddings`` (64 for SigLIP) as the fixed sequence
    length so the exported ONNX matches what the HF pipeline produces when
    tokenizing with ``padding="max_length"``. Inputs stay at the Optimum
    default (``input_ids`` only — SigLIP's tokenizer does not emit
    ``attention_mask``). Outputs stay at the Optimum default
    (``last_hidden_state``, ``pooler_output``).
    """

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        sequence_length="max_position_embeddings",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (MaxLengthTextInputGenerator,)


@register_onnx_overwrite("siglip_vision_model", "feature-extraction", library_name="transformers")
class SiglipVisionModelIOConfig(SiglipVisionModelOnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for SiglipVisionModel (vision encoder only).

    Uses Optimum defaults; no overrides needed.
    """


__all__ = [
    "MODEL_CLASS_MAPPING",
    "SIGLIP_CONFIG",
    "SiglipTextModelIOConfig",
    "SiglipVisionModelIOConfig",
]
