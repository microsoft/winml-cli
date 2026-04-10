# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""BERT HuggingFace Model Configuration.

BERT benefits from GELU, LayerNorm, and GEMM fusion for optimal performance.

This module provides:
- BERT_CONFIG: WinML build configuration with optimizations
- BertIOConfig: ONNX export config using max_position_embeddings as sequence_length

The BertIOConfig registers with Optimum's TasksManager to override default BERT configs
for all supported tasks, using max_position_embeddings (e.g., 512) instead of hardcoded 16.
"""

from __future__ import annotations

from optimum.exporters.onnx.model_configs import (
    COMMON_TEXT_TASKS,
    BertOnnxConfig,
)
from optimum.utils import NormalizedTextConfig

from ...config import WinMLBuildConfig
from ...export import MaxLengthTextInputGenerator, register_onnx_overwrite
from ...optim import WinMLOptimizationConfig


# =============================================================================
# WinML Build Config
# =============================================================================

BERT_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        clamp_constant_values=True,
    ),
)


# =============================================================================
# BERT OnnxConfig with max_position_embeddings as sequence_length
# =============================================================================


@register_onnx_overwrite("bert", *COMMON_TEXT_TASKS, library_name="transformers")
class BertIOConfig(BertOnnxConfig):
    """BERT OnnxConfig using max_position_embeddings as sequence_length.

    Inputs:
        - input_ids: {0: "batch_size", 1: "sequence_length"}
        - attention_mask: {0: "batch_size", 1: "sequence_length"}
        - token_type_ids: {0: "batch_size", 1: "sequence_length"}

    Key difference from Optimum's default:
        - sequence_length = max_position_embeddings (e.g., 512 for BERT)
        - Instead of hardcoded 16
    """

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        sequence_length="max_position_embeddings",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (MaxLengthTextInputGenerator,)
