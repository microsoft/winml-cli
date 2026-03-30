# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Roberta-family and MPNet HuggingFace Model Configuration.

Roberta/XLM-R/CamemBERT/MPNet set max_position_embeddings = usable_length + pad_token_id + 1
(e.g., 514 = 512 + 1 + 1). Using the raw value as sequence_length causes position index
out-of-bounds during ONNX export tracing ("index out of range in self").

This module registers OnnxConfig overrides that adjust max_position_embeddings to the
actual usable sequence length. Produces seq_len=512 for Roberta/MPNet (was 514) while
leaving BERT (512), CLIP (77), and DeBERTa (512) unchanged.

This module provides:
- RobertaIOConfig: ONNX export config for Roberta
- XLMRobertaIOConfig: ONNX export config for XLM-Roberta
- CamemBERTIOConfig: ONNX export config for CamemBERT
- MPNetIOConfig: ONNX export config for MPNet
"""

from __future__ import annotations

import logging

from optimum.exporters.onnx.model_configs import (
    COMMON_TEXT_TASKS,
    CamembertOnnxConfig,
    MPNetOnnxConfig,
    RobertaOnnxConfig,
    XLMRobertaOnnxConfig,
)
from optimum.utils import NormalizedTextConfig

from ...config import WinMLBuildConfig
from ...export import MaxLengthTextInputGenerator, register_onnx_overwrite
from ...optim import WinMLOptimizationConfig


logger = logging.getLogger(__name__)


# =============================================================================
# WinML Build Config
# =============================================================================

ROBERTA_FAMILY_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        clamp_constant_values=True,
    ),
)


# =============================================================================
# Position offset adjustment
# =============================================================================


def _adjust_position_embeddings(config) -> None:
    """Adjust max_position_embeddings for Roberta-style position offset.

    Roberta-family models define:
        max_position_embeddings = usable_length + pad_token_id + 1
    E.g., Roberta-base: 514 = 512 + 1 + 1 (pad_token_id=1).

    Adjusts config.max_position_embeddings in-place to the usable length.
    A sentinel prevents double-adjustment if the config is reused.
    """
    if getattr(config, "_position_offset_applied", False):
        logger.debug("Position offset already applied; skipping.")
        return

    if not hasattr(config, "max_position_embeddings"):
        logger.warning(
            "Config %s has no max_position_embeddings; skipping position offset adjustment.",
            type(config).__name__,
        )
        return

    pad_token_id = getattr(config, "pad_token_id", 0) or 0
    if pad_token_id > 0:
        original = config.max_position_embeddings
        adjusted = original - pad_token_id - 1
        if adjusted <= 0:
            raise ValueError(
                f"Position offset adjustment would produce non-positive "
                f"max_position_embeddings={adjusted} "
                f"(original={original}, pad_token_id={pad_token_id})"
            )
        config.max_position_embeddings = adjusted
        config._position_offset_applied = True
        logger.debug(
            "Adjusted max_position_embeddings: %d -> %d (pad_token_id=%d)",
            original,
            adjusted,
            pad_token_id,
        )


# =============================================================================
# Roberta-family OnnxConfig with position-offset-adjusted sequence_length
# =============================================================================


class _RobertaPositionOffsetMixin:
    """Mixin that adjusts max_position_embeddings and uses it as sequence_length.

    Shared by all Roberta-family IOConfigs (Roberta, XLM-Roberta, CamemBERT).
    Must be listed first in MRO to override __init__ and class attributes.
    """

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        sequence_length="max_position_embeddings",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (MaxLengthTextInputGenerator,)

    def __init__(self, config, task, **kwargs):
        _adjust_position_embeddings(config)
        super().__init__(config, task, **kwargs)


@register_onnx_overwrite("roberta", *COMMON_TEXT_TASKS, library_name="transformers")
class RobertaIOConfig(_RobertaPositionOffsetMixin, RobertaOnnxConfig):
    """Roberta OnnxConfig with position-offset-adjusted sequence_length.

    Inputs (same as DistilBERT — no token_type_ids):
        - input_ids: {0: "batch_size", 1: "sequence_length"}
        - attention_mask: {0: "batch_size", 1: "sequence_length"}

    Key difference from Optimum's default:
        - Adjusts max_position_embeddings to usable length (e.g., 514 -> 512)
        - sequence_length = adjusted max_position_embeddings
        - Prevents position index OOB during ONNX export tracing
    """


@register_onnx_overwrite("xlm-roberta", *COMMON_TEXT_TASKS, library_name="transformers")
class XLMRobertaIOConfig(_RobertaPositionOffsetMixin, XLMRobertaOnnxConfig):
    """XLM-Roberta OnnxConfig with position-offset-adjusted sequence_length."""


@register_onnx_overwrite("camembert", *COMMON_TEXT_TASKS, library_name="transformers")
class CamemBERTIOConfig(_RobertaPositionOffsetMixin, CamembertOnnxConfig):
    """CamemBERT OnnxConfig with position-offset-adjusted sequence_length."""


@register_onnx_overwrite("mpnet", *COMMON_TEXT_TASKS, library_name="transformers")
class MPNetIOConfig(_RobertaPositionOffsetMixin, MPNetOnnxConfig):
    """MPNet OnnxConfig with position-offset-adjusted sequence_length.

    MPNet, like Roberta-family models, sets:
        max_position_embeddings = usable_length + pad_token_id + 1
    E.g., all-mpnet-base-v2: 514 = 512 + 1 + 1 (pad_token_id=1).

    Using the raw value causes position index OOB during ONNX export tracing.
    This config adjusts max_position_embeddings to the usable sequence length.
    """
