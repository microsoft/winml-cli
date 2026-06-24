# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""MGP-STR (Multi-Granularity Prediction for Scene Text Recognition) HuggingFace Model Configuration.

MGP-STR is a Vision Transformer-based scene text recognition (STR) model. The
upstream ``MgpstrForSceneTextRecognition`` head produces three logit tensors —
``char_logits``, ``bpe_logits``, ``wp_logits`` — at three granularities
(character / byte-pair / word-piece), which the ``MgpstrProcessor`` combines
into the final decoded string.

The vendor ``MgpstrOnnxConfig`` (Optimum) already exposes the 3-head outputs
correctly but is registered ONLY under the ``feature-extraction`` task. End
users naturally reach for the ``image-to-text`` task label for STR work; this
module registers the same export config under ``image-to-text`` so the
user-facing task resolves cleanly.

This is an Effort-L1-light contribution per the `adding-model-support` skill:
no new ONNX-export logic, just a task-label alias + HF model-class binding.
"""

from __future__ import annotations

from optimum.exporters.onnx.model_configs import MgpstrOnnxConfig
from transformers import MgpstrForSceneTextRecognition

from ...export import register_onnx_overwrite


# =============================================================================
# Image-to-text alias for MGP-STR
# =============================================================================


@register_onnx_overwrite("mgp-str", "image-to-text", library_name="transformers")
class MgpstrImage2TextOnnxConfig(MgpstrOnnxConfig):
    """MGP-STR ONNX config bound to the ``image-to-text`` task.

    The 3-head ``(char_logits, bpe_logits, wp_logits)`` output contract and
    the ``pixel_values`` input contract are inherited unchanged from
    ``MgpstrOnnxConfig``. The only purpose of this subclass is to register
    the same export semantics under the ``image-to-text`` task name so users
    can build MGP-STR with the natural task label.
    """


# =============================================================================
# Model Class Mapping
# =============================================================================

# (model_type, task) -> HF model class. Binds the ``image-to-text`` task on
# MGP-STR to ``MgpstrForSceneTextRecognition`` (the head-bearing class with the
# 3-granularity outputs), instead of letting the loader fall back to
# ``AutoModelForVision2Seq`` — MGP-STR is NOT a Vision2Seq architecture.
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("mgp-str", "image-to-text"): MgpstrForSceneTextRecognition,
}
