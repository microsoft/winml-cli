# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HuggingFace Model Configurations.

This package contains model-specific configurations for HuggingFace models,
including WinML build configs and ONNX export config registrations.

Note:
    Most models no longer need explicit optim configs. The analyzer autoconf
    loop discovers fusion flags automatically during build. See issue #232.
    Only models with non-autoconf-discoverable flags (e.g., clamp_constant_values)
    or custom export/loader overrides retain explicit configs.

    Model patches for ONNX export compatibility (ConvNeXT LayerNorm, SAM2
    window partition, etc.) are registered via Optimum's PATCHING_SPECS /
    _MODEL_PATCHER mechanism on each model's OnnxConfig. They are applied
    as a context manager during export only.

Exports:
- MODEL_BUILD_CONFIGS: Registry of model_type -> WinMLBuildConfig
- MODEL_CLASS_MAPPING: HF model class overrides (e.g., CLIP task->class mapping)
"""

from __future__ import annotations

# Import configs - importing triggers ONNX config registration with TasksManager
# ConvNeXT and SAM2 modules also register PATCHING_SPECS / _MODEL_PATCHER
# on their OnnxConfig classes at import time.
from .bert import BERT_CONFIG
from .blip import BLIP_CONFIG
from .blip import MODEL_CLASS_MAPPING as _BLIP_CLASS_MAPPING
from .blip import BlipDecoderIOConfig as _BlipDecoderIOConfig  # triggers registration
from .blip import BlipVisionEncoderIOConfig as _BlipVisionEncoderIOConfig  # triggers registration
from .clip import CLIP_CONFIG
from .clip import MODEL_CLASS_MAPPING as _CLIP_CLASS_MAPPING
from .convnext import ConvNextIOConfig as _ConvNextIOConfig  # triggers registration
from .depth_anything import DepthAnythingIOConfig as _DepthAnythingIOConfig  # triggers registration
from .depth_pro import DepthProIOConfig as _DepthProIOConfig  # triggers registration
from .detr import DETR_CONFIG
from .mu2 import MODEL_CLASS_MAPPING as _MU2_CLASS_MAPPING
from .mu2 import MU2_CONFIG
from .mu2 import Mu2DecoderIOConfig as _Mu2DecoderIOConfig  # triggers registration
from .mu2 import Mu2EncoderIOConfig as _Mu2EncoderIOConfig  # triggers registration
from .qwen import MODEL_CLASS_MAPPING as _QWEN_CLASS_MAPPING
from .qwen import QWEN_CONFIG
from .qwen import QwenGenIOConfig as _QwenGenIOConfig
from .qwen import QwenPrefillIOConfig as _QwenPrefillIOConfig
from .roberta import ROBERTA_FAMILY_CONFIG
from .roberta import RobertaIOConfig as _RobertaIOConfig  # triggers registration
from .sam import MODEL_CLASS_MAPPING as _SAM2_CLASS_MAPPING
from .segformer import MODEL_CLASS_MAPPING as _SEGFORMER_CLASS_MAPPING
from .segformer import SegformerIOConfig as _SegformerIOConfig  # triggers registration
from .t5 import MODEL_CLASS_MAPPING as _T5_CLASS_MAPPING
from .t5 import T5_CONFIG
from .t5 import T5DecoderIOConfig as _T5DecoderIOConfig  # triggers registration
from .t5 import T5EncoderIOConfig as _T5EncoderIOConfig  # triggers registration
from .vision_encoder_decoder import MODEL_CLASS_MAPPING as _VED_CLASS_MAPPING
from .vision_encoder_decoder import VISION_ENCODER_DECODER_CONFIG
from .vision_encoder_decoder import (
    VisionDecoderIOConfig as _VisionDecoderIOConfig,  # triggers registration
)
from .vision_encoder_decoder import VisionEncoderIOConfig as _VisionEncoderIOConfig
from .zoedepth import ZoeDepthIOConfig as _ZoeDepthIOConfig  # triggers registration


# Aggregated model class mappings: (model_type, task) -> HF model class
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    **_BLIP_CLASS_MAPPING,
    **_CLIP_CLASS_MAPPING,
    **_MU2_CLASS_MAPPING,
    **_QWEN_CLASS_MAPPING,
    **_SAM2_CLASS_MAPPING,
    **_SEGFORMER_CLASS_MAPPING,
    **_T5_CLASS_MAPPING,
    **_VED_CLASS_MAPPING,
}

# Registry: model_type -> WinMLBuildConfig
# Only models that need non-autoconf-discoverable settings retain configs.
# Models with only optim flags rely on the analyzer autoconf loop.
MODEL_BUILD_CONFIGS = {
    "bert": BERT_CONFIG,
    "blip": BLIP_CONFIG,
    "camembert": ROBERTA_FAMILY_CONFIG,
    "clip": CLIP_CONFIG,
    "clip-text-model": CLIP_CONFIG,
    "clip-vision-model": CLIP_CONFIG,
    "detr": DETR_CONFIG,
    "roberta": ROBERTA_FAMILY_CONFIG,
    "mu2": MU2_CONFIG,
    "qwen3": QWEN_CONFIG,
    "t5": T5_CONFIG,
    "vision-encoder-decoder": VISION_ENCODER_DECODER_CONFIG,
    "xlm-roberta": ROBERTA_FAMILY_CONFIG,
}

__all__ = [
    "MODEL_BUILD_CONFIGS",
    "MODEL_CLASS_MAPPING",
]
