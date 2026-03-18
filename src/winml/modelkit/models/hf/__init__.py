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
from .clip import CLIP_CONFIG
from .clip import MODEL_CLASS_MAPPING as _CLIP_CLASS_MAPPING
from .convnext import ConvNextIOConfig as _ConvNextIOConfig  # triggers registration
from .depth_anything import DepthAnythingIOConfig as _DepthAnythingIOConfig  # triggers registration
from .detr import DETR_CONFIG
from .roberta import RobertaIOConfig as _RobertaIOConfig  # triggers registration
from .sam import MODEL_CLASS_MAPPING as _SAM2_CLASS_MAPPING


# Aggregated model class mappings: (model_type, task) -> HF model class
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    **_CLIP_CLASS_MAPPING,
    **_SAM2_CLASS_MAPPING,
}

# Registry: model_type -> WinMLBuildConfig
# Only models that need non-autoconf-discoverable settings retain configs.
# Models with only optim flags rely on the analyzer autoconf loop.
MODEL_BUILD_CONFIGS = {
    "bert": BERT_CONFIG,
    "blip": BLIP_CONFIG,
    "clip": CLIP_CONFIG,
    "clip-text-model": CLIP_CONFIG,
    "clip-vision-model": CLIP_CONFIG,
    "detr": DETR_CONFIG,
}

__all__ = [
    "MODEL_BUILD_CONFIGS",
    "MODEL_CLASS_MAPPING",
]
