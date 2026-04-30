# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Models Package.

Provides WinMLAutoModel factory and task-specific model classes.

Architecture:
- WinMLAutoModel: Factory class that orchestrates model building pipeline
- WinMLPreTrainedModel: Base class for all inference models (HF compatible)
- WinMLModelFor*: Task-specific inference wrappers
- HF_MODEL_SPECIALIZATIONS: HuggingFace model class overrides (CLIP, etc.)

Usage:
    from winml.modelkit.models import WinMLAutoModel
    model = WinMLAutoModel.from_pretrained("microsoft/resnet-50")

    # Or import specific classes
    from winml.modelkit.models import WinMLModelForImageClassification
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .hf import MODEL_BUILD_CONFIGS

# HuggingFace model class mappings (aggregated from hf/ subpackage)
# Importing triggers ONNX config registration with Optimum's TasksManager
from .hf import MODEL_CLASS_MAPPING as HF_MODEL_CLASS_MAPPING
from .hf import MODEL_TASK_DEFAULTS as HF_MODEL_TASK_DEFAULTS

# Re-export from winml/ subpackage (WinML inference class mappings)
# These have no circular dependencies with loader/
from .winml import (
    TASK_TO_WINML_CLASS,
    WINML_MODEL_CLASS_MAPPING,
    ImageSegmentationOutput,
    WinMLModelForGenericTask,
    WinMLModelForImageClassification,
    WinMLModelForImageSegmentation,
    WinMLModelForObjectDetection,
    WinMLModelForSemanticSegmentation,
    WinMLModelForSequenceClassification,
    WinMLPreTrainedModel,
    get_supported_tasks,
    get_winml_class,
    register_specialization,
)


if TYPE_CHECKING:
    from .auto import WinMLAutoModel


# Lazy loading for modules that cause circular imports
# WinMLAutoModel imports from loader/, which imports from models/
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "WinMLAutoModel": (".auto", "WinMLAutoModel"),
}


def __getattr__(name: str):
    """Lazy load modules that would cause circular imports."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(set(list(globals()) + __all__))


__all__ = [
    "HF_MODEL_CLASS_MAPPING",
    "HF_MODEL_TASK_DEFAULTS",
    "MODEL_BUILD_CONFIGS",
    "TASK_TO_WINML_CLASS",
    "WINML_MODEL_CLASS_MAPPING",
    "ImageSegmentationOutput",
    "WinMLAutoModel",
    "WinMLModelForGenericTask",
    "WinMLModelForImageClassification",
    "WinMLModelForImageSegmentation",
    "WinMLModelForObjectDetection",
    "WinMLModelForSemanticSegmentation",
    "WinMLModelForSequenceClassification",
    "WinMLPreTrainedModel",
    "get_supported_tasks",
    "get_winml_class",
    "register_specialization",
]
