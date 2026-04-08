# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WML ModelKit - Accelerate Model Deployment on WinML.

ModelKit provides tools for converting PyTorch models to optimized ONNX format
with support for QNN (Qualcomm Neural Processing SDK) and OpenVINO backends.

Key Features:
- Universal ONNX export with hierarchy preservation
- QNN-optimized configurations
- Static batch enforcement for hardware compatibility
- Model-agnostic design principles
- Automatic task detection and model selection

Usage:
    from winml.modelkit import WinMLAutoModel, WinMLBuildConfig

    # Auto-detect task and load model
    model = WinMLAutoModel.from_pretrained("microsoft/resnet-50")

    # With custom config
    from .optim import WinMLOptimizationConfig
    config = WinMLBuildConfig(
        optim=WinMLOptimizationConfig(gelu_fusion=True),
    )
    model = WinMLAutoModel.from_pretrained("facebook/convnext-tiny-224", config=config)
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .config import WinMLBuildConfig
    from .models import (
        WinMLAutoModel,
        WinMLModelForImageClassification,
        WinMLPreTrainedModel,
    )

try:
    __version__ = version("winml-modelkit")
except PackageNotFoundError:
    __version__ = "0.0.1.dev0"

__all__ = [
    "WinMLAutoModel",
    "WinMLBuildConfig",
    "WinMLModelForImageClassification",
    "WinMLPreTrainedModel",
    "__version__",
]

# Lazy imports — heavy ML dependencies (torch, transformers, optimum,
# diffusers) are only loaded when a symbol is actually accessed, so
# lightweight entry-points like ``winml sys`` stay fast.
_LAZY_IMPORT_MAP: dict[str, str] = {
    "WinMLBuildConfig": ".config",
    "WinMLAutoModel": ".models",
    "WinMLModelForImageClassification": ".models",
    "WinMLPreTrainedModel": ".models",
}

_warnings_configured = False


def __getattr__(name: str) -> object:
    global _warnings_configured
    module_path = _LAZY_IMPORT_MAP.get(name)
    if module_path is not None:
        # Configure warning filters before the first heavy import
        if not _warnings_configured:
            _warnings_configured = True
            from . import _warnings
        mod = importlib.import_module(module_path, __name__)
        attr = getattr(mod, name)
        # Cache on the module so __getattr__ is not called again
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
