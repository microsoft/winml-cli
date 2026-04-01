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

from importlib.metadata import PackageNotFoundError, version

from . import _warnings  # Configure warning filters before importing subpackages
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
