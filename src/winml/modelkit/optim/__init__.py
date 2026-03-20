# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX Optimizer public API.

This package provides a capability-based system for ONNX graph optimization
using ONNX Runtime's optimization features.

Example:
    from winml.modelkit.optim import optimize_onnx

    # Basic usage
    model = optimize_onnx("model.onnx", gelu_fusion=True)

    # With config file
    model = optimize_onnx("model.onnx", config="optimize.json")

    # With WinMLOptimizationConfig
    from winml.modelkit.optim import WinMLOptimizationConfig
    config = WinMLOptimizationConfig(level="extended", gelu_fusion=True)
    model = optimize_onnx("model.onnx", **config.to_optimizer_kwargs())
"""

from __future__ import annotations

from .api import optimize_onnx
from .config import WinMLOptimizationConfig
from .errors import ConfigurationError, ModelValidationError, OptimizationError


__all__ = [
    "ConfigurationError",
    "ModelValidationError",
    "OptimizationError",
    "WinMLOptimizationConfig",
    "optimize_onnx",
]
