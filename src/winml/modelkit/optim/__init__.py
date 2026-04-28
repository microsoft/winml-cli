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
from .optimizer import Optimizer
from .registry import (
    BoolCapability,
    ChoiceCapability,
    IntCapability,
    auto_enable_dependencies,
    validate,
    validate_dependencies,
)


__all__ = [
    "BoolCapability",
    "ChoiceCapability",
    "ConfigurationError",
    "IntCapability",
    "ModelValidationError",
    "OptimizationError",
    "Optimizer",
    "WinMLOptimizationConfig",
    "auto_enable_dependencies",
    "optimize_onnx",
    "validate",
    "validate_dependencies",
]


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "get_all_capabilities": (".pipes", "get_all_capabilities"),
}


def __getattr__(name: str):
    """Lazy-load pipe utilities that pull in heavy dependencies."""
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
