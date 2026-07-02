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

import logging
from importlib.metadata import PackageNotFoundError, version


logging.getLogger(__name__).addHandler(logging.NullHandler())

# _warnings configures filters before any subpackage imports.
# transformers_compat arms a sys.meta_path hook — the shim fires lazily
# the first time anything imports optimum.*; lightweight commands
# (``winml sys``, ``winml --help``) never pay the transformers cost.
from . import _warnings  # noqa: I001
from . import transformers_compat  # noqa: I001

transformers_compat.arm()


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


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "WinMLBuildConfig": (".config", "WinMLBuildConfig"),
    "WinMLAutoModel": (".models", "WinMLAutoModel"),
    "WinMLPreTrainedModel": (".models", "WinMLPreTrainedModel"),
    "WinMLModelForImageClassification": (".models", "WinMLModelForImageClassification"),
}


def __getattr__(name: str):
    """Lazy-load heavy exports on first access (PEP 562).

    This avoids importing torch/transformers/optimum (~30s) when only
    lightweight operations are needed (e.g., ``winml --help``).
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy attributes in dir() for debugger/IPython compatibility."""
    return list(set(list(globals()) + __all__))
