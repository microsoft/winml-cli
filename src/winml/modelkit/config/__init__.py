# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Build Configuration.

This module provides:
- WinMLBuildConfig: Pipeline configuration dataclass
- generate_build_config: Backward-compatible dispatcher
- generate_hf_build_config: Config from HuggingFace model (Scenarios A/B/C)
- generate_onnx_build_config: Config from pre-exported ONNX (Scenario D)

Example:
    >>> from winml.modelkit.config import WinMLBuildConfig, generate_build_config
    >>>
    >>> # Auto-generate complete config
    >>> config = generate_build_config("microsoft/resnet-50")
    >>> config.loader.task
    'image-classification'
    >>>
    >>> # Use dataclass directly
    >>> config = WinMLBuildConfig()
"""

from ..utils.config_utils import merge_config
from .build import (
    SubmoduleClassNotFoundError,
    WinMLBuildConfig,
    generate_build_config,
    generate_hf_build_config,
    generate_onnx_build_config,
    resolve_quant_compile_config,
)
from .precision import (
    PrecisionPolicy,
    is_quantized_precision,
    resolve_precision,
    resolve_quant_types,
)


__all__ = [
    "PrecisionPolicy",
    "SubmoduleClassNotFoundError",
    "WinMLBuildConfig",
    "generate_build_config",
    "generate_hf_build_config",
    "generate_onnx_build_config",
    "is_quantized_precision",
    "merge_config",
    "resolve_precision",
    "resolve_quant_compile_config",
    "resolve_quant_types",
]
