# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Quantization passes sub-package."""

from .base import BaseQuantPass
from .fp16 import FP16Pass
from .rtn import RTNPass
from .static import StaticPass


__all__ = [
    "BaseQuantPass",
    "FP16Pass",
    "RTNPass",
    "StaticPass",
]
