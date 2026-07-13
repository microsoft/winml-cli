# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Model-type-specific quantization policies (calibration readers + schemes).

This subpackage stays import-light on purpose: it exposes only the registry
API. The individual finalizer modules (which pull in torch/transformers) are
imported lazily by :func:`get_quant_finalizer` when their ``model_type`` is
quantized.
"""

from __future__ import annotations

from .base import QuantConfigFinalizer
from .registry import QUANT_FINALIZERS, get_quant_finalizer, has_quant_finalizer


__all__ = [
    "QUANT_FINALIZERS",
    "QuantConfigFinalizer",
    "get_quant_finalizer",
    "has_quant_finalizer",
]
