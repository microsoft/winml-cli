# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the quant finalizer registry.

Fast, offline: no model download, no ONNX Runtime. Verifies that the
``model_type`` -> quant policy dispatch (lazy import + decorator registration)
resolves the registered Qwen3 finalizer and falls back to ``None`` (the
quantizer's default DatasetCalibrationReader path) for everything else.
"""

from __future__ import annotations

from winml.modelkit.quant import get_quant_finalizer
from winml.modelkit.quant.calibration import QuantConfigFinalizer


def test_registered_model_type_resolves_finalizer():
    """The qwen3_transformer_only policy is found via lazy registry import."""
    finalizer = get_quant_finalizer("qwen3_transformer_only")
    assert finalizer is not None
    assert isinstance(finalizer, QuantConfigFinalizer)
    assert hasattr(finalizer, "finalize")
    # Registry returns the concrete policy class, not the generic protocol.
    assert type(finalizer).__name__ == "Qwen3TransformerOnlyQuantFinalizer"


def test_unregistered_model_type_returns_none():
    """Unknown / native model types have no policy -> default reader path."""
    assert get_quant_finalizer("resnet") is None
    assert get_quant_finalizer("qwen3") is None


def test_none_model_type_returns_none():
    """A missing model_type must not raise and must not dispatch a policy."""
    assert get_quant_finalizer(None) is None
    assert get_quant_finalizer("") is None
