# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX model state detection functions.

Canonical home for lightweight "is this model quantized / compiled?" checks
that only need a file path. Heavier inspection utilities live in
``modelkit.onnx.inspection``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .persistence import load_onnx


if TYPE_CHECKING:
    from pathlib import Path

    import onnx

logger = logging.getLogger(__name__)


def _load_model_lightweight(model_path: Path, operation: str) -> onnx.ModelProto:
    """Load an ONNX model without external data, with descriptive error context."""
    path_str = str(model_path)
    try:
        return load_onnx(path_str, load_weights=False, validate=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"ONNX model not found during {operation}: {model_path}") from None
    except Exception as e:
        raise ValueError(
            f"Failed to load ONNX model for {operation}: {model_path}. "
            f"Ensure the file is a valid ONNX model. Error: {e}"
        ) from e


def is_quantized_onnx(model_path: Path) -> bool:
    """Check if ONNX model is quantized (QDQ or QOperator format).

    Returns ``True`` for either:

    * **QDQ format** -- contains ``QuantizeLinear`` / ``DequantizeLinear``
      pairs around float ops (the default ``onnxruntime.quantization``
      output and the format QNN expects).
    * **QOperator format** -- contains fused integer ops such as
      ``ConvInteger``, ``MatMulInteger``, or ``QLinear*`` (used by
      ``QuantFormat.QOperator`` exports and by Hub repos like
      ``onnx-community/sam3-tracker-ONNX``).

    Both formats indicate the model is "already quantized" and the
    ``optimize`` + ``quantize`` build stages should be skipped.
    """
    model = _load_model_lightweight(model_path, "quantization check")
    from ..compiler import QUANTIZATION_OP_TYPES

    return any(n.op_type in QUANTIZATION_OP_TYPES for n in model.graph.node)


def is_compiled_onnx(model_path: Path) -> bool:
    """Check if ONNX model is pre-compiled (contains EPContext nodes)."""
    model = _load_model_lightweight(model_path, "compilation check")
    return any(n.op_type == "EPContext" for n in model.graph.node)
