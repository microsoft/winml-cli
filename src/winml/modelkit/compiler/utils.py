"""Detection utilities for ONNX model state."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

# Canonical definition of ONNX QDQ operator types.
# Import this constant instead of redefining {"QuantizeLinear", "DequantizeLinear"}.
QDQ_OP_TYPES: frozenset[str] = frozenset({"QuantizeLinear", "DequantizeLinear"})


def needs_format_conversion(model_path: Path, ep: str) -> bool:
    """Check if model's quant format is compatible with target EP.

    Minimal detection: checks for QLinear ops targeting QDQ-only EPs.
    FIXME: Expand to full EP-to-format compatibility matrix.
    """
    from ..onnx import load_onnx

    model = load_onnx(model_path, load_weights=False, validate=False)
    op_types = {n.op_type for n in model.graph.node}
    has_qlinear = any(op.startswith("QLinear") for op in op_types)
    has_qdq = bool(op_types & QDQ_OP_TYPES)

    if ep == "qnn" and has_qlinear and not has_qdq:  # noqa: SIM103
        return True  # QNN requires QDQ format
    # FIXME: add more EP rules as needed
    return False
