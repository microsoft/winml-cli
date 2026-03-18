"""Operator-level profiling for ModelKit."""
from __future__ import annotations


def is_qnn_profiling_available() -> bool:
    """Check if QNN EP is available for op-tracing."""
    try:
        import onnxruntime as ort

        return "QNNExecutionProvider" in ort.get_available_providers()
    except (ImportError, AttributeError):
        return False
