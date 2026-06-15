# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HTP (Hierarchical Trace-and-Project) Strategy with IO/ABC Architecture.

This strategy uses execution tracing with PyTorch hooks to capture module context
during forward pass, then projects this onto ONNX operations.

Key Features:
- Works with complex models and control flow
- Built-in module tracking for better accuracy
- Conservative tag propagation
- Optimized for HuggingFace transformers
- New IO/ABC-based monitoring architecture

Variations:
- Standard HTP: Hook-based execution tracing
- Built-in HTP: Uses PyTorch's internal module tracking

TODO: Future folder structure refactoring
Currently keeping the folder structure flat for simplicity.
In the future, consider organizing into:
- core/: Core HTP logic (htp_exporter.py, metadata_builder.py)
- writers/: All output writers (console_writer.py, metadata_writer.py, etc.)
"""

# HTP strategy version (defined before imports to avoid circular dependencies)
__version__: str = "1.0.0"  # HTP strategy version
__spec_version__: str = ".".join(__version__.split(".")[:2])  # "1.0"

from .base_writer import ExportStep
from .exporter import HTPExporter
from .monitor import HTPExportMonitor


__all__ = [
    "ExportStep",
    "HTPExportMonitor",
    "HTPExporter",
    "__spec_version__",
    "__version__",
]
