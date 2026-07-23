# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HTP (Hierarchical Trace-and-Project) strategy with IO/ABC architecture.

HTP preserves PyTorch module context on ONNX nodes. The hierarchy source follows
the selected exporter:

- TorchDynamo (default): reconstruct hierarchy from ONNX node metadata after export.
- TorchScript (``--no-dynamo``): trace module execution with PyTorch forward hooks,
  then project the traced hierarchy onto ONNX nodes.

Both paths emit the same ``winml.hierarchy.*`` metadata contract for downstream
inspection, benchmarking, and optimization.

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
