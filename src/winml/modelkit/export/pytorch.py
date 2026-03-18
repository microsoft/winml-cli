"""ONNX Export for PyTorch nn.Module models.

Exports any nn.Module to ONNX using WinMLExportConfig for I/O specification.
Uses HTPExporter internally for hierarchy preservation and metadata.

No HuggingFace dependency required when input_tensors are populated.

Example:
    >>> import torch.nn as nn
    >>> from winml.modelkit.export.config import WinMLExportConfig, InputTensorSpec
    >>> from winml.modelkit.export.pytorch import export_pytorch
    >>>
    >>> model = nn.Linear(10, 5)
    >>> config = WinMLExportConfig(
    ...     input_tensors=[InputTensorSpec(name="input", dtype="float32", shape=(1, 10))],
    ... )
    >>> export_pytorch(model, "model.onnx", config)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import torch.nn as nn

    from .config import WinMLExportConfig

logger = logging.getLogger(__name__)


def export_pytorch(
    model: nn.Module,
    output_path: str | Path,
    export_config: WinMLExportConfig,
    *,
    model_name_or_path: str | None = None,
    model_id: str | None = None,
    task: str | None = None,
    verbose: bool = False,
    enable_reporting: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Export a PyTorch nn.Module to ONNX.

    Uses WinMLExportConfig for I/O specification and HTPExporter for
    hierarchy-preserving ONNX export. Works with any nn.Module — no
    HuggingFace dependency when input_tensors are populated in config.

    Args:
        model: PyTorch model to export.
        output_path: Path for the output .onnx file.
        export_config: Export configuration with input/output tensor specs.
        model_name_or_path: HF model ID for auto-input generation fallback.
        task: Task for auto-input generation fallback.
        verbose: Enable verbose logging.
        enable_reporting: Generate export report file.

    Returns:
        Export statistics dict from HTPExporter.
    """
    from .htp.exporter import HTPExporter

    # Accept both model_name_or_path and model_id (backward compat)
    model_name_or_path = model_name_or_path or model_id

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generic nn.Module won't have .config — add a minimal stub.
    if not hasattr(model, "config"):
        model.config = type("Config", (), {"model_type": "pytorch"})()

    exporter = HTPExporter(
        verbose=verbose,
        enable_reporting=enable_reporting,
        embed_hierarchy_attributes=export_config.enable_hierarchy_tags,
    )
    return exporter.export(
        model=model,
        output_path=str(output_path),
        export_config=export_config,
        model_name_or_path=model_name_or_path,
        task=task,
        **kwargs,
    )
