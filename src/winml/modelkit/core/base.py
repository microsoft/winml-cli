# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Base Classes and Shared Utilities for Model Export.

This module provides the base hierarchy exporter interface and shared utilities
used across all export strategies.

Components:
- BaseHierarchyExporter: Abstract base class for all export strategies
- Shared utility functions for module filtering and hierarchy building
- Common data structures and type definitions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch


class BaseHierarchyExporter(ABC):
    """Abstract base class for all hierarchy-preserving ONNX exporters.

    Defines the common interface that all export strategies must implement,
    ensuring consistency across FX, HTP, and usage-based strategies.
    """

    def __init__(self) -> None:
        """Initialize base exporter with common state tracking."""
        self._model_root: torch.nn.Module | None = None
        self._export_stats: dict[str, Any] = {}

    @abstractmethod
    def export(
        self,
        model: torch.nn.Module,
        example_inputs: torch.Tensor | tuple | dict,
        output_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Export PyTorch model to ONNX with hierarchy preservation.

        Args:
            model: PyTorch model to export
            example_inputs: Example inputs for tracing/export
            output_path: Path to save ONNX model
            **kwargs: Strategy-specific arguments

        Returns:
            Export metadata with hierarchy information
        """

    @abstractmethod
    def extract_subgraph(self, onnx_path: str, target_module: str) -> dict[str, Any]:
        """Extract subgraph for specific module hierarchy.

        Args:
            onnx_path: Path to ONNX model file
            target_module: Target module hierarchy path

        Returns:
            Subgraph extraction results
        """

    def get_export_stats(self) -> dict[str, Any]:
        """Get statistics from the last export operation."""
        return self._export_stats.copy()


def should_include_in_hierarchy(
    module: torch.nn.Module, exceptions: list[str] | None = None
) -> bool:
    """Determine if a module should be included in the hierarchy structure.

    Filters out torch.nn infrastructure modules while preserving semantically important ones.
    NOTE: This function is primarily used for ONNX tag filtering. For complete hierarchy
    reports, consider bypassing this filter entirely.

    Args:
        module: PyTorch module to evaluate
        exceptions: List of torch.nn class names to include despite being infrastructure.
                   If None, uses default exceptions: [] (MUST-002 compliance)

    Returns:
        True if module should be included in hierarchy, False otherwise

    Examples:
        >>> # HuggingFace modules - always included
        >>> should_include_in_hierarchy(bert_embeddings)  # BertEmbeddings -> True
        >>> should_include_in_hierarchy(bert_attention)   # BertAttention -> True

        >>> # torch.nn modules - excluded by default (MUST-002)
        >>> should_include_in_hierarchy(layer_norm)       # LayerNorm -> False

        >>> # torch.nn infrastructure - excluded
        >>> should_include_in_hierarchy(embedding)        # Embedding -> False
        >>> should_include_in_hierarchy(dropout)          # Dropout -> False
        >>> should_include_in_hierarchy(relu)            # ReLU -> False
        >>> should_include_in_hierarchy(linear)          # Linear -> False

        >>> # Custom exceptions
        >>> should_include_in_hierarchy(
        ...     embedding, exceptions=["LayerNorm", "Embedding"]
        ... )  # Embedding -> True
    """
    # Input validation
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(module)}")

    # Default exception for semantically important torch.nn modules
    if exceptions is None:
        exceptions = []  # MUST-002: No torch.nn classes should appear in hierarchy tags

    module_class_name = module.__class__.__name__

    # Check if it's torch.nn infrastructure
    is_torch_infrastructure = module.__class__.__module__.startswith(
        "torch.nn"
    ) or module.__class__.__module__.startswith("torch._C")

    if is_torch_infrastructure:
        # Include only if it's in the exceptions list
        return module_class_name in exceptions
    # Include all non-torch.nn modules (HuggingFace, user-defined, etc.)
    return True


def build_hierarchy_path(
    model_root: torch.nn.Module, module_path: str, all_modules: dict[str, torch.nn.Module]
) -> str:
    """Build hierarchical class path for a module instance.

    Args:
        model_root: Root model (e.g., BertModel)
        module_path: Module instance path (e.g., "encoder.layer.0.attention.self")
        all_modules: Dict mapping module paths to modules from named_modules()

    Returns:
        Hierarchical path (e.g.,
        "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention")

    Examples:
        >>> build_hierarchy_path(bert_model, "embeddings", all_modules)
        "/BertModel/BertEmbeddings"

        >>> build_hierarchy_path(bert_model, "encoder.layer.0.attention.self", all_modules)
        "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention"

        >>> build_hierarchy_path(bert_model, "embeddings.LayerNorm", all_modules)
        "/BertModel/BertEmbeddings/LayerNorm"
    """
    # Start with root model class name
    root_class = model_root.__class__.__name__
    hierarchy_parts = [root_class]

    if not module_path:  # Root module
        return "/" + "/".join(hierarchy_parts)

    # Split the module path into parts
    path_parts = module_path.split(".")
    current_path = ""

    for i, part in enumerate(path_parts):
        if current_path:
            current_path += "." + part
        else:
            current_path = part

        # Get the module at this path
        if current_path in all_modules:
            module = all_modules[current_path]
            class_name = module.__class__.__name__

            # MUST-002: Filter out torch.nn modules from hierarchy paths
            if should_include_in_hierarchy(module):
                # Handle numeric indices in the path (e.g., "layer.0" -> "BertLayer.0")
                if i > 0 and path_parts[i - 1] in ["layer"] and part.isdigit():
                    # Previous part was "layer" and current part is a number
                    # Combine with the class name: "BertLayer.0"
                    hierarchy_parts[-1] = f"{class_name}.{part}"
                else:
                    hierarchy_parts.append(class_name)

    return "/" + "/".join(hierarchy_parts)


def validate_output_path(output_path: str) -> str:
    """Validate and normalize output path for ONNX export.

    Args:
        output_path: User-provided output path

    Returns:
        Validated and normalized output path

    Raises:
        ValueError: If path is invalid
    """
    path = Path(output_path)

    # Ensure .onnx extension
    if not path.suffix:
        path = path.with_suffix(".onnx")
    elif path.suffix.lower() != ".onnx":
        raise ValueError(f"Output path must have .onnx extension, got: {path.suffix}")

    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    return str(path)


def get_model_signature(model: torch.nn.Module) -> str:
    """Generate a unique signature for a model for caching and identification.

    Args:
        model: PyTorch model

    Returns:
        Unique model signature string
    """
    class_name = model.__class__.__name__
    module_count = len(list(model.named_modules()))
    param_count = sum(p.numel() for p in model.parameters())

    return f"{class_name}_{module_count}_{param_count}"


def extract_forward_signature(module: torch.nn.Module) -> dict[str, Any]:
    """Extract forward method signature for module metadata.

    Args:
        module: PyTorch module

    Returns:
        Dictionary with forward method signature information
    """
    import inspect

    try:
        sig = inspect.signature(module.forward)
        return {
            "forward_args": list(sig.parameters.keys()),
            "forward_defaults": {
                name: param.default if param.default != param.empty else None
                for name, param in sig.parameters.items()
            },
        }
    except Exception:
        return {"forward_args": [], "forward_defaults": {}}
