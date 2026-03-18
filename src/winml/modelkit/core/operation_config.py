"""Centralized Operation Configuration for PyTorch to ONNX Mapping.

This module provides the OperationConfig class which serves as a single source
of truth for PyTorch operation definitions and their corresponding ONNX operation
types. This eliminates duplication between patching operations and ONNX mapping
across different export strategies.

Key Features:
- Universal operation registry with PyTorch to ONNX mappings
- Priority-based operation organization
- Support for both torch and torch.nn.functional operations
- Extensible operation registry
- Centralized configuration for all export strategies
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class OperationConfig:
    """Centralized operation configuration for both patching and ONNX mapping.

    This class provides a single source of truth for PyTorch operation definitions
    and their corresponding ONNX operation types, eliminating duplication between
    _patch_torch_operations() and _project_execution_trace_to_onnx().
    """

    # Single source of truth for operation mappings
    OPERATION_REGISTRY: ClassVar[dict] = {
        # Core mathematical operations
        "matmul": {
            "patch_targets": [("torch", "matmul")],
            "onnx_types": ["MatMul", "Gemm"],
            "priority": 1,
        },
        "add": {
            "patch_targets": [("torch", "add")],
            "onnx_types": ["Add"],
            "priority": 1,
        },
        "sub": {
            "patch_targets": [("torch", "sub")],
            "onnx_types": ["Sub"],
            "priority": 1,
        },
        "mul": {
            "patch_targets": [("torch", "mul")],
            "onnx_types": ["Mul"],
            "priority": 1,
        },
        "div": {
            "patch_targets": [("torch", "div")],
            "onnx_types": ["Div"],
            "priority": 1,
        },
        "pow": {
            "patch_targets": [("torch", "pow")],
            "onnx_types": ["Pow"],
            "priority": 1,
        },
        "sqrt": {
            "patch_targets": [("torch", "sqrt")],
            "onnx_types": ["Sqrt"],
            "priority": 1,
        },
        "erf": {
            "patch_targets": [("torch", "erf")],
            "onnx_types": ["Erf"],
            "priority": 1,
        },
        "tanh": {
            "patch_targets": [("torch", "tanh"), ("F", "tanh")],
            "onnx_types": ["Tanh"],
            "priority": 1,
        },
        "relu": {
            "patch_targets": [("torch", "relu"), ("F", "relu")],
            "onnx_types": ["Relu"],
            "priority": 1,
        },
        "bmm": {
            "patch_targets": [("torch", "bmm")],
            "onnx_types": ["MatMul"],
            "priority": 1,
        },
        "abs": {
            "patch_targets": [("torch", "abs")],
            "onnx_types": ["Abs"],
            "priority": 1,
        },
        "neg": {
            "patch_targets": [("torch", "neg")],
            "onnx_types": ["Neg"],
            "priority": 1,
        },
        "reciprocal": {
            "patch_targets": [("torch", "reciprocal")],
            "onnx_types": ["Reciprocal"],
            "priority": 1,
        },
        "sigmoid": {
            "patch_targets": [("torch", "sigmoid"), ("F", "sigmoid")],
            "onnx_types": ["Sigmoid"],
            "priority": 1,
        },
        "log": {
            "patch_targets": [("torch", "log")],
            "onnx_types": ["Log"],
            "priority": 1,
        },
        "exp": {
            "patch_targets": [("torch", "exp")],
            "onnx_types": ["Exp"],
            "priority": 1,
        },
        "floor": {
            "patch_targets": [("torch", "floor")],
            "onnx_types": ["Floor"],
            "priority": 1,
        },
        "ceil": {
            "patch_targets": [("torch", "ceil")],
            "onnx_types": ["Ceil"],
            "priority": 1,
        },
        # Indexing and gathering operations
        "index_select": {
            "patch_targets": [("torch", "index_select")],
            "onnx_types": ["Gather"],
            "priority": 2,
        },
        "gather": {
            "patch_targets": [("torch", "gather")],
            "onnx_types": ["Gather"],
            "priority": 2,
        },
        "embedding": {
            "patch_targets": [("torch", "embedding"), ("F", "embedding")],
            "onnx_types": ["Gather"],
            "priority": 2,
        },
        "where": {
            "patch_targets": [("torch", "where")],
            "onnx_types": ["Where"],
            "priority": 2,
        },
        "eq": {
            "patch_targets": [("torch", "eq")],
            "onnx_types": ["Equal"],
            "priority": 2,
        },
        "equal": {
            "patch_targets": [("torch", "equal")],
            "onnx_types": ["Equal"],
            "priority": 2,
        },
        # Shape operations
        "reshape": {
            "patch_targets": [("torch", "reshape")],
            "onnx_types": ["Reshape"],
            "priority": 3,
        },
        "transpose": {
            "patch_targets": [("torch", "transpose")],
            "onnx_types": ["Transpose"],
            "priority": 3,
        },
        "unsqueeze": {
            "patch_targets": [("torch", "unsqueeze")],
            "onnx_types": ["Unsqueeze"],
            "priority": 3,
        },
        "squeeze": {
            "patch_targets": [("torch", "squeeze")],
            "onnx_types": ["Squeeze"],
            "priority": 3,
        },
        "cat": {
            "patch_targets": [("torch", "cat")],
            "onnx_types": ["Concat"],
            "priority": 3,
        },
        # Note: expand is a tensor method, not a torch function
        # slice: PyTorch slicing (x[1:5]) converts to ONNX Slice nodes
        # but there's no torch.slice function to patch - handled by ONNX conversion
        "slice": {
            "patch_targets": [],  # No patchable function - tensor[1:5] syntax handled by ONNX
            "onnx_types": ["Slice"],
            "priority": 3,
        },
        "narrow": {
            "patch_targets": [("torch", "narrow")],
            "onnx_types": ["Slice"],
            "priority": 3,
        },
        "select": {
            "patch_targets": [("torch", "select")],
            "onnx_types": ["Gather", "Slice"],
            "priority": 3,
        },
        "take": {
            "patch_targets": [("torch", "take")],
            "onnx_types": ["Gather"],
            "priority": 3,
        },
        # Reduction operations
        "mean": {
            "patch_targets": [("torch", "mean")],
            "onnx_types": ["ReduceMean"],
            "priority": 4,
        },
        "sum": {
            "patch_targets": [("torch", "sum")],
            "onnx_types": ["ReduceSum"],
            "priority": 4,
        },
        "cumsum": {
            "patch_targets": [("torch", "cumsum")],
            "onnx_types": ["CumSum"],
            "priority": 4,
        },
        "cumprod": {
            "patch_targets": [("torch", "cumprod")],
            "onnx_types": ["CumProd"],
            "priority": 4,
        },
        # Note: cast is typically done via .to() method, not a torch function
        # High-level functional operations
        "linear": {
            "patch_targets": [("F", "linear")],
            "onnx_types": ["Gemm", "MatMul"],
            "priority": 6,
        },
        "softmax": {
            "patch_targets": [("F", "softmax")],
            "onnx_types": ["Softmax"],
            "priority": 6,
        },
        "layer_norm": {
            "patch_targets": [("F", "layer_norm")],
            "onnx_types": [
                "LayerNormalization",
                "Add",
                "Mul",
                "Div",
                "ReduceMean",
                "Sub",
                "Sqrt",
                "Pow",
            ],
            "priority": 6,
        },
        "pad": {"patch_targets": [("F", "pad")], "onnx_types": ["Pad"], "priority": 6},
        "dropout": {
            "patch_targets": [("F", "dropout")],
            "onnx_types": ["Dropout"],
            "priority": 6,
        },
        "gelu": {
            "patch_targets": [("F", "gelu")],
            "onnx_types": ["Erf", "Add", "Mul", "Div"],
            "priority": 6,
        },
        # Native operations (highest priority)
        "scaled_dot_product_attention": {
            "patch_targets": [("F", "scaled_dot_product_attention")],
            "onnx_types": [
                "MatMul",
                "Div",
                "Softmax",
                "MatMul",
            ],  # Typical decomposition pattern
            "priority": 10,
        },
        # Additional ONNX-only mappings (no patch targets)
        "size": {"patch_targets": [], "onnx_types": ["Shape"], "priority": 3},
        "shape": {"patch_targets": [], "onnx_types": ["Shape"], "priority": 3},
        "zeros": {
            "patch_targets": [],
            "onnx_types": ["ConstantOfShape"],
            "priority": 5,
        },
        "ones": {"patch_targets": [], "onnx_types": ["ConstantOfShape"], "priority": 5},
        "full": {"patch_targets": [], "onnx_types": ["ConstantOfShape"], "priority": 5},
        "tensor": {"patch_targets": [], "onnx_types": ["Constant"], "priority": 5},
    }

    @classmethod
    def get_operations_to_patch(cls) -> list[tuple]:
        """Get list of (module_name, operation_name) tuples for patching.

        Returns:
            List of tuples suitable for patching PyTorch operations
        """
        import torch
        import torch.nn.functional as F

        module_map = {"torch": torch, "F": F}

        operations = []
        for op_data in cls.OPERATION_REGISTRY.values():
            for module_name, op_name in op_data["patch_targets"]:
                if module_name in module_map:
                    operations.append((module_map[module_name], op_name))

        return operations

    @classmethod
    def get_torch_to_onnx_mapping(cls) -> dict[str, list[str]]:
        """Get mapping from PyTorch operation names to ONNX operation types.

        Returns:
            Dictionary mapping operation names to lists of ONNX types
        """
        return {
            op_name: op_data["onnx_types"] for op_name, op_data in cls.OPERATION_REGISTRY.items()
        }

    @classmethod
    def add_operation(
        cls,
        op_name: str,
        patch_targets: list[tuple[str, str]],
        onnx_types: list[str],
        priority: int = 5,
    ) -> None:
        """Add a new operation to the registry.

        Args:
            op_name: Name of the operation
            patch_targets: List of (module_name, operation_name) for patching
            onnx_types: List of corresponding ONNX operation types
            priority: Priority level (1=highest, 10=lowest)
        """
        cls.OPERATION_REGISTRY[op_name] = {
            "patch_targets": patch_targets,
            "onnx_types": onnx_types,
            "priority": priority,
        }
