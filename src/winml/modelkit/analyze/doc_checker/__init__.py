# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QNN Operator Constraint Checker and Operator Mapping Framework.

This package provides:
1. Constraint validation tools for checking ONNX models against QNN constraints (ADR-001)
2. Operator mapping framework for ONNX to QNN conversion (ADR-002)

Architecture:
- Base constraint checkers (shared infrastructure)
- Operator mapping wrappers
- Constraint type constants

Note: Constraints are stored as JSON dictionaries with runtime function lookup.
      See docs/CONSTRAINT_SYSTEM.md for implementation details.
"""

from .mapping_checkers import CHECKER_REGISTRY, get_qnn_op_for_onnx_node
from .shape_checker import ShapeConstraintChecker
from .value_checker import ValueConstraintChecker


__all__ = [
    "CHECKER_REGISTRY",
    "ShapeConstraintChecker",
    "ValueConstraintChecker",
    "get_qnn_op_for_onnx_node",
]

__version__ = "1.0.0"
