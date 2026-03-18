"""Core pattern builders for ONNX graph optimization testing.

This module provides fundamental pattern builders that are used across
multiple test scenarios for ORT graph optimizations.

Pattern Builders:
- identity_relu_builder: Identity -> Relu pattern
- constant_folding_builder: Cast -> Mul(Const*Scale) -> Add pattern
- cse_builder: Where(true, x, x) -> Relu pattern (CSE elimination)
- reshape_builder: Reshape chain with identity Slice pattern
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper


__all__ = [
    "constant_folding_builder",
    "cse_builder",
    "identity_relu_builder",
    "reshape_builder",
]


def identity_relu_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Identity -> Relu pattern."""
    return [
        helper.make_node("Identity", [input_name], [f"{prefix}id_out"], name=f"{prefix}identity"),
        helper.make_node("Relu", [f"{prefix}id_out"], [output_name], name=f"{prefix}relu"),
    ]


def constant_folding_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build ConstantFolding pattern: Input -> Cast -> Add(Const*Scale).

    P1-02: Cast(to=FLOAT) on already float32 input is eliminable.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    initializers.append(
        numpy_helper.from_array(rng.randn(1, 64).astype(np.float32) * 0.1, f"{prefix}const")
    )
    initializers.append(
        numpy_helper.from_array(np.array([2.0], dtype=np.float32), f"{prefix}scale")
    )

    return [
        helper.make_node(
            "Cast", [input_name], [f"{prefix}cast_out"], name=f"{prefix}cast", to=TensorProto.FLOAT
        ),
        helper.make_node(
            "Mul", [f"{prefix}const", f"{prefix}scale"], [f"{prefix}scaled"], name=f"{prefix}mul"
        ),
        helper.make_node(
            "Add", [f"{prefix}cast_out", f"{prefix}scaled"], [output_name], name=f"{prefix}add"
        ),
    ]


def cse_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build CSE pattern: Input -> Where(true, x, x) -> Relu.

    P1-03: Where(condition=true_tensor, x, x) always outputs x, hence eliminable.
    """
    # Create true condition tensor (all True)
    initializers.append(numpy_helper.from_array(np.array([True], dtype=bool), f"{prefix}true_cond"))

    return [
        helper.make_node(
            "Where",
            [f"{prefix}true_cond", input_name, input_name],
            [f"{prefix}where_out"],
            name=f"{prefix}where",
        ),
        helper.make_node("Relu", [f"{prefix}where_out"], [output_name], name=f"{prefix}relu"),
    ]


def reshape_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build Reshape pattern with identity Slice (shape-preserving).

    P1-04: Slice(starts=[0], ends=[64], axes=[1]) on shape [1,64] is identity, hence eliminable.
    P1-07: Chain 2 identity reshapes - tests reshape-elimination and reshape-fusion.
    """
    # P1-07: Chain identity reshapes (1,64) -> (1,64) -> (1,64)
    initializers.append(
        numpy_helper.from_array(np.array([1, 64], dtype=np.int64), f"{prefix}shape1")
    )
    initializers.append(
        numpy_helper.from_array(np.array([1, 64], dtype=np.int64), f"{prefix}shape2")
    )
    # Identity slice parameters
    initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), f"{prefix}starts"))
    initializers.append(numpy_helper.from_array(np.array([64], dtype=np.int64), f"{prefix}ends"))
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), f"{prefix}axes"))

    return [
        # P1-07: First identity reshape
        helper.make_node(
            "Reshape",
            [input_name, f"{prefix}shape1"],
            [f"{prefix}reshape1_out"],
            name=f"{prefix}reshape1",
        ),
        # P1-07: Second identity reshape
        helper.make_node(
            "Reshape",
            [f"{prefix}reshape1_out", f"{prefix}shape2"],
            [f"{prefix}reshape2_out"],
            name=f"{prefix}reshape2",
        ),
        # P1-04: Identity slice
        helper.make_node(
            "Slice",
            [f"{prefix}reshape2_out", f"{prefix}starts", f"{prefix}ends", f"{prefix}axes"],
            [output_name],
            name=f"{prefix}slice",
        ),
    ]
