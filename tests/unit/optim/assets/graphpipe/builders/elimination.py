# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Elimination pattern builders for ORT Graph Optimization tests.

This module contains builders for patterns that test ORT's elimination optimizations:
- EliminateSlice: Identity slice removal (starts=0, ends=INT64_MAX)
- UnsqueezeElimination: Unsqueeze on constant initializer
- ReshapeElimination: Contiguous reshape fusion
- ConcatSliceElimination: Concat followed by exact slice extraction

Key Requirements (from ORT source analysis):
1. EliminateSlice: ends must be >= INT64_MAX for identity detection
2. UnsqueezeElimination: Input must be constant initializer (not runtime tensor)
3. ReshapeElimination: Shape tensor must be constant initializer
4. ConcatSliceElimination: axis=0, inputs must be initializers, slice exact boundaries
"""

from __future__ import annotations

import numpy as np
from onnx import helper, numpy_helper


# INT64_MAX value for identity slice detection
INT64_MAX = 9223372036854775807


def slice_elimination_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build identity Slice pattern for EliminateSlice optimization.

    ORT eliminates Slice when starts=0 and ends >= INT64_MAX.
    From slice_elimination.cc line 119-122:
        for (size_t i = 0; i < starts.size(); ++i) {
            if (starts[i] != 0 || ends[i] < INT64_MAX) {
                return false;
            }
        }

    CRITICAL: EliminateSlice runs at Level 1 (ORT_ENABLE_BASIC).
    The CanRemoveNode check requires the Slice to have downstream consumers
    that can have their inputs rewired - direct output connections prevent this.

    This builder creates:
    Input -> Slice(identity) -> Abs -> Relu -> Output

    The Abs node acts as a downstream consumer, allowing ORT to remove the Slice
    and rewire Abs to consume the original input directly.

    NOTE: This optimization runs at Level 1, not Level 2. The current test
    framework uses Level 2 by default. To test EliminateSlice properly,
    need optimization_level=1 support in GraphPipe.
    """
    # Slice parameters for all dimensions: starts=0, ends=INT64_MAX
    # Must cover both dimensions of the [1, 64] input
    initializers.append(
        numpy_helper.from_array(np.array([0, 0], dtype=np.int64), f"{prefix}starts")
    )
    initializers.append(
        numpy_helper.from_array(np.array([INT64_MAX, INT64_MAX], dtype=np.int64), f"{prefix}ends")
    )
    # Axes in any order, as long as all dims are covered
    initializers.append(numpy_helper.from_array(np.array([0, 1], dtype=np.int64), f"{prefix}axes"))

    return [
        # Identity slice (starts=0, ends=INT64_MAX for all dims) - eliminable by ORT
        helper.make_node(
            "Slice",
            [input_name, f"{prefix}starts", f"{prefix}ends", f"{prefix}axes"],
            [f"{prefix}slice_out"],
            name=f"{prefix}slice",
        ),
        # Downstream consumer - allows CanRemoveNode to pass
        helper.make_node("Abs", [f"{prefix}slice_out"], [f"{prefix}abs_out"], name=f"{prefix}abs"),
        helper.make_node("Relu", [f"{prefix}abs_out"], [output_name], name=f"{prefix}relu"),
    ]


def unsqueeze_elimination_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Unsqueeze pattern for UnsqueezeElimination optimization.

    ORT eliminates Unsqueeze when input is a constant initializer by folding
    the Unsqueeze result into a new initializer with the unsqueezed shape.

    From unsqueeze_elimination.cc line 88-90:
        bool UnsqueezeElimination::SatisfyCondition(...) const {
            return graph_utils::IsConstantInitializer(graph, node.InputDefs()[0]->Name());
        }

    This builder creates:
    Constant[64] -> Unsqueeze(axes=[0]) -> [1,64] -> Add(input) -> Relu -> Output

    The Unsqueeze operates on a constant, making it eliminable.
    ORT folds the unsqueezed constant into initializers, removing the Unsqueeze node.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Create a constant tensor with shape [64] that Unsqueeze will expand to [1, 64]
    # This constant is what makes the Unsqueeze eliminable
    const_data = rng.randn(64).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(const_data, f"{prefix}const_tensor"))

    # Axes for Unsqueeze (opset 13+) - must be constant initializer
    # Adding axis 0 transforms [64] -> [1, 64]
    initializers.append(
        numpy_helper.from_array(np.array([0], dtype=np.int64), f"{prefix}unsqueeze_axes")
    )

    return [
        # Unsqueeze on constant [64] -> [1, 64] - eliminable by ORT
        helper.make_node(
            "Unsqueeze",
            [f"{prefix}const_tensor", f"{prefix}unsqueeze_axes"],
            [f"{prefix}unsqueeze_out"],
            name=f"{prefix}unsqueeze",
        ),
        # Add runtime input [1, 64] to unsqueezed constant [1, 64]
        helper.make_node(
            "Add",
            [input_name, f"{prefix}unsqueeze_out"],
            [f"{prefix}add_out"],
            name=f"{prefix}add",
        ),
        helper.make_node("Relu", [f"{prefix}add_out"], [output_name], name=f"{prefix}relu"),
    ]


def reshape_elimination_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Reshape pattern for ReshapeElimination optimization.

    ORT eliminates Reshape when input is a constant initializer by folding
    the Reshape result into a new initializer with the reshaped shape.

    This builder creates:
    Constant[64] -> Reshape([1, 64]) -> Add(input) -> Relu -> Output

    The Reshape operates on a constant, making it eliminable.
    ORT folds the reshaped constant into initializers, removing the Reshape node.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Create a constant tensor with shape [64] that Reshape will transform to [1, 64]
    const_data = rng.randn(64).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(const_data, f"{prefix}const_tensor"))

    # Shape initializer to reshape [64] -> [1, 64]
    initializers.append(
        numpy_helper.from_array(np.array([1, 64], dtype=np.int64), f"{prefix}shape")
    )

    return [
        # Reshape on constant [64] -> [1, 64] - eliminable by ORT
        helper.make_node(
            "Reshape",
            [f"{prefix}const_tensor", f"{prefix}shape"],
            [f"{prefix}reshape_out"],
            name=f"{prefix}reshape",
        ),
        # Add runtime input [1, 64] to reshaped constant [1, 64]
        helper.make_node(
            "Add",
            [input_name, f"{prefix}reshape_out"],
            [f"{prefix}add_out"],
            name=f"{prefix}add",
        ),
        helper.make_node("Relu", [f"{prefix}add_out"], [output_name], name=f"{prefix}relu"),
    ]


def expand_elimination_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Expand elimination pattern for ExpandElimination optimization.

    ORT eliminates Expand when the operation is identity (output shape equals input shape)
    or when the Expand can be folded into a constant.

    From expand_elimination.cc:
    - ExpandElimination removes Expand nodes where output_shape == input_shape
    - Also handles Expand on constant initializers by folding

    This builder creates:
    Input [1,64] -> Expand([1,64]) -> Relu -> Output [1,64]

    The Expand to same shape is identity and should be eliminated by ORT.
    """
    # Expand to same shape [1, 64] - this is an identity operation
    initializers.append(
        numpy_helper.from_array(np.array([1, 64], dtype=np.int64), f"{prefix}expand_shape")
    )

    return [
        # Identity Expand (same shape) - eliminable by ORT
        helper.make_node(
            "Expand",
            [input_name, f"{prefix}expand_shape"],
            [f"{prefix}expand_out"],
            name=f"{prefix}expand",
        ),
        helper.make_node("Relu", [f"{prefix}expand_out"], [output_name], name=f"{prefix}relu"),
    ]


def concat_slice_elimination_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Concat+Slice elimination pattern for ConcatSliceElimination optimization.

    ORT eliminates Concat+Slice when:
    1. Concat axis = 0
    2. All concat inputs are constant initializers
    3. Each slice exactly extracts one original concat input
    4. Number of slices equals number of concat inputs

    From concat_slice_elimination.cc line 173-188:
        if (concat_outputs.size() != concat_inputs.size()) return false;
        ...
        for (size_t i = 0; i < num_inputs; i++) {
            is_valid = is_valid && graph_utils::IsInitializer(graph, ...);
        }
        if (!is_valid) return false;
        ...
        if (axis_attr->i() != 0) return false;

    This builder creates:
    Three constant tensors -> Concat(axis=0) -> Slice[0:len0] -> Add(input) -> Relu -> Output

    Note: This creates the pattern but actual elimination requires all 3 slices consuming
    the concat output. For testing purposes, we show the pattern structure.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Create three constant tensors as concat inputs (must be initializers)
    # Using 1D tensors for axis=0 concat
    const0 = rng.randn(20).astype(np.float32) * 0.01
    const1 = rng.randn(22).astype(np.float32) * 0.01
    const2 = rng.randn(22).astype(np.float32) * 0.01

    initializers.append(numpy_helper.from_array(const0, f"{prefix}const0"))
    initializers.append(numpy_helper.from_array(const1, f"{prefix}const1"))
    initializers.append(numpy_helper.from_array(const2, f"{prefix}const2"))

    # Slice parameters to extract first segment [0:20]
    initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), f"{prefix}starts"))
    initializers.append(numpy_helper.from_array(np.array([20], dtype=np.int64), f"{prefix}ends"))
    initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), f"{prefix}axes"))
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), f"{prefix}steps"))

    # Pad to restore shape for template compatibility: [20] -> [64]
    initializers.append(numpy_helper.from_array(np.array([0, 44], dtype=np.int64), f"{prefix}pads"))
    initializers.append(
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), f"{prefix}pad_value")
    )

    # Reshape to match expected output shape [1, 64]
    initializers.append(
        numpy_helper.from_array(np.array([1, 64], dtype=np.int64), f"{prefix}out_shape")
    )

    return [
        # Concat three constant tensors along axis=0
        helper.make_node(
            "Concat",
            [f"{prefix}const0", f"{prefix}const1", f"{prefix}const2"],
            [f"{prefix}concat_out"],
            name=f"{prefix}concat",
            axis=0,
        ),
        # Slice to extract first segment [0:20] - matches const0 exactly
        helper.make_node(
            "Slice",
            [
                f"{prefix}concat_out",
                f"{prefix}starts",
                f"{prefix}ends",
                f"{prefix}axes",
                f"{prefix}steps",
            ],
            [f"{prefix}slice_out"],
            name=f"{prefix}slice",
        ),
        # Pad to 64 elements
        helper.make_node(
            "Pad",
            [f"{prefix}slice_out", f"{prefix}pads", f"{prefix}pad_value"],
            [f"{prefix}padded"],
            name=f"{prefix}pad",
            mode="constant",
        ),
        # Reshape to [1, 64] for template compatibility
        helper.make_node(
            "Reshape",
            [f"{prefix}padded", f"{prefix}out_shape"],
            [f"{prefix}reshaped"],
            name=f"{prefix}reshape",
        ),
        # Add to input and activate
        helper.make_node(
            "Add",
            [input_name, f"{prefix}reshaped"],
            [f"{prefix}add_out"],
            name=f"{prefix}add",
        ),
        helper.make_node("Relu", [f"{prefix}add_out"], [output_name], name=f"{prefix}relu"),
    ]
