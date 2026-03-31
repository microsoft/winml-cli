# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Miscellaneous pattern builders for ORT Graph Optimization tests.

This module contains builder functions for miscellaneous ONNX patterns
that test specific ONNX Runtime graph optimizations.

Pattern Status:
- GatherSliceToSplitFusion: Requires multiple Gather/Slice from same input with scalar indices
- GatherToSliceFusion: Range+Gather pattern (ORT requires Range feeding Gather indices)
- NotWhereFusion: Condition must be dynamic (from graph input), not constant
- DoubleQDQPairsRemover: Zero-point must be uint8, single consumer structure
- PadFusion: Pad+Conv fusion
- SoftmaxFusion: Tests bias-softmax-fusion capability
- TransposeOptimizer: Two inverse transposes should cancel out
- ReduceSoftmax: ReduceSum+Softmax patterns
- NoopElimination: Add(x,0), Mul(x,1), Sub(x,0) elimination
- GatherSliceToSplitFusion (gather_split): Multiple Gather -> Split + Squeeze
- ConcatSliceElimination: Split -> Concat -> Slice elimination
"""

import numpy as np
from onnx import helper, numpy_helper


def gather_slice_to_split_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Gather pattern for GatherSliceToSplitFusion.

    GatherSliceToSplitFusion requires:
    1. Multiple Gather nodes consuming the same input tensor
    2. Scalar indices (rank 0 tensor) - ORT checks indices_n_dims == 0 for squeeze
    3. Indices covering all elements on the axis without overlap
    4. Only opset 13+ supported

    Pattern: Input -> Gather(idx=0) + Gather(idx=1) -> Concat -> Output

    The fusion converts Gather(scalar) operations into Split + Squeeze.
    """
    # Scalar indices (rank 0) for Gather - this is what ORT requires
    # Using numpy scalar to ensure rank 0 tensor
    initializers.append(numpy_helper.from_array(np.int64(0), f"{prefix}idx0"))
    initializers.append(numpy_helper.from_array(np.int64(1), f"{prefix}idx1"))

    # For a [1, 2] shaped input, gather along axis 1 with indices 0 and 1
    # This covers all elements on axis 1

    return [
        # First gather: get element at index 0 on axis 1 -> [1]
        helper.make_node(
            "Gather",
            [input_name, f"{prefix}idx0"],
            [f"{prefix}gather0_out"],
            name=f"{prefix}gather0",
            axis=1,
        ),
        # Second gather: get element at index 1 on axis 1 -> [1]
        helper.make_node(
            "Gather",
            [input_name, f"{prefix}idx1"],
            [f"{prefix}gather1_out"],
            name=f"{prefix}gather1",
            axis=1,
        ),
        # Unsqueeze to restore the axis dimension for concat
        helper.make_node(
            "Unsqueeze",
            [f"{prefix}gather0_out", f"{prefix}unsqueeze_axes"],
            [f"{prefix}unsqueeze0_out"],
            name=f"{prefix}unsqueeze0",
        ),
        helper.make_node(
            "Unsqueeze",
            [f"{prefix}gather1_out", f"{prefix}unsqueeze_axes"],
            [f"{prefix}unsqueeze1_out"],
            name=f"{prefix}unsqueeze1",
        ),
        # Concat back together -> [1, 2]
        helper.make_node(
            "Concat",
            [f"{prefix}unsqueeze0_out", f"{prefix}unsqueeze1_out"],
            [output_name],
            name=f"{prefix}concat",
            axis=1,
        ),
    ]


def gather_slice_to_split_builder_init(prefix: str) -> tuple[tuple[int, ...], list]:
    """Return input shape and additional initializers for gather_slice_to_split pattern.

    This pattern requires a specific input shape [1, 2] where axis 1 has dim 2
    to match the two Gather operations with indices 0 and 1.
    """
    initializers = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), f"{prefix}unsqueeze_axes"),
    ]
    return (1, 2), initializers


def gather_to_slice_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Range+Gather pattern for GatherToSliceFusion.

    GatherToSliceFusion (from ORT's gather_fusion.cc) requires:
    1. A Range node producing contiguous indices
    2. Range output connected to Gather's indices input (position [1])
    3. Range must have exactly one output edge (to the Gather)
    4. Both nodes must use compatible execution providers

    The fusion transforms Range->Gather into Slice by:
    - Using Gather's data input as Slice data
    - Using Range's start/stop values as Slice starts/ends
    - Using Range's step value as Slice step
    - Using Gather's axis attribute as Slice axis

    Pattern: Range(start, limit, delta) -> Gather(data, Range_out) -> Output
    After fusion: data -> Slice(starts, ends, axes, steps) -> Output

    IMPORTANT: This pattern uses constant Range parameters. When testing in
    isolation, ConstantFolding must be disabled (via extra_disabled in the
    test case) to prevent the Range node from being folded away before
    GatherToSliceFusion can see the pattern.

    Note: GatherToSliceFusion is a 1:1 transform (Range+Gather -> Slice+Unsqueezes),
    so node count may not decrease significantly (and may even increase due to
    the added Unsqueeze nodes the fusion creates).
    """
    # Range parameters as scalar initializers (required by Range op)
    # Range(start=0, limit=32, delta=1) produces [0, 1, 2, ..., 31]
    initializers.append(numpy_helper.from_array(np.int64(0), f"{prefix}range_start"))
    initializers.append(numpy_helper.from_array(np.int64(32), f"{prefix}range_limit"))
    initializers.append(numpy_helper.from_array(np.int64(1), f"{prefix}range_delta"))

    return [
        # Range produces contiguous indices [0, 1, 2, ..., 31]
        helper.make_node(
            "Range",
            [f"{prefix}range_start", f"{prefix}range_limit", f"{prefix}range_delta"],
            [f"{prefix}range_out"],
            name=f"{prefix}range",
        ),
        # Gather uses Range output as indices (input[1])
        helper.make_node(
            "Gather",
            [input_name, f"{prefix}range_out"],
            [output_name],
            name=f"{prefix}gather",
            axis=1,
        ),
    ]


def not_where_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build Not+Where pattern for NotWhereFusion.

    NotWhereFusion transforms:
        Not(cond) -> Where(Not_out, A, B)
    Into:
        Where(cond, B, A) (swaps A and B, removes Not)

    CRITICAL REQUIREMENTS for NotWhereFusion to work:
    1. Not node must have ONLY Where nodes as consumers
    2. Both Not and Where must have same execution provider
    3. Not node must be removable (not a graph output, single output usage)

    This builder creates an ISOLATED pattern where each Not has exactly ONE
    Where consumer. This ensures the fusion works regardless of CSE or other
    optimizations that might merge subexpressions.

    To achieve isolation, we create UNIQUE conditions per-pattern by using
    different thresholds, which prevents CSE from merging the Greater nodes.

    Pattern:
        Input -> Greater(unique_threshold) -> Not -> Where(A, B) -> Add(input) -> Output

    The unique threshold per prefix ensures this Greater won't be merged with
    other patterns' Greater nodes via CSE.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Use a UNIQUE threshold per pattern to prevent CSE from merging Greater nodes.
    # The seed is derived from prefix, so each pattern instance gets a different value.
    unique_threshold = rng.uniform(-1.0, 1.0)
    initializers.append(
        numpy_helper.from_array(np.array(unique_threshold, dtype=np.float32), f"{prefix}threshold")
    )
    # Alternative values for Where - also unique per pattern to prevent any CSE
    initializers.append(
        numpy_helper.from_array(rng.randn(64).astype(np.float32) * 0.1, f"{prefix}value_a")
    )
    initializers.append(
        numpy_helper.from_array(rng.randn(64).astype(np.float32) * 0.1, f"{prefix}value_b")
    )

    return [
        # Create dynamic condition from input (Greater returns bool)
        # Using unique threshold prevents CSE from merging this with other Greaters
        helper.make_node(
            "Greater",
            [input_name, f"{prefix}threshold"],
            [f"{prefix}condition"],
            name=f"{prefix}greater",
        ),
        # Not(condition) -> inverted_condition
        helper.make_node(
            "Not",
            [f"{prefix}condition"],
            [f"{prefix}not_out"],
            name=f"{prefix}not",
        ),
        # Where(inverted_condition, A, B) - should fuse to Where(condition, B, A)
        helper.make_node(
            "Where",
            [f"{prefix}not_out", f"{prefix}value_a", f"{prefix}value_b"],
            [f"{prefix}where_out"],
            name=f"{prefix}where",
        ),
        # Add input to make shape-preserving
        helper.make_node(
            "Add",
            [input_name, f"{prefix}where_out"],
            [output_name],
            name=f"{prefix}add",
        ),
    ]


def qdq_pairs_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build QDQ pairs pattern for DoubleQDQPairsRemover.

    CRITICAL FIXES for DoubleQDQPairsRemover:
    1. Zero-point MUST be uint8 (not int8) - ORT defaults to uint8 when zp is absent
    2. Each Q/DQ node should have single consumer for the optimizer to work
    3. Pattern is Q1 -> DQ1 -> Q2 -> DQ2, optimizer removes the middle DQ1-Q2 pair

    The optimizer:
    - Looks for Q -> DQ -> Q -> DQ pattern
    - Removes the middle DQ -> Q pair
    - Recomputes scale/zp for the outer Q -> DQ pair

    Pattern: Input -> Q1 -> DQ1 -> Q2 -> DQ2 -> Output
    After fusion: Input -> Q1' -> DQ2' -> Output (with adjusted scale/zp)
    """
    # Quantization parameters - MUST use uint8 for zero_point
    initializers.append(numpy_helper.from_array(np.array(0.1, dtype=np.float32), f"{prefix}scale1"))
    initializers.append(numpy_helper.from_array(np.array(128, dtype=np.uint8), f"{prefix}zp1"))
    initializers.append(
        numpy_helper.from_array(np.array(0.05, dtype=np.float32), f"{prefix}scale2")
    )
    initializers.append(numpy_helper.from_array(np.array(100, dtype=np.uint8), f"{prefix}zp2"))

    return [
        # First QDQ pair (Q1 -> DQ1)
        helper.make_node(
            "QuantizeLinear",
            [input_name, f"{prefix}scale1", f"{prefix}zp1"],
            [f"{prefix}q1_out"],
            name=f"{prefix}q1",
        ),
        helper.make_node(
            "DequantizeLinear",
            [f"{prefix}q1_out", f"{prefix}scale1", f"{prefix}zp1"],
            [f"{prefix}dq1_out"],
            name=f"{prefix}dq1",
        ),
        # Second QDQ pair (Q2 -> DQ2) - the middle DQ1-Q2 should be removed
        helper.make_node(
            "QuantizeLinear",
            [f"{prefix}dq1_out", f"{prefix}scale2", f"{prefix}zp2"],
            [f"{prefix}q2_out"],
            name=f"{prefix}q2",
        ),
        helper.make_node(
            "DequantizeLinear",
            [f"{prefix}q2_out", f"{prefix}scale2", f"{prefix}zp2"],
            [output_name],
            name=f"{prefix}dq2",
        ),
    ]


def pad_fusion_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build Pad+Conv pattern for Pad_Fusion.

    Pattern: Input -> Reshape -> Pad -> Conv -> Reshape -> Output
    ORT fuses Pad into Conv by adjusting Conv's pads attribute.

    Requirements for PadFusion:
    - Pad mode must be "constant" with constant_value = 0
    - Pad must be followed by Conv
    - Conv must have compatible kernel/pad configuration
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Reshape input from [1, 64] to [1, 16, 2, 2] for Conv2D
    initializers.append(
        numpy_helper.from_array(np.array([1, 16, 2, 2], dtype=np.int64), f"{prefix}reshape_shape")
    )
    # Conv weights: 16 input channels, 16 output channels, 3x3 kernel
    initializers.append(
        numpy_helper.from_array(rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1, f"{prefix}conv_w")
    )
    # Pad values: [x1_begin, x2_begin, x3_begin, x4_begin, x1_end, x2_end, x3_end, x4_end]
    # Pad spatial dimensions (H, W) by 1 on each side
    initializers.append(
        numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64), f"{prefix}pads")
    )
    initializers.append(
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), f"{prefix}pad_value")
    )
    # Reshape back to [1, 64]
    initializers.append(
        numpy_helper.from_array(np.array([1, 64], dtype=np.int64), f"{prefix}out_shape")
    )

    return [
        # Reshape to 4D for Conv
        helper.make_node(
            "Reshape",
            [input_name, f"{prefix}reshape_shape"],
            [f"{prefix}reshaped"],
            name=f"{prefix}reshape1",
        ),
        # Pad with constant 0 (required for fusion)
        helper.make_node(
            "Pad",
            [f"{prefix}reshaped", f"{prefix}pads", f"{prefix}pad_value"],
            [f"{prefix}padded"],
            name=f"{prefix}pad",
            mode="constant",
        ),
        # Conv that will absorb the padding
        helper.make_node(
            "Conv",
            [f"{prefix}padded", f"{prefix}conv_w"],
            [f"{prefix}conv_out"],
            name=f"{prefix}conv",
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0],  # No explicit padding - will be absorbed from Pad
        ),
        # Reshape back to [1, 64]
        helper.make_node(
            "Reshape",
            [f"{prefix}conv_out", f"{prefix}out_shape"],
            [output_name],
            name=f"{prefix}reshape2",
        ),
    ]


def softmax_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build Softmax pattern: Input -> Softmax.

    P2-01: Tests bias-softmax-fusion capability (ORT name: BiasSoftmaxFusion).
    """
    return [
        helper.make_node("Softmax", [input_name], [output_name], name=f"{prefix}softmax", axis=-1)
    ]


def transpose_chain_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Transpose chain pattern: Input -> Transpose(perm=[1,0]) -> Transpose(perm=[1,0]).

    P2-04: Tests transpose-optimizer capability (ORT name: TransposeOptimizer).
    Two inverse transposes should cancel out.
    """
    return [
        helper.make_node(
            "Transpose",
            [input_name],
            [f"{prefix}transpose1_out"],
            name=f"{prefix}transpose1",
            perm=[1, 0],
        ),
        helper.make_node(
            "Transpose",
            [f"{prefix}transpose1_out"],
            [output_name],
            name=f"{prefix}transpose2",
            perm=[1, 0],
        ),
    ]


def reduce_softmax_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build ReduceSum+Softmax pattern: Input -> ReduceSum -> Softmax.

    P2-06: Tests additional softmax patterns and fusion opportunities.
    """
    # ReduceSum axes as input (opset 13+)
    initializers.append(
        numpy_helper.from_array(np.array([-1], dtype=np.int64), f"{prefix}reduce_axes")
    )

    return [
        helper.make_node(
            "ReduceSum",
            [input_name, f"{prefix}reduce_axes"],
            [f"{prefix}reduce_out"],
            name=f"{prefix}reducesum",
            keepdims=1,
        ),
        helper.make_node(
            "Softmax", [f"{prefix}reduce_out"], [output_name], name=f"{prefix}softmax", axis=-1
        ),
    ]


def noop_elimination_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build no-op elimination pattern: Add(x, 0), Mul(x, 1), Sub(x, 0).

    P4-04: Tests noop-elimination capability (ORT name: NoopElimination).
    Identity operations that can be eliminated:
    - Add(x, 0) = x
    - Mul(x, 1) = x
    - Sub(x, 0) = x
    """
    # Constants for no-op operations
    initializers.append(numpy_helper.from_array(np.array([0.0], dtype=np.float32), f"{prefix}zero"))
    initializers.append(numpy_helper.from_array(np.array([1.0], dtype=np.float32), f"{prefix}one"))

    return [
        # Add(x, 0) - no-op, should be eliminated
        helper.make_node(
            "Add",
            [input_name, f"{prefix}zero"],
            [f"{prefix}add_out"],
            name=f"{prefix}add_zero",
        ),
        # Mul(x, 1) - no-op, should be eliminated
        helper.make_node(
            "Mul",
            [f"{prefix}add_out", f"{prefix}one"],
            [f"{prefix}mul_out"],
            name=f"{prefix}mul_one",
        ),
        # Sub(x, 0) - no-op, should be eliminated
        helper.make_node(
            "Sub",
            [f"{prefix}mul_out", f"{prefix}zero"],
            [output_name],
            name=f"{prefix}sub_zero",
        ),
    ]


def gather_split_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build multiple Gather operations pattern for GatherSliceToSplitFusion.

    Pattern: Input -> Gather(idx=0) + Gather(idx=1) -> Concat -> output.
    Tests gather-slice-to-split-fusion (ORT name: GatherSliceToSplitFusion).

    GatherSliceToSplitFusion requires:
    1. Multiple Gather nodes consuming the same input tensor
    2. Scalar indices (rank 0 tensor) - ORT checks indices_n_dims == 0 for squeeze
    3. Indices covering all elements on the axis without overlap
    4. Only opset 13+ supported

    The fusion converts Gather(scalar) operations into Split + Squeeze.

    IMPORTANT: This pattern must be shape-preserving for the template.
    Input shape: [1, 2], Output shape: [1, 2]

    Each Gather with scalar index on axis 1 produces [1] (removes axis).
    We use Unsqueeze to restore dimension and Concat to rebuild [1, 2].

    NOTE: GatherSliceToSplitFusion may not reduce nodes on CPU since it creates
    Split + Squeeze to replace Gather nodes. The test expectation should be
    min_node_reduction=0 if the fusion doesn't reduce total node count.
    """
    # Scalar indices (rank 0) for Gather - this is what ORT requires
    # Using numpy scalar to ensure rank 0 tensor
    initializers.append(numpy_helper.from_array(np.int64(0), f"{prefix}idx0"))
    initializers.append(numpy_helper.from_array(np.int64(1), f"{prefix}idx1"))
    # Axes for unsqueeze to restore dimension
    initializers.append(
        numpy_helper.from_array(np.array([1], dtype=np.int64), f"{prefix}unsqueeze_axes")
    )

    # For a [1, 2] shaped input, gather along axis 1 with indices 0 and 1
    # This covers all elements on axis 1
    # Each Gather(scalar) produces output shape [1] (removes the gathered axis)

    return [
        # First gather: get element at index 0 on axis 1 -> shape [1]
        helper.make_node(
            "Gather",
            [input_name, f"{prefix}idx0"],
            [f"{prefix}gather0_out"],
            name=f"{prefix}gather0",
            axis=1,
        ),
        # Second gather: get element at index 1 on axis 1 -> shape [1]
        helper.make_node(
            "Gather",
            [input_name, f"{prefix}idx1"],
            [f"{prefix}gather1_out"],
            name=f"{prefix}gather1",
            axis=1,
        ),
        # Unsqueeze both to restore [1, 1] shape for concat
        helper.make_node(
            "Unsqueeze",
            [f"{prefix}gather0_out", f"{prefix}unsqueeze_axes"],
            [f"{prefix}unsqueeze0_out"],
            name=f"{prefix}unsqueeze0",
        ),
        helper.make_node(
            "Unsqueeze",
            [f"{prefix}gather1_out", f"{prefix}unsqueeze_axes"],
            [f"{prefix}unsqueeze1_out"],
            name=f"{prefix}unsqueeze1",
        ),
        # Concat to restore [1, 2] shape
        helper.make_node(
            "Concat",
            [f"{prefix}unsqueeze0_out", f"{prefix}unsqueeze1_out"],
            [output_name],
            name=f"{prefix}concat",
            axis=1,
        ),
    ]


def concat_slice_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Concat+Slice elimination pattern: Split -> Concat -> Slice.

    Tests concat-slice-elimination capability (ORT name: ConcatSliceElimination).
    When tensors are concatenated and then sliced to extract the original tensors,
    the concat-slice pair can be eliminated.
    Example: Concat(A, B, C) -> Slice(A), Slice(B), Slice(C) can eliminate redundant concat.
    """
    # Split input [1, 64] into three parts: [1, 21], [1, 21], [1, 22]
    initializers.append(
        numpy_helper.from_array(np.array([21, 21, 22], dtype=np.int64), f"{prefix}split_sizes")
    )

    # Slice parameters to extract first part [0:21]
    initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), f"{prefix}starts"))
    initializers.append(numpy_helper.from_array(np.array([21], dtype=np.int64), f"{prefix}ends"))
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), f"{prefix}axes"))

    # Pad parameters to restore [1, 64] from [1, 21]
    # Pad format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    initializers.append(
        numpy_helper.from_array(np.array([0, 0, 0, 43], dtype=np.int64), f"{prefix}pads")
    )
    initializers.append(
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), f"{prefix}pad_value")
    )

    return [
        # Split input into 3 parts: [1,21], [1,21], [1,22]
        helper.make_node(
            "Split",
            [input_name, f"{prefix}split_sizes"],
            [f"{prefix}split1_out", f"{prefix}split2_out", f"{prefix}split3_out"],
            name=f"{prefix}split",
            axis=1,
        ),
        # Concat them back together (redundant operation)
        helper.make_node(
            "Concat",
            [f"{prefix}split1_out", f"{prefix}split2_out", f"{prefix}split3_out"],
            [f"{prefix}concat_out"],
            name=f"{prefix}concat",
            axis=1,
        ),
        # Slice to extract first part (which was split1_out) - this should be eliminated
        helper.make_node(
            "Slice",
            [f"{prefix}concat_out", f"{prefix}starts", f"{prefix}ends", f"{prefix}axes"],
            [f"{prefix}slice_out"],
            name=f"{prefix}slice",
        ),
        # Pad back to original shape for compatibility with template
        helper.make_node(
            "Pad",
            [f"{prefix}slice_out", f"{prefix}pads", f"{prefix}pad_value"],
            [output_name],
            name=f"{prefix}pad",
            mode="constant",
        ),
    ]
