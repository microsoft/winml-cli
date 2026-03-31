# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Gemm pattern builders for ORT Graph Optimization tests.

Contains builders for Gemm-based fusion patterns:
- GemmActivation: Gemm → Relu (standard activation fusion)
- GemmSum: Gemm → Sum (NOT Gemm → Gemm - ORT expects Sum operator)
- GemmTranspose: Transpose → Gemm (fold transpose into transA/transB)

References:
- onnxruntime/core/optimizer/gemm_activation_fusion.cc
- onnxruntime/core/optimizer/gemm_sum_fusion.cc (line 117: expects "Sum" op, not "Gemm")
- onnxruntime/core/optimizer/gemm_transpose_fusion.cc
"""

from __future__ import annotations

import numpy as np
from onnx import helper, numpy_helper


def gemm_activation_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Gemm+Activation pattern: Input → Gemm → Relu.

    P3-01: Tests gemm-activation-fusion capability (ORT name: GemmActivationFusion).
    Gemm followed by Relu should fuse to a single fused Gemm node.

    Pattern:
        Input [1,64] → Gemm(A, B, C) → Relu → Output [1,64]
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Gemm: Y = alpha * A * B + beta * C
    # For shape [1, 64] input, we need [64, 64] weights and [64] bias
    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}weight")
    )
    initializers.append(
        numpy_helper.from_array(rng.randn(64).astype(np.float32) * 0.1, f"{prefix}bias")
    )

    return [
        helper.make_node(
            "Gemm",
            [input_name, f"{prefix}weight", f"{prefix}bias"],
            [f"{prefix}gemm_out"],
            name=f"{prefix}gemm",
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0,
        ),
        helper.make_node("Relu", [f"{prefix}gemm_out"], [output_name], name=f"{prefix}relu"),
    ]


def gemm_sum_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build Gemm+Sum pattern: Input → Gemm → Sum.

    P3-06: Tests gemm-sum-fusion capability (ORT name: GemmSumFusion).

    IMPORTANT: ORT's GemmSumFusion expects Gemm followed by Sum, NOT Gemm followed by Gemm.
    See onnxruntime/core/optimizer/gemm_sum_fusion.cc lines 116-122:
        IsSupportedOptypeVersionAndDomain(output_node, "Sum", {1, 6, 8, 13})

    The fusion works when:
    - Gemm has only A and B inputs (no C bias) - this is critical!
    - Gemm has exactly one output edge to Sum
    - Sum has exactly two inputs
    - The other Sum input has compatible shape for broadcast

    Pattern:
        Input [1,64] → Gemm(A,B) → Sum(gemm_out, bias) → Output [1,64]

    After fusion:
        Input [1,64] → Gemm(A, B, C=bias) → Output [1,64]

    Note: The Gemm must NOT have a C input initially; the Sum's other input becomes C.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Gemm weights: [64, 64] - A * B = [1,64] x [64,64] = [1,64]
    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}weight")
    )
    # Bias for Sum - this becomes C in the fused Gemm
    # Shape [64] or [1, 64] for proper broadcast with Gemm output [1, 64]
    initializers.append(
        numpy_helper.from_array(rng.randn(64).astype(np.float32) * 0.1, f"{prefix}bias")
    )

    return [
        # Gemm WITHOUT C input (only A and B) - required for GemmSumFusion
        helper.make_node(
            "Gemm",
            [input_name, f"{prefix}weight"],  # Only 2 inputs, no bias (C)
            [f"{prefix}gemm_out"],
            name=f"{prefix}gemm",
            alpha=1.0,
            beta=1.0,  # beta will be used after fusion with Sum
            transA=0,
            transB=0,
        ),
        # Sum adds bias to Gemm output - ORT fuses this into Gemm's C input
        helper.make_node(
            "Sum",
            [f"{prefix}gemm_out", f"{prefix}bias"],
            [output_name],
            name=f"{prefix}sum",
        ),
    ]


def gemm_transpose_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Transpose+Gemm pattern: Input → Transpose → Gemm.

    P3-07: Tests gemm-transpose-fusion capability (ORT name: GemmTransposeFusion).
    Transpose followed by Gemm should fold transpose into Gemm's transA/transB attributes.

    Pattern:
        Input [64,64] → Transpose(perm=[1,0]) → Gemm → Transpose(perm=[1,0]) → Output [64,64]

    Note: Input shape is [64,64] (square) for proper transpose handling.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Gemm weights: [64, 64]
    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}weight")
    )
    initializers.append(
        numpy_helper.from_array(rng.randn(64).astype(np.float32) * 0.1, f"{prefix}bias")
    )

    return [
        # Transpose [64, 64] → [64, 64] (swaps dims, should fold into Gemm's transA)
        helper.make_node(
            "Transpose",
            [input_name],
            [f"{prefix}transpose_out"],
            name=f"{prefix}transpose",
            perm=[1, 0],
        ),
        # Gemm with transposed input
        helper.make_node(
            "Gemm",
            [f"{prefix}transpose_out", f"{prefix}weight", f"{prefix}bias"],
            [f"{prefix}gemm_out"],
            name=f"{prefix}gemm",
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=1,  # Transpose B to get proper output shape
        ),
        # Transpose back to restore expected output shape
        helper.make_node(
            "Transpose",
            [f"{prefix}gemm_out"],
            [output_name],
            name=f"{prefix}transpose_back",
            perm=[1, 0],
        ),
    ]
