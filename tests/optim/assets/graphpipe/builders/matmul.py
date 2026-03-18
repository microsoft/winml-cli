"""MatMul pattern builders for ORT Graph Optimization tests.

Contains builders for MatMul-based fusion patterns:
- MatMulAddRelu: MatMul → Add → Relu (with identity Unsqueeze/Squeeze pair)
- MatMulActivation: MatMul → Softmax (ORT only supports Softmax, NOT Sigmoid)
- MatMulTranspose: Transpose → MatMul → FusedMatMul (creates new op)
- MatMulScale: MatMul → Mul(scale)
- MatMulBN: MatMul → BatchNormalization
- DynamicQuantizeMatMul: DynamicQuantizeLinear → MatMulInteger (quantized inference)

References:
- onnxruntime/core/optimizer/matmul_activation_fusion.cc (line 24: only Softmax supported)
- onnxruntime/core/optimizer/matmul_transpose_fusion.cc (creates FusedMatMul in kMSDomain)
- onnxruntime/core/optimizer/matmul_scale_fusion.cc
- onnxruntime/core/optimizer/matmul_bn_fusion.cc
- onnxruntime/core/optimizer/qdq_transformer/dynamic_quantize_matmul_fusion.cc
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper


def matmul_add_relu_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build MatMul → Add → Relu pattern (shape-preserving: 64→64).

    P1-06: Unsqueeze→Squeeze pair placed AFTER Relu to not break MatMul+Add fusion.
    The identity pair tests P1-06 elimination capability.

    Pattern:
        Input [1,64] → MatMul → Add → Relu → Unsqueeze → Squeeze → Output [1,64]
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # MatMul weights: [64, 64] to preserve shape [1, 64] → [1, 64]
    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}weight")
    )
    # Add bias: [64] broadcasts to [1, 64]
    initializers.append(
        numpy_helper.from_array(rng.randn(64).astype(np.float32) * 0.1, f"{prefix}bias")
    )
    # Axes for Unsqueeze/Squeeze
    initializers.append(numpy_helper.from_array(np.array([2], dtype=np.int64), f"{prefix}axes"))

    return [
        helper.make_node(
            "MatMul", [input_name, f"{prefix}weight"], [f"{prefix}mm_out"], name=f"{prefix}matmul"
        ),
        helper.make_node(
            "Add", [f"{prefix}mm_out", f"{prefix}bias"], [f"{prefix}add_out"], name=f"{prefix}add"
        ),
        helper.make_node("Relu", [f"{prefix}add_out"], [f"{prefix}relu_out"], name=f"{prefix}relu"),
        # P1-06: Unsqueeze→Squeeze pair (identity) - placed AFTER Relu to not break MatMul+Add
        helper.make_node(
            "Unsqueeze",
            [f"{prefix}relu_out", f"{prefix}axes"],
            [f"{prefix}unsqueeze_out"],
            name=f"{prefix}unsqueeze",
        ),
        helper.make_node(
            "Squeeze",
            [f"{prefix}unsqueeze_out", f"{prefix}axes"],
            [output_name],
            name=f"{prefix}squeeze",
        ),
    ]


def matmul_activation_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build MatMul+Activation pattern: Input → MatMul → Softmax.

    P2-03: Tests matmul-activation-fusion capability (ORT name: MatMulActivationFusion).

    IMPORTANT: ORT's MatMulActivationFusion ONLY supports Softmax activation.
    See onnxruntime/core/optimizer/matmul_activation_fusion.cc line 24:
        IsSupportedOptypeVersionAndDomain(node, "Softmax", {1, 11, 13}, kOnnxDomain)

    This creates FusedMatMul → Softmax which can further fuse to FusedMatMulActivation.

    Pattern:
        Input [1,64] → MatMul → Softmax → Output [1,64]
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}weight")
    )

    return [
        helper.make_node(
            "MatMul", [input_name, f"{prefix}weight"], [f"{prefix}mm_out"], name=f"{prefix}matmul"
        ),
        # Use Softmax (NOT Sigmoid) - ORT only supports Softmax for this fusion
        helper.make_node(
            "Softmax", [f"{prefix}mm_out"], [output_name], name=f"{prefix}softmax", axis=-1
        ),
    ]


def matmul_transpose_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Transpose+MatMul pattern: Input → Transpose → MatMul.

    P4-03: Tests matmul-transpose-fusion capability (ORT name: MatmulTransposeFusion).

    IMPORTANT: This fusion creates a NEW operator "FusedMatMul" in kMSDomain.
    The Transpose is absorbed into FusedMatMul's transA/transB attributes.
    See onnxruntime/core/optimizer/matmul_transpose_fusion.cc lines 395-399.

    The fusion works when:
    - Transpose's last axis stays in last two dims after transpose
    - Batch dims keep same relative order
    - Data types are float, float16, double, or bfloat16

    Pattern:
        Input [64,64] → Transpose(perm=[1,0]) → MatMul → Transpose(perm=[1,0]) → Output [64,64]

    After fusion:
        Input [64,64] → FusedMatMul(transA=1) → Transpose(perm=[1,0]) → Output [64,64]

    Note: Input shape is [64,64] (square) for proper transpose handling.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # MatMul weights: [64, 64]
    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}weight")
    )

    return [
        # Transpose input: [64, 64] → [64, 64] (swaps dims)
        helper.make_node(
            "Transpose",
            [input_name],
            [f"{prefix}transpose_out"],
            name=f"{prefix}transpose",
            perm=[1, 0],
        ),
        helper.make_node(
            "MatMul",
            [f"{prefix}transpose_out", f"{prefix}weight"],
            [f"{prefix}mm_out"],
            name=f"{prefix}matmul",
        ),
        # Transpose back to restore expected output shape
        helper.make_node(
            "Transpose",
            [f"{prefix}mm_out"],
            [output_name],
            name=f"{prefix}transpose_back",
            perm=[1, 0],
        ),
    ]


def matmul_scale_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build MatMul+Scale pattern: Input → MatMul → Mul(scale).

    P4-02: Tests matmul-scale-fusion capability (ORT name: MatMulScaleFusion).
    MatMul followed by Mul with a scalar scale should fuse into MatMul with scaling.

    Pattern:
        Input [1,64] → MatMul → Mul(scale) → Output [1,64]
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # MatMul weights: [64, 64]
    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}weight")
    )
    # Scale factor (broadcast-compatible)
    initializers.append(
        numpy_helper.from_array(np.array([2.0], dtype=np.float32), f"{prefix}scale")
    )

    return [
        helper.make_node(
            "MatMul", [input_name, f"{prefix}weight"], [f"{prefix}mm_out"], name=f"{prefix}matmul"
        ),
        helper.make_node(
            "Mul", [f"{prefix}mm_out", f"{prefix}scale"], [output_name], name=f"{prefix}mul_scale"
        ),
    ]


def matmul_bn_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build MatMul+BatchNorm pattern: Input → MatMul → BatchNormalization.

    P4-01: Tests matmul-bn-fusion capability (ORT name: MatMul_BatchNormalization_Fusion).
    MatMul followed by BatchNormalization should fuse into a single operation.

    Pattern:
        Input [1,64] → MatMul → BatchNormalization → Output [1,64]
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # MatMul weights: [64, 64]
    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}weight")
    )
    # BatchNorm parameters: BN expects [C] for 2D input [N, C]
    initializers.append(numpy_helper.from_array(np.ones(64, dtype=np.float32), f"{prefix}bn_scale"))
    initializers.append(numpy_helper.from_array(np.zeros(64, dtype=np.float32), f"{prefix}bn_bias"))
    initializers.append(numpy_helper.from_array(np.zeros(64, dtype=np.float32), f"{prefix}bn_mean"))
    initializers.append(numpy_helper.from_array(np.ones(64, dtype=np.float32), f"{prefix}bn_var"))

    return [
        helper.make_node(
            "MatMul", [input_name, f"{prefix}weight"], [f"{prefix}mm_out"], name=f"{prefix}matmul"
        ),
        helper.make_node(
            "BatchNormalization",
            [
                f"{prefix}mm_out",
                f"{prefix}bn_scale",
                f"{prefix}bn_bias",
                f"{prefix}bn_mean",
                f"{prefix}bn_var",
            ],
            [output_name],
            name=f"{prefix}bn",
            epsilon=1e-5,
        ),
    ]


def dynamic_quantize_matmul_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build DynamicQuantize+MatMul pattern: Input → DynamicQuantizeLinear → MatMulInteger.

    P4-04: Tests dynamic-quantize-matmul-fusion capability (ORT name: DynamicQuantizeMatMulFusion).
    DynamicQuantizeLinear followed by MatMulInteger should fuse for efficient quantized inference.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Quantized weight (int8) and its scale/zero_point
    weight_int8 = (rng.randn(64, 64) * 10).astype(np.int8)
    initializers.append(numpy_helper.from_array(weight_int8, f"{prefix}weight_quant"))
    initializers.append(
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), f"{prefix}weight_scale")
    )
    initializers.append(numpy_helper.from_array(np.array(0, dtype=np.int8), f"{prefix}weight_zp"))

    return [
        # DynamicQuantizeLinear: float32 → uint8 + scale + zero_point
        helper.make_node(
            "DynamicQuantizeLinear",
            [input_name],
            [f"{prefix}input_quant", f"{prefix}input_scale", f"{prefix}input_zp"],
            name=f"{prefix}dyn_quant",
        ),
        # MatMulInteger: uint8 x int8 -> int32
        helper.make_node(
            "MatMulInteger",
            [
                f"{prefix}input_quant",
                f"{prefix}weight_quant",
                f"{prefix}input_zp",
                f"{prefix}weight_zp",
            ],
            [f"{prefix}mm_int_out"],
            name=f"{prefix}matmul_int",
        ),
        # Cast int32 → float32
        helper.make_node(
            "Cast",
            [f"{prefix}mm_int_out"],
            [f"{prefix}mm_float"],
            name=f"{prefix}cast_float",
            to=TensorProto.FLOAT,
        ),
        # Rescale: multiply by input_scale * weight_scale
        helper.make_node(
            "Mul",
            [f"{prefix}input_scale", f"{prefix}weight_scale"],
            [f"{prefix}combined_scale"],
            name=f"{prefix}mul_scales",
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}mm_float", f"{prefix}combined_scale"],
            [output_name],
            name=f"{prefix}rescale",
        ),
    ]
