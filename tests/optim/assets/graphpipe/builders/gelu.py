"""GELU pattern builders for ONNX optimization testing.

This module provides builder functions for various GELU activation patterns
that can be fused by ONNX Runtime optimizers.
"""

import math

import numpy as np
from onnx import helper, numpy_helper


__all__ = [
    "bias_gelu_builder",
    "fast_gelu_builder",
    "gelu_approximation_builder",
    "gelu_fusion_builder",
    "quick_gelu_builder",
]


def gelu_fusion_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build decomposed GELU pattern for GeluFusionL2.

    Tests gelu-fusion capability (ORT name: GeluFusionL2).
    Formula: 0.5*x*(1+erf(x/sqrt(2)))

    This is the standard GELU decomposition that ORT's GeluFusionL2 optimizer
    recognizes and fuses into a single Gelu op.

    From onnxruntime/core/optimizer/gelu_fusion.cc:
    - Pattern: x * 0.5 * (1 + erf(x / sqrt(2)))
    - Fuses to: Gelu(x)

    Note: This is different from GeluApproximation which converts Gelu to FastGelu.
    GeluFusionL2 CREATES a Gelu op from decomposed pattern.

    Pattern:
        Input [1,64] → Div(sqrt2) → Erf → Add(1) → Mul(x) → Mul(0.5) → Output [1,64]
    """
    # Constants for GELU computation
    initializers.append(
        numpy_helper.from_array(np.array([math.sqrt(2)], dtype=np.float32), f"{prefix}sqrt2")
    )
    initializers.append(numpy_helper.from_array(np.array([1.0], dtype=np.float32), f"{prefix}one"))
    initializers.append(numpy_helper.from_array(np.array([0.5], dtype=np.float32), f"{prefix}half"))

    return [
        # x/sqrt(2)
        helper.make_node(
            "Div", [input_name, f"{prefix}sqrt2"], [f"{prefix}div"], name=f"{prefix}div"
        ),
        # erf(x/sqrt(2))
        helper.make_node("Erf", [f"{prefix}div"], [f"{prefix}erf"], name=f"{prefix}erf"),
        # 1 + erf(x/sqrt(2))
        helper.make_node(
            "Add", [f"{prefix}erf", f"{prefix}one"], [f"{prefix}add1"], name=f"{prefix}add1"
        ),
        # x*(1 + erf(x/sqrt(2)))
        helper.make_node(
            "Mul",
            [input_name, f"{prefix}add1"],
            [f"{prefix}mul1"],
            name=f"{prefix}mul1",
        ),
        # 0.5*x*(1 + erf(x/sqrt(2)))
        helper.make_node(
            "Mul", [f"{prefix}mul1", f"{prefix}half"], [output_name], name=f"{prefix}mul2"
        ),
    ]


def bias_gelu_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build BiasGelu pattern: Add → Div → Erf → Add → Mul → Mul."""
    rng = np.random.RandomState(hash(prefix) % (2**32))

    initializers.append(
        numpy_helper.from_array(rng.randn(64).astype(np.float32) * 0.1, f"{prefix}bias")
    )
    initializers.append(
        numpy_helper.from_array(np.array([math.sqrt(2)], dtype=np.float32), f"{prefix}sqrt2")
    )
    initializers.append(numpy_helper.from_array(np.array([1.0], dtype=np.float32), f"{prefix}one"))
    initializers.append(numpy_helper.from_array(np.array([0.5], dtype=np.float32), f"{prefix}half"))

    return [
        helper.make_node(
            "Add", [input_name, f"{prefix}bias"], [f"{prefix}biased"], name=f"{prefix}add_bias"
        ),
        helper.make_node(
            "Div", [f"{prefix}biased", f"{prefix}sqrt2"], [f"{prefix}div"], name=f"{prefix}div"
        ),
        helper.make_node("Erf", [f"{prefix}div"], [f"{prefix}erf"], name=f"{prefix}erf"),
        helper.make_node(
            "Add", [f"{prefix}erf", f"{prefix}one"], [f"{prefix}add1"], name=f"{prefix}add1"
        ),
        helper.make_node(
            "Mul", [f"{prefix}biased", f"{prefix}add1"], [f"{prefix}mul1"], name=f"{prefix}mul1"
        ),
        helper.make_node(
            "Mul", [f"{prefix}mul1", f"{prefix}half"], [output_name], name=f"{prefix}mul2"
        ),
    ]


def fast_gelu_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build FastGelu pattern: tanh approximation of GELU.

    Tests fast-gelu-fusion capability (ORT name: FastGeluFusion).
    Formula: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044714998453855515*x^3)))

    From fast_gelu_fusion.cc, the pattern must be exact without additional ops.
    ORT expects scalar (0-D) constants, not [1]-shaped arrays.
    """
    # Constants for FastGelu computation - must be scalar (0-D) tensors
    initializers.append(
        numpy_helper.from_array(
            np.asarray([0.044714998453855515]).astype(np.float32).reshape(()), f"{prefix}coeff"
        )
    )
    initializers.append(
        numpy_helper.from_array(
            np.asarray([0.7978845834732056]).astype(np.float32).reshape(()), f"{prefix}sqrt_2_pi"
        )
    )
    initializers.append(
        numpy_helper.from_array(np.asarray([3.0]).astype(np.float32).reshape(()), f"{prefix}three")
    )
    initializers.append(
        numpy_helper.from_array(np.asarray([1.0]).astype(np.float32).reshape(()), f"{prefix}one")
    )
    initializers.append(
        numpy_helper.from_array(np.asarray([0.5]).astype(np.float32).reshape(()), f"{prefix}half")
    )

    return [
        # x^3
        helper.make_node(
            "Pow", [input_name, f"{prefix}three"], [f"{prefix}x_cubed"], name=f"{prefix}pow"
        ),
        # 0.044714998453855515*x^3
        helper.make_node(
            "Mul",
            [f"{prefix}x_cubed", f"{prefix}coeff"],
            [f"{prefix}term"],
            name=f"{prefix}mul1",
        ),
        # x + 0.044714998453855515*x^3
        helper.make_node(
            "Add", [input_name, f"{prefix}term"], [f"{prefix}sum"], name=f"{prefix}add1"
        ),
        # sqrt(2/pi)*(x + 0.044714998453855515*x^3)
        helper.make_node(
            "Mul",
            [f"{prefix}sum", f"{prefix}sqrt_2_pi"],
            [f"{prefix}scaled"],
            name=f"{prefix}mul2",
        ),
        # tanh(...)
        helper.make_node("Tanh", [f"{prefix}scaled"], [f"{prefix}tanh"], name=f"{prefix}tanh"),
        # 1 + tanh(...)
        helper.make_node(
            "Add", [f"{prefix}tanh", f"{prefix}one"], [f"{prefix}add2"], name=f"{prefix}add2"
        ),
        # x * 0.5 (ORT expects this structure: mul6 = x * 0.5)
        helper.make_node(
            "Mul", [input_name, f"{prefix}half"], [f"{prefix}mul6"], name=f"{prefix}mul6"
        ),
        # (x * 0.5) * (1 + tanh(...)) (ORT expects: mul5 = mul6 * add2)
        helper.make_node(
            "Mul", [f"{prefix}mul6", f"{prefix}add2"], [output_name], name=f"{prefix}mul5"
        ),
    ]


def quick_gelu_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build QuickGelu pattern: sigmoid approximation.

    Tests quick-gelu-fusion capability (ORT name: QuickGeluFusion).
    Formula: x*sigmoid(1.702*x)
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Add bias for more realistic pattern
    initializers.append(
        numpy_helper.from_array(rng.randn(64).astype(np.float32) * 0.1, f"{prefix}bias")
    )
    # Constant for QuickGelu
    initializers.append(
        numpy_helper.from_array(np.array([1.702], dtype=np.float32), f"{prefix}alpha")
    )

    return [
        # Add bias
        helper.make_node(
            "Add", [input_name, f"{prefix}bias"], [f"{prefix}biased"], name=f"{prefix}add_bias"
        ),
        # 1.702*x
        helper.make_node(
            "Mul",
            [f"{prefix}biased", f"{prefix}alpha"],
            [f"{prefix}scaled"],
            name=f"{prefix}mul1",
        ),
        # sigmoid(1.702*x)
        helper.make_node(
            "Sigmoid", [f"{prefix}scaled"], [f"{prefix}sigmoid"], name=f"{prefix}sigmoid"
        ),
        # x*sigmoid(1.702*x)
        helper.make_node(
            "Mul",
            [f"{prefix}biased", f"{prefix}sigmoid"],
            [output_name],
            name=f"{prefix}mul2",
        ),
    ]


def gelu_approximation_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build GeluApproximation pattern: standard GELU with Erf.

    Tests gelu-approximation capability (ORT name: GeluApproximation).
    Formula: 0.5*x*(1+erf(x/sqrt(2)))
    This is the standard GELU, ORT can fuse it to a single Gelu op.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    initializers.append(
        numpy_helper.from_array(rng.randn(64).astype(np.float32) * 0.1, f"{prefix}bias")
    )
    initializers.append(
        numpy_helper.from_array(np.array([math.sqrt(2)], dtype=np.float32), f"{prefix}sqrt2")
    )
    initializers.append(numpy_helper.from_array(np.array([1.0], dtype=np.float32), f"{prefix}one"))
    initializers.append(numpy_helper.from_array(np.array([0.5], dtype=np.float32), f"{prefix}half"))

    return [
        # Add bias
        helper.make_node(
            "Add", [input_name, f"{prefix}bias"], [f"{prefix}biased"], name=f"{prefix}add_bias"
        ),
        # x/sqrt(2)
        helper.make_node(
            "Div", [f"{prefix}biased", f"{prefix}sqrt2"], [f"{prefix}div"], name=f"{prefix}div"
        ),
        # erf(x/sqrt(2))
        helper.make_node("Erf", [f"{prefix}div"], [f"{prefix}erf"], name=f"{prefix}erf"),
        # 1 + erf(x/sqrt(2))
        helper.make_node(
            "Add", [f"{prefix}erf", f"{prefix}one"], [f"{prefix}add1"], name=f"{prefix}add1"
        ),
        # x*(1 + erf(x/sqrt(2)))
        helper.make_node(
            "Mul",
            [f"{prefix}biased", f"{prefix}add1"],
            [f"{prefix}mul1"],
            name=f"{prefix}mul1",
        ),
        # 0.5*x*(1 + erf(x/sqrt(2)))
        helper.make_node(
            "Mul", [f"{prefix}mul1", f"{prefix}half"], [output_name], name=f"{prefix}mul2"
        ),
    ]
