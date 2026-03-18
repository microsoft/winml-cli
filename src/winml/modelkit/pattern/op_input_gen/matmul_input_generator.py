# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generators for matrix multiplication ONNX operators.

This module provides input generators for matrix multiplication operators:
- MatMul: General matrix multiplication with broadcasting
- Gemm: General matrix multiplication with optional scaling and bias

Matrix multiplication operators perform matrix product operations on tensors,
following Numpy's matmul semantics with broadcasting support for higher dimensions.
"""

from operator import ne

import winml.modelkit.onnx.dtypes as dtypes

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    OpInputGenerator,
    QDQParameterConfig,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class MatMulInputGenerator(OpInputGenerator):
    """Input generator for MatMul operator.

    MatMul signature:
    - Inputs: A (N-D matrix), B (N-D matrix)
    - Attributes: None
    - Output: Matrix product following numpy.matmul semantics

    Operation: Matrix multiplication with broadcasting on higher dimensions.
    """

    op_name = "MatMul"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """MatMul has no attributes."""
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for MatMul.

        Covers various matrix shapes and broadcasting patterns.

        Matrix multiplication rules:
        - For 2D: A(M, K) @ B(K, N) -> (M, N)
        - For ND: Higher dims must broadcast, last 2 dims follow 2D rules

        Returns 36 pairs covering all combinations of 1D-6D x 1D-6D, ensuring:
        - A_shape == B_shape (same shape cases)
        - Unidirectional broadcast A -> B (A has smaller/broadcastable dims)
        - Unidirectional broadcast B -> A (B has smaller/broadcastable dims)
        - Bidirectional broadcast A <-> B (both have broadcastable dims)
        """
        combinations = []

        # Common matrix shape pairs for comprehensive testing
        shape_pairs = [
            # 1D x 1D: dot product (A_shape == B_shape)
            ((4,), (4,)),
            # 1D x 2D: vector-matrix product (B -> A broadcast)
            ((4,), (4, 5)),
            # 1D x 3D
            ((4,), (2, 4, 5)),
            # 1D x 4D
            ((4,), (2, 3, 4, 5)),
            # 1D x 5D
            ((4,), (2, 2, 3, 4, 5)),
            # 1D x 6D
            ((4,), (2, 2, 2, 3, 4, 5)),
            # 2D x 1D: matrix-vector product (A -> B broadcast)
            ((3, 4), (4,)),
            # 2D x 2D: standard matrix multiplication (A_shape == B_shape for square)
            ((4, 4), (4, 4)),
            # 2D x 3D
            ((3, 4), (2, 4, 5)),
            # 2D x 4D
            ((3, 4), (2, 3, 4, 5)),
            # 2D x 5D
            ((3, 4), (2, 2, 3, 4, 5)),
            # 2D x 6D
            ((3, 4), (2, 2, 2, 3, 4, 5)),
            # 3D x 1D
            ((2, 3, 4), (4,)),
            # 3D x 2D
            ((2, 3, 4), (4, 5)),
            # 3D x 3D: batched (A_shape == B_shape)
            ((2, 3, 4), (2, 4, 5)),
            # 3D x 4D (unidirectional broadcast A -> B)
            ((1, 3, 4), (2, 3, 4, 5)),
            # 3D x 5D
            ((3, 3, 4), (2, 2, 3, 4, 5)),
            # 3D x 6D
            ((3, 3, 4), (2, 2, 2, 3, 4, 5)),
            # 4D x 1D
            ((2, 3, 3, 4), (4,)),
            # 4D x 2D
            ((2, 3, 3, 4), (4, 5)),
            # 4D x 3D (unidirectional broadcast B -> A)
            ((2, 3, 3, 4), (1, 4, 5)),
            # 4D x 4D: batched (A_shape == B_shape)
            ((2, 3, 4, 5), (2, 3, 5, 6)),
            # 4D x 5D
            ((2, 3, 4, 5), (2, 2, 3, 5, 6)),
            # 4D x 6D (bidirectional broadcast A <-> B)
            ((2, 1, 4, 5), (2, 2, 2, 3, 5, 6)),
            # 5D x 1D
            ((2, 2, 3, 3, 4), (4,)),
            # 5D x 2D
            ((2, 2, 3, 3, 4), (4, 5)),
            # 5D x 3D
            ((2, 2, 3, 3, 4), (3, 4, 5)),
            # 5D x 4D
            ((2, 2, 3, 4, 5), (2, 3, 5, 6)),
            # 5D x 5D: batched (A_shape == B_shape)
            ((2, 2, 3, 4, 5), (2, 2, 3, 5, 6)),
            # 5D x 6D
            ((2, 2, 3, 4, 5), (2, 2, 2, 3, 5, 6)),
            # 6D x 1D
            ((2, 2, 2, 3, 3, 4), (4,)),
            # 6D x 2D
            ((2, 2, 2, 3, 3, 4), (4, 5)),
            # 6D x 3D
            ((2, 2, 2, 3, 3, 4), (3, 4, 5)),
            # 6D x 4D
            ((2, 2, 2, 3, 4, 5), (2, 3, 5, 6)),
            # 6D x 5D
            ((2, 2, 2, 3, 4, 5), (2, 2, 3, 5, 6)),
            # 6D x 6D: batched (A_shape == B_shape)
            ((2, 2, 2, 3, 4, 5), (2, 2, 2, 3, 5, 6)),
        ]

        for a_shape, b_shape in shape_pairs:
            combinations.append(
                {
                    "A": InputShapeConstraint(a_shape),
                    "B": InputShapeConstraint(b_shape),
                }
            )
        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for MatMul operator testing.

        Args:
            properties: Base properties containing A_shape and B_shape

        Returns:
            Updated properties with MatMul-specific derived values (A_dim, B_dim)
        """
        item = properties.copy()
        item["A_dim"] = len(item["A_shape"])
        item["B_dim"] = len(item["B_shape"])
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes with infinite possibilities
        """
        return ["A_shape", "B_shape"]

    def get_qdq_config(self):
        return {
            "A": QDQParameterConfig(support_activation=True, support_weight=True),
            "B": QDQParameterConfig(support_activation=True, support_weight=True),
        }


@register_runtime_checker_op
class GemmInputGenerator(OpInputGenerator):
    """Input generator for Gemm operator.

    Gemm signature:
    - Inputs: A (2D matrix), B (2D matrix), C (optional, broadcastable to output)
    - Attributes: alpha (float, default 1.0), beta (float, default 1.0),
                  transA (int, default 0), transB (int, default 0)
    - Output: Y = alpha * A' * B' + beta * C

    Operation: General matrix multiplication with optional transpose, scaling, and bias.
    A' = transpose(A) if transA else A
    B' = transpose(B) if transB else B
    """

    op_name = "Gemm"
    expand_optionals = False

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute sets for Gemm.

        Attributes:
        - alpha: Scalar multiplier for A*B (use single value close to 1.0)
        - beta: Scalar multiplier for C (use single value close to 1.0)
        - transA: Whether to transpose A (0 or 1)
        - transB: Whether to transpose B (0 or 1)
        """
        return {
            "alpha": [None, 1.0],  # Single value for float attribute
            "beta": [None, 1.0],  # Single value for float attribute
            "transA": [None, 0, 1],
            "transB": [None, 0, 1],
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for Gemm.

        Gemm operates on 2D matrices only. Shapes depend on transpose flags:
        - A: (M, K) if transA=0, (K, M) if transA=1
        - B: (K, N) if transB=0, (N, K) if transB=1
        - C: broadcastable to (M, N), can be None

        We create 16 combinations by varying:
        - A shape for transA=0 vs transA=1 (2 options)
        - B shape for transB=0 vs transB=1 (2 options)
        - C shape: same as output, 1D, scalar, or None (4 options)
        Total: 2 * 2 * 4 = 16 combinations
        """
        combinations = []

        # Use m=3, k=4, n=5 as base dimensions
        # Output shape will be (3, 5) after multiplication
        m, k, n = 3, 4, 5

        # A shapes: for transA=0 (m, k), for transA=1 (k, m)
        a_shapes = [
            (m, k),  # transA=0: (3, 4)
            (k, m),  # transA=1: (4, 3) -> becomes (3, 4) after transpose
        ]

        # B shapes: for transB=0 (k, n), for transB=1 (n, k)
        b_shapes = [
            (k, n),  # transB=0: (4, 5)
            (n, k),  # transB=1: (5, 4) -> becomes (4, 5) after transpose
        ]

        # C options: full output, 1D, scalar, or None (no C)
        # Output after A@B is always (m, n) = (3, 5)
        c_options: list[InputConstraint | None] = [
            InputShapeConstraint((m, n)),  # Full bias matrix: (3, 5)
            InputShapeConstraint((n,)),  # 1D bias (broadcasts to (3, 5)): (5,)
            InputShapeConstraint(()),  # Scalar broadcast: ()
            None,  # C not provided
        ]

        # Generate all 16 combinations: 2 A shapes × 2 B shapes × 4 C options
        for a_shape in a_shapes:
            for b_shape in b_shapes:
                for c_option in c_options:
                    combination: dict[str, InputConstraint] = {
                        "A": InputShapeConstraint(a_shape),
                        "B": InputShapeConstraint(b_shape),
                    }
                    combination["C"] = c_option

                    combinations.append(combination)

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for Gemm operator testing.

        Args:
            properties: Base properties containing A_shape, B_shape, and optionally C_shape

        Returns:
            Updated properties with Gemm-specific derived values (A_dim, B_dim, C_dim)
        """
        item = properties.copy()
        item["A_dim"] = len(item["A_shape"])
        item["B_dim"] = len(item["B_shape"])
        # C is optional, only add C_dim if C_shape exists
        if "C_shape" in item:
            item["C_dim"] = len(item["C_shape"]) if item["C_shape"] else 0
        else:
            item["C_dim"] = 0
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes with infinite possibilities
        """
        return ["A_shape", "B_shape", "C_shape"]

    def get_qdq_config(self):
        # https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/operators/gemm.py
        return {
            "A": QDQParameterConfig(support_activation=True),
            "B": QDQParameterConfig(support_weight=True),
            "C": QDQParameterConfig(weight_type=dtypes.SupportedONNXType.INT32),
        }
