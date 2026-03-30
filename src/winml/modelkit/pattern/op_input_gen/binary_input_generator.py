# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generators for binary ONNX operators.

This module provides input generators for binary operators that take two inputs
and support broadcasting. Binary operators perform element-wise operations on
two tensors with Numpy-style broadcasting.

Broadcasting test coverage:
    For inputs A and B, we test 4 broadcasting scenarios:
    1. A broadcasts to B (A has fewer/smaller dims, B is larger)
    2. B broadcasts to A (B has fewer/smaller dims, A is larger)
    3. Bidirectional broadcast (both A and B broadcast to common shape)
    4. Equal shapes (no broadcasting needed)

Dimensionality coverage:
    - Test dimensions from 1D to 6D (max per ONNX spec)
    - Axis sizes limited to max 6 per axis (as per spec)
    - Ordered from smallest to largest dimensions
"""

from typing import Any

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    OpInputGenerator,
    QDQParameterConfig,
    register_runtime_checker_op,
)


class BinaryInputGenerator(OpInputGenerator):
    """Universal input generator for binary ONNX operators.

    Binary operators perform element-wise operations on two input tensors
    with Numpy-style broadcasting support.

    Supported operators:
    - Arithmetic: Add, Sub, Mul, Div, Pow
    - Logical: And, Or
    - Bitwise: BitwiseAnd, BitwiseOr, BitwiseXor
    - Comparison: Equal, Greater, GreaterOrEqual, Less, LessOrEqual
    - Activation: PRelu

    Operator characteristics (based on Add documentation):
    - Inputs: Two tensors A and B of type T
    - Output: Single tensor of same type T
    - Attributes: None (most binary ops have no attributes)
    - Operation: Element-wise transformation with broadcasting
    - Broadcasting: Multidirectional (Numpy-style) for most ops
                   Unidirectional for PRelu (slope broadcasts to X)

    Test coverage strategy:
    - Broadcasting patterns: A->B, B->A, A<->B, equal shapes
    - Input dimensions: 1D through 6D (max per spec)
    - Various shapes to test different broadcasting scenarios
    - Ordered from smallest to largest dimensions
    """

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Returns finite attribute sets for binary operators.

        Most binary operators have no attributes, so return empty dict.
        Subclasses can override if specific operators have attributes.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Returns comprehensive input combinations for binary operators.

        Coverage strategy:
        1. Broadcasting patterns (for each dimension level):
           - A broadcasts to B (A smaller, B larger)
           - B broadcasts to A (B smaller, A larger)
           - Bidirectional broadcast (both broadcast to common shape)
           - Equal shapes (no broadcasting)

        2. Dimensionality: 1D through 6D

        3. Shape patterns:
           - Scalar vs tensor
           - Different dimension counts
           - Different axis sizes
           - Compatible broadcasting shapes

        Note: Parameter names vary (A/B for most ops, X/slope for PRelu).
        This method automatically detects the correct parameter names.
        """
        # Get input parameter names from the operator schema
        first_param = self.op_input_names[0]  # Usually 'A' or 'X'
        second_param = self.op_input_names[1]  # Usually 'B' or 'slope'

        return [
            # Group by max dimensions to ensure coverage across dimensionalities
            # ===== 0D Inputs (dimension 0) =====
            {
                first_param: InputShapeConstraint(()),
                second_param: InputShapeConstraint(()),
            },
            # ===== 1D Inputs (dimension 1) =====
            # Equal shapes - no broadcasting
            {
                first_param: InputShapeConstraint((6,)),
                second_param: InputShapeConstraint((6,)),
            },
            # 0 <=> 1
            {
                first_param: InputShapeConstraint(()),
                second_param: InputShapeConstraint((6,)),
            },
            {
                first_param: InputShapeConstraint((6,)),
                second_param: InputShapeConstraint(()),
            },
            # A broadcasts to B (scalar to vector)
            {
                first_param: InputShapeConstraint((1,)),
                second_param: InputShapeConstraint((6,)),
            },
            # B broadcasts to A (scalar to vector)
            {
                first_param: InputShapeConstraint((6,)),
                second_param: InputShapeConstraint((1,)),
            },
            # ===== 2D Inputs (dimension 2) =====
            # Equal shapes
            {
                first_param: InputShapeConstraint((4, 5)),
                second_param: InputShapeConstraint((4, 5)),
            },
            # 0 <=> 2
            {
                first_param: InputShapeConstraint(()),
                second_param: InputShapeConstraint((4, 5)),
            },
            {
                first_param: InputShapeConstraint((4, 5)),
                second_param: InputShapeConstraint(()),
            },
            # A broadcasts to B (1D to 2D)
            {
                first_param: InputShapeConstraint((5,)),
                second_param: InputShapeConstraint((4, 5)),
            },
            # B broadcasts to A (1D to 2D)
            {
                first_param: InputShapeConstraint((4, 5)),
                second_param: InputShapeConstraint((5,)),
            },
            # Bidirectional broadcast (both have size-1 dims)
            {
                first_param: InputShapeConstraint((4, 1)),
                second_param: InputShapeConstraint((1, 5)),
            },
            # A broadcasts to B (column vector to matrix)
            {
                first_param: InputShapeConstraint((4, 1)),
                second_param: InputShapeConstraint((4, 5)),
            },
            # B broadcasts to A (row vector to matrix)
            {
                first_param: InputShapeConstraint((4, 5)),
                second_param: InputShapeConstraint((1, 5)),
            },
            # ===== 3D Inputs (dimension 3) =====
            # Equal shapes
            {
                first_param: InputShapeConstraint((3, 2, 5)),
                second_param: InputShapeConstraint((3, 2, 5)),
            },
            # 0 <=> 3
            {
                first_param: InputShapeConstraint(()),
                second_param: InputShapeConstraint((3, 2, 5)),
            },
            {
                first_param: InputShapeConstraint((3, 2, 5)),
                second_param: InputShapeConstraint(()),
            },
            # 1D to 3D broadcasting patterns from P0 models
            {
                first_param: InputShapeConstraint((768,)),
                second_param: InputShapeConstraint((1, 77, 768)),
            },
            # A broadcasts to B (2D to 3D)
            {
                first_param: InputShapeConstraint((2, 5)),
                second_param: InputShapeConstraint((3, 2, 5)),
            },
            # B broadcasts to A (2D to 3D)
            {
                first_param: InputShapeConstraint((3, 2, 5)),
                second_param: InputShapeConstraint((2, 5)),
            },
            # Bidirectional broadcast in 3D
            {
                first_param: InputShapeConstraint((3, 1, 5)),
                second_param: InputShapeConstraint((1, 2, 1)),
            },
            # A broadcasts to B (with size-1 dims)
            {
                first_param: InputShapeConstraint((3, 1, 1)),
                second_param: InputShapeConstraint((3, 2, 5)),
            },
            # B broadcasts to A (with size-1 dims)
            {
                first_param: InputShapeConstraint((3, 2, 5)),
                second_param: InputShapeConstraint((1, 1, 5)),
            },
            # ===== 4D Inputs (dimension 4) =====
            # Equal shapes (batch, channels, height, width)
            {
                first_param: InputShapeConstraint((2, 4, 5, 6)),
                second_param: InputShapeConstraint((2, 4, 5, 6)),
            },
            # 0 <=> 4
            {
                first_param: InputShapeConstraint(()),
                second_param: InputShapeConstraint((2, 3, 4, 5)),
            },
            {
                first_param: InputShapeConstraint((2, 3, 4, 5)),
                second_param: InputShapeConstraint(()),
            },
            # Convnext (B to A)
            {
                first_param: InputShapeConstraint((1, 56, 56, 96)),
                second_param: InputShapeConstraint((96,)),
            },
            # detr (Bidirectional broadcast)
            {
                first_param: InputShapeConstraint((1, 25, 25, 1)),
                second_param: InputShapeConstraint((128,)),
            },
            # Convnext
            {
                first_param: InputShapeConstraint((96,)),
                second_param: InputShapeConstraint((1, 56, 56, 96)),
            },
            # A broadcasts to B (3D to 4D)
            {
                first_param: InputShapeConstraint((4, 5, 6)),
                second_param: InputShapeConstraint((2, 4, 5, 6)),
            },
            # B broadcasts to A (3D to 4D)
            {
                first_param: InputShapeConstraint((2, 4, 5, 6)),
                second_param: InputShapeConstraint((4, 5, 6)),
            },
            # Bidirectional broadcast in 4D
            {
                first_param: InputShapeConstraint((2, 1, 5, 1)),
                second_param: InputShapeConstraint((1, 4, 1, 6)),
            },
            # A broadcasts to B (channel-wise operation)
            {
                first_param: InputShapeConstraint((1, 4, 1, 1)),
                second_param: InputShapeConstraint((2, 4, 5, 6)),
            },
            # B broadcasts to A (spatial broadcast)
            {
                first_param: InputShapeConstraint((2, 4, 5, 6)),
                second_param: InputShapeConstraint((1, 1, 5, 6)),
            },
            # ===== 5D Inputs (dimension 5) =====
            # Equal shapes (batch, channels, depth, height, width)
            {
                first_param: InputShapeConstraint((2, 2, 3, 4, 5)),
                second_param: InputShapeConstraint((2, 2, 3, 4, 5)),
            },
            # A broadcasts to B (4D to 5D)
            {
                first_param: InputShapeConstraint((2, 3, 4, 5)),
                second_param: InputShapeConstraint((2, 2, 3, 4, 5)),
            },
            # B broadcasts to A (4D to 5D)
            {
                first_param: InputShapeConstraint((2, 2, 3, 4, 5)),
                second_param: InputShapeConstraint((2, 3, 4, 5)),
            },
            # Bidirectional broadcast in 5D
            {
                first_param: InputShapeConstraint((2, 1, 3, 1, 5)),
                second_param: InputShapeConstraint((1, 2, 1, 4, 1)),
            },
            # ===== 6D Inputs (dimension 6 - maximum) =====
            # Equal shapes
            {
                first_param: InputShapeConstraint((2, 2, 2, 2, 3, 3)),
                second_param: InputShapeConstraint((2, 2, 2, 2, 3, 3)),
            },
            # A broadcasts to B (5D to 6D)
            {
                first_param: InputShapeConstraint((2, 2, 2, 3, 3)),
                second_param: InputShapeConstraint((2, 2, 2, 2, 3, 3)),
            },
            # B broadcasts to A (5D to 6D)
            {
                first_param: InputShapeConstraint((2, 2, 2, 2, 3, 3)),
                second_param: InputShapeConstraint((2, 2, 2, 3, 3)),
            },
            # Bidirectional broadcast in 6D
            {
                first_param: InputShapeConstraint((2, 1, 2, 1, 3, 1)),
                second_param: InputShapeConstraint((1, 2, 1, 2, 1, 3)),
            },
        ]

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties from input shapes."""
        x_name = self.op_input_names[0]
        y_name = self.op_input_names[1]
        item = properties.copy()
        item[f"{x_name}_dim"] = len(item[f"{x_name}_shape"])
        item[f"{y_name}_dim"] = len(item[f"{y_name}_shape"])
        return item

    def _derive_broadcasting_properties(
        self, properties: dict, x_name: str | None = None, y_name: str | None = None
    ) -> dict:
        x_name = x_name if x_name is not None else self.op_input_names[0]
        y_name = y_name if y_name is not None else self.op_input_names[1]
        item = properties.copy()
        x_shape = tuple(item[f"{x_name}_shape"])
        y_shape = tuple(item[f"{y_name}_shape"])

        # Check if broadcasting occurs due to different dimensionality
        dim_x = len(x_shape)
        dim_y = len(y_shape)
        x_broadcasting_to_y = dim_x < dim_y
        y_broadcasting_to_x = dim_y < dim_x

        # make shapes same length by prepending 1s
        if len(x_shape) < len(y_shape):
            x_shape = (1,) * (len(y_shape) - len(x_shape)) + x_shape
        elif len(y_shape) < len(x_shape):
            y_shape = (1,) * (len(x_shape) - len(y_shape)) + y_shape

        # Check if broadcasting occurs due to different axis sizes
        # Only compare if both dimensions are integers (not symbolic strings)
        x_broadcasting_to_y = x_broadcasting_to_y or any(
            isinstance(xs, int) and isinstance(ys, int) and xs != 0 and xs < ys
            for xs, ys in zip(x_shape, y_shape, strict=False)
        )
        y_broadcasting_to_x = y_broadcasting_to_x or any(
            isinstance(ys, int) and isinstance(xs, int) and ys != 0 and ys < xs
            for xs, ys in zip(x_shape, y_shape, strict=False)
        )

        item["x_broadcasting_to_y"] = x_broadcasting_to_y
        item["y_broadcasting_to_x"] = y_broadcasting_to_x
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite value ranges."""
        x_name = self.op_input_names[0]
        y_name = self.op_input_names[1]
        return [f"{x_name}_shape", f"{y_name}_shape"]

    def get_qdq_config(self):
        """Return QDQ configuration for binary operator inputs."""
        return {
            self.op_input_names[0]: QDQParameterConfig(
                support_activation=True, support_weight=True
            ),
            self.op_input_names[1]: QDQParameterConfig(
                support_activation=True, support_weight=True
            ),
        }


# ===== Arithmetic Operators =====


@register_runtime_checker_op
class AddInputGenerator(BinaryInputGenerator):
    """Input generator for Add operator."""

    op_name = "Add"

    def derive_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Derive properties including broadcasting information."""
        item = super().derive_properties(properties)
        return self._derive_broadcasting_properties(item)


@register_runtime_checker_op
class SubInputGenerator(BinaryInputGenerator):
    """Input generator for Sub operator."""

    op_name = "Sub"

    def derive_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Derive properties including broadcasting information."""
        item = super().derive_properties(properties)
        return self._derive_broadcasting_properties(item)


@register_runtime_checker_op
class MulInputGenerator(BinaryInputGenerator):
    """Input generator for Mul operator."""

    op_name = "Mul"

    def derive_properties(self, properties):
        """Derive properties including broadcasting information."""
        item = super().derive_properties(properties)
        return self._derive_broadcasting_properties(item)


@register_runtime_checker_op
class DivInputGenerator(BinaryInputGenerator):
    """Input generator for Div operator."""

    op_name = "Div"

    def derive_properties(self, properties):
        """Derive properties including broadcasting information."""
        item = super().derive_properties(properties)
        return self._derive_broadcasting_properties(item)

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Returns input combinations for Div operator with division-by-zero protection.

        Overrides parent method to set min_max=(2, 3) for the divisor (B parameter)
        to avoid division by zero during testing.
        """
        combinations = super().get_input_and_infinite_attribute_combinations()
        # Get the divisor parameter name (second input)
        divisor_name = self.op_input_names[1]

        # Set min_max for B parameter to avoid division by zero
        # Rebuild combinations with min_max set for B parameter
        new_combinations = []
        for combo in combinations:
            new_combo = {}
            for key, value in combo.items():
                # Check if this is the second parameter (divisor)
                if key == divisor_name:
                    # Create a new InputShapeConstraint with min_max set
                    new_constraint = InputShapeConstraint(value.shape, min_max=(2, 3))
                    new_combo[key] = new_constraint
                else:
                    new_combo[key] = value
            new_combinations.append(new_combo)

        return new_combinations


@register_runtime_checker_op
class PowInputGenerator(BinaryInputGenerator):
    """Input generator for Pow operator."""

    op_name = "Pow"

    def derive_properties(self, properties):
        """Derive properties including broadcasting information."""
        item = super().derive_properties(properties)
        return self._derive_broadcasting_properties(item)


# ===== Logical Operators =====


@register_runtime_checker_op
class AndInputGenerator(BinaryInputGenerator):
    """Input generator for And operator."""

    op_name = "And"


@register_runtime_checker_op
class OrInputGenerator(BinaryInputGenerator):
    """Input generator for Or operator."""

    op_name = "Or"


# ===== Bitwise Operators =====


@register_runtime_checker_op
class BitwiseAndInputGenerator(BinaryInputGenerator):
    """Input generator for BitwiseAnd operator."""

    op_name = "BitwiseAnd"


@register_runtime_checker_op
class BitwiseOrInputGenerator(BinaryInputGenerator):
    """Input generator for BitwiseOr operator."""

    op_name = "BitwiseOr"


@register_runtime_checker_op
class BitwiseXorInputGenerator(BinaryInputGenerator):
    """Input generator for BitwiseXor operator."""

    op_name = "BitwiseXor"


# ===== Comparison Operators =====


class ComparisonInputGenerator(BinaryInputGenerator):
    """Base input generator for comparison operators (Equal, Greater, etc.).

    Comparison operators have the same input characteristics as other binary
    operators, but we can add common properties related to comparison operations
    here if needed in the future.
    """

    def get_qdq_config(self):
        """Return QDQ configuration for comparison operator inputs."""
        return {
            self.op_input_names[0]: QDQParameterConfig(
                support_activation=True, support_weight=True
            ),
            self.op_input_names[1]: QDQParameterConfig(
                support_activation=True, support_weight=True
            ),
            "C": QDQParameterConfig(support_non_qdq=True),
        }


@register_runtime_checker_op
class EqualInputGenerator(ComparisonInputGenerator):
    """Input generator for Equal operator."""

    op_name = "Equal"


@register_runtime_checker_op
class GreaterInputGenerator(ComparisonInputGenerator):
    """Input generator for Greater operator."""

    op_name = "Greater"


@register_runtime_checker_op
class GreaterOrEqualInputGenerator(ComparisonInputGenerator):
    """Input generator for GreaterOrEqual operator."""

    op_name = "GreaterOrEqual"


@register_runtime_checker_op
class LessInputGenerator(ComparisonInputGenerator):
    """Input generator for Less operator."""

    op_name = "Less"


@register_runtime_checker_op
class LessOrEqualInputGenerator(ComparisonInputGenerator):
    """Input generator for LessOrEqual operator."""

    op_name = "LessOrEqual"


# ===== Activation Operators =====


@register_runtime_checker_op
class PReluInputGenerator(BinaryInputGenerator):
    """Input generator for PRelu operator.

    Note: PRelu uses unidirectional broadcasting (slope broadcasts to X),
    but the test cases from BinaryInputGenerator still cover this properly
    since they test both A->B and B->A broadcasting patterns.
    """

    op_name = "PRelu"

    def derive_properties(self, properties: dict) -> dict:
        """Derive properties including broadcasting information."""
        item = super().derive_properties(properties)
        return self._derive_broadcasting_properties(item)
