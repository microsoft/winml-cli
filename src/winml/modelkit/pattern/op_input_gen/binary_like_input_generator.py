# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generators for binary-like ONNX operators.

Binary-like operators have two primary input tensors but may have additional
attributes or optional inputs. This module provides input generators for operators
that can reuse the BinaryInputGenerator base shapes while adding operator-specific
attributes or input handling.

Key principle: NEVER redefine input shapes - reuse parent class shapes combined
with additional attributes or inputs.
"""

from typing import Any, cast

import numpy as np

from .binary_input_generator import BinaryInputGenerator
from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    InputValueConstraint,
    QDQParameterConfig,
    register_runtime_checker_op,
)


# ============================================================================
# Category 1: Binary inputs + string attributes
# ============================================================================


@register_runtime_checker_op
class BitShiftInputGenerator(BinaryInputGenerator):
    """Input generator for BitShift operator.

    Signature: BitShift(X, Y, *, direction) -> output
    Two inputs with string attribute for shift direction.

    Broadcasting: Supports multidirectional (Numpy-style) broadcasting.
    The operator inherits all broadcasting test cases from BinaryInputGenerator.
    """

    op_name = "BitShift"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return direction values to test.

        direction can be "LEFT" or "RIGHT" for bit shifting.
        """
        return {"direction": ["LEFT", "RIGHT"]}

    def derive_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Derive properties including broadcasting information."""
        item = super().derive_properties(properties)
        return self._derive_broadcasting_properties(item)


# ============================================================================
# Category 2: Binary inputs + int attributes
# ============================================================================


@register_runtime_checker_op
class ModInputGenerator(BinaryInputGenerator):
    """Input generator for Mod (modulo) operator.

    Signature: Mod(A, B, *, fmod=0) -> output
    Two inputs with int attribute for modulo semantics.

    Broadcasting: Supports multidirectional (Numpy-style) broadcasting.
    The operator inherits all broadcasting test cases from BinaryInputGenerator.

    fmod values:
    - 0 (default): Python % semantics, integer types, result has sign of divisor
    - 1: C fmod semantics, floating point types, result has sign of dividend

    CRITICAL: Divisor (B) must never be zero to avoid divide-by-zero errors.
    """

    op_name = "Mod"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return fmod values to test.

        fmod controls the modulo semantics:
        - 0: Python-style % operator (integer mod)
        - 1: C-style fmod function (floating point mod)
        """
        return {"fmod": [0, 1]}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, object]]:
        """Return input combinations for Mod operator.

        Strategy:
        1. Get A and B combinations from parent (all broadcasting patterns)
        2. Override B (divisor) with min_max=(1, 10) to avoid divide-by-zero

        This ensures:
        - All broadcasting patterns are tested
        - Divisor is never zero (sampled from 1 to 10)
        - All dimensional variations (1D-6D) are covered
        """
        # Get parameter names from the operator schema
        a_name = self.op_input_names[0]  # Usually 'A'
        b_name = self.op_input_names[1]  # Usually 'B' (divisor)

        # Get parent's A/B combinations (all broadcasting patterns)
        parent_combinations = super().get_input_and_infinite_attribute_combinations()

        # Extract parameter names from first combination
        if not parent_combinations:
            return []

        first_combo = parent_combinations[0]
        parent_param_names = list(first_combo.keys())
        parent_first = parent_param_names[0] if parent_param_names else "A"
        parent_second = parent_param_names[1] if len(parent_param_names) > 1 else "B"

        combinations = []

        for parent_combo in parent_combinations:
            # Get A constraint from parent
            a_constraint = parent_combo[parent_first]

            # Get B shape from parent, but override with min_max to ensure non-zero divisor
            # Parent binary combos hold InputShapeConstraint values for these inputs.
            b_shape = cast(InputShapeConstraint, parent_combo[parent_second]).shape

            # Create non-zero divisor: sample values from 1 to 10
            # This avoids divide-by-zero while providing varied test data
            b_constraint = InputShapeConstraint(b_shape, min_max=(1, 10))

            combinations.append(
                {
                    a_name: a_constraint,
                    b_name: b_constraint,
                }
            )

        return combinations


# ============================================================================
# Category 3: Ternary operators (3 inputs with broadcasting)
# ============================================================================


@register_runtime_checker_op
class WhereInputGenerator(BinaryInputGenerator):
    """Input generator for Where operator (ternary conditional operator).

    Signature: Where(condition, X, Y) -> output
    Three inputs: condition (bool), X (data), Y (data).
    No attributes.

    Broadcasting: All three inputs support multidirectional broadcasting.
    - condition broadcasts with X and Y
    - X and Y broadcast with each other

    Strategy:
    - Reuse parent BinaryInputGenerator shapes for X and Y
    - Add condition input that matches or broadcasts with X/Y shapes
    - Condition is boolean type, X and Y are data types
    """

    op_name = "Where"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Where has no attributes."""
        return {}

    # TODO: use InputValueConstraint or InputShapeConstraint for condition input?
    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, object]]:
        """Return input combinations for Where operator.

        Strategy:
        1. Get X and Y combinations from parent (all broadcasting patterns)
        2. For each X/Y pair, add condition input
        3. Condition options:
           - Same shape as X (most common case)
           - Same shape as Y (when X != Y shape, tests asymmetric broadcasting)

        This ensures we test:
        - Condition broadcasting scenarios (all three input broadcasting)
        - X and Y broadcasting scenarios (from parent)
        - All dimensional variations (1D-6D from parent)
        """
        # Get parameter names from the operator schema
        condition_name = self.op_input_names[0]  # 'condition'
        x_name = self.op_input_names[1]  # 'X'
        y_name = self.op_input_names[2]  # 'Y'

        # Get parent's X/Y combinations (all broadcasting patterns)
        parent_combinations = super().get_input_and_infinite_attribute_combinations()

        # For Where, we need to map parent's param names to X/Y
        # Get the first parent combination to determine param names
        if not parent_combinations:
            return []

        # Extract parameter names from first combination
        first_combo = parent_combinations[0]
        parent_param_names = list(first_combo.keys())
        parent_first = parent_param_names[0] if parent_param_names else "A"
        parent_second = parent_param_names[1] if len(parent_param_names) > 1 else "B"

        combinations: list[dict[str, object]] = []

        for parent_combo in parent_combinations:
            # Get X and Y shapes from parent combination
            x_constraint = cast(InputShapeConstraint, parent_combo[parent_first])
            y_constraint = cast(InputShapeConstraint, parent_combo[parent_second])

            x_shape = x_constraint.shape
            y_shape = y_constraint.shape

            # Option 1: Condition has same shape as X (most common)
            condition_x_shape = InputValueConstraint(np.ones(x_shape, dtype=np.bool_))
            combinations.append(
                {
                    condition_name: condition_x_shape,
                    x_name: x_constraint,
                    y_name: y_constraint,
                }
            )

            # Option 2: Condition has same shape as Y (when X != Y, tests asymmetric broadcasting)
            if x_shape != y_shape:
                condition_y_shape = InputValueConstraint(np.ones(y_shape, dtype=np.bool_))
                combinations.append(
                    {
                        condition_name: condition_y_shape,
                        x_name: x_constraint,
                        y_name: y_constraint,
                    }
                )

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for Where operator.

        Adds:
        - condition_dim: dimensionality of condition input
        - x_dim: dimensionality of X input
        - y_dim: dimensionality of Y input
        - x_broadcasting_to_y / y_broadcasting_to_x: X-Y broadcasting
        - condition_broadcasting_to_x / x_broadcasting_to_condition: condition-X broadcasting
        """
        condition_name = self.op_input_names[0]
        x_name = self.op_input_names[1]
        y_name = self.op_input_names[2]

        item = properties.copy()
        condition_value = item.get(f"{condition_name}_value")
        if condition_value is None:
            # Fall back to shape information when value is unknown (non-constant input)
            condition_shape = item.get(f"{condition_name}_shape")
            condition_dim = len(condition_shape) if condition_shape is not None else 0
        else:
            condition_dim = 0 if type(condition_value) is bool else np.array(condition_value).ndim

        item[f"{condition_name}_dim"] = condition_dim
        item[f"{x_name}_dim"] = len(item[f"{x_name}_shape"])
        item[f"{y_name}_dim"] = len(item[f"{y_name}_shape"])

        item = self._derive_broadcasting_properties(item, x_name=x_name, y_name=y_name)

        # Derive condition-to-X broadcasting properties
        if condition_value is not None and type(condition_value) is not bool:
            cond_shape = tuple(np.array(condition_value).shape)
        elif condition_value is not None:
            cond_shape = ()
        else:
            cond_shape = tuple(item.get(f"{condition_name}_shape", ()))

        x_shape = tuple(item[f"{x_name}_shape"])

        # Pad shapes to same length for comparison
        max_len = max(len(cond_shape), len(x_shape))
        padded_cond = (1,) * (max_len - len(cond_shape)) + cond_shape
        padded_x = (1,) * (max_len - len(x_shape)) + x_shape

        condition_broadcasting_to_x = len(cond_shape) < len(x_shape) or any(
            isinstance(c, int) and isinstance(x, int) and c != 0 and c < x
            for c, x in zip(padded_cond, padded_x, strict=False)
        )
        x_broadcasting_to_condition = len(x_shape) < len(cond_shape) or any(
            isinstance(x, int) and isinstance(c, int) and x != 0 and x < c
            for c, x in zip(padded_cond, padded_x, strict=False)
        )

        item["condition_broadcasting_to_x"] = condition_broadcasting_to_x
        item["x_broadcasting_to_condition"] = x_broadcasting_to_condition

        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite value ranges."""
        condition_name = self.op_input_names[0]
        x_name = self.op_input_names[1]
        y_name = self.op_input_names[2]
        return [
            f"{condition_name}_value",
            f"{x_name}_shape",
            f"{y_name}_shape",
        ]

    def get_qdq_config(self) -> dict[str, QDQParameterConfig]:
        """Return QDQ configuration for Where operator inputs."""
        return {
            self.op_input_names[0]: QDQParameterConfig(support_non_qdq=True),
            self.op_input_names[1]: QDQParameterConfig(
                support_activation=True, support_non_qdq=True
            ),
            self.op_input_names[2]: QDQParameterConfig(
                support_activation=True, support_non_qdq=True
            ),
        }
