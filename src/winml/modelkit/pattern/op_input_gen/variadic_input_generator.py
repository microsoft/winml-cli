"""Input generators for variadic operators (Concat, Sum, Max, Min, Mean)."""

from typing import Any

import numpy as np

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    OpInputGenerator,
    VariadicInputConstraint,
    QDQParameterConfig,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class ConcatInputGenerator(OpInputGenerator):
    """Input generator for Concat operator.

    Concat signature:
    - Inputs: *inputs (variadic) - List of tensors to concatenate
    - Attributes: axis - Which axis to concatenate on

    All input tensors must have the same shape, except for the dimension
    size of the axis to concatenate on.
    """

    op_name = "Concat"

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Return finite attribute sets for Concat.

        Since we specify axis directly in each input combination,
        return empty dict to avoid cross-product iteration.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for Concat.

        Test cases systematically cover:
        - Different tensor ranks (1D through 6D)
        - For each dimension: 3 test cases (axis=0, axis=-1, axis=middle)
        - All cases use exactly 2 input tensors
        - Shapes differ only on the concatenation axis
        """
        combinations = []

        # 1D tensors - only axis=0 (same as axis=-1)
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((3,)),
                        InputShapeConstraint((5,)),
                    ]
                ),
                "axis": 0,
            }
        )

        # 2D tensors - axis=0, axis=-1
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 4)),
                        InputShapeConstraint((3, 4)),
                    ]
                ),
                "axis": 0,
            }
        )
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((3, 2)),
                        InputShapeConstraint((3, 5)),
                    ]
                ),
                "axis": -1,
            }
        )

        # 3D tensors - axis=0, axis=-1, axis=1 (middle)
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 3, 4)),
                        InputShapeConstraint((5, 3, 4)),
                    ]
                ),
                "axis": 0,
            }
        )
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 3, 4)),
                        InputShapeConstraint((2, 3, 6)),
                    ]
                ),
                "axis": -1,
            }
        )
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 3, 4)),
                        InputShapeConstraint((2, 6, 4)),
                    ]
                ),
                "axis": 1,
            }
        )

        # 4D tensors - axis=0, axis=-1, axis=2 (middle)
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 3, 4, 4)),
                        InputShapeConstraint((5, 3, 4, 4)),
                    ]
                ),
                "axis": 0,
            }
        )
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 3, 4, 4)),
                        InputShapeConstraint((2, 3, 4, 6)),
                    ]
                ),
                "axis": -1,
            }
        )
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 3, 4, 4)),
                        InputShapeConstraint((2, 3, 6, 4)),
                    ]
                ),
                "axis": 2,
            }
        )

        # 5D tensors - axis=0, axis=-1, axis=2 (middle)
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 3, 2, 3, 3)),
                        InputShapeConstraint((5, 3, 2, 3, 3)),
                    ]
                ),
                "axis": 0,
            }
        )
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 3, 2, 3, 3)),
                        InputShapeConstraint((2, 3, 2, 3, 6)),
                    ]
                ),
                "axis": -1,
            }
        )
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 3, 2, 3, 3)),
                        InputShapeConstraint((2, 3, 6, 3, 3)),
                    ]
                ),
                "axis": 2,
            }
        )

        # 6D tensors - axis=0, axis=-1, axis=3 (middle)
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                        InputShapeConstraint((5, 2, 2, 2, 2, 3)),
                    ]
                ),
                "axis": 0,
            }
        )
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                        InputShapeConstraint((2, 2, 2, 2, 2, 6)),
                    ]
                ),
                "axis": -1,
            }
        )
        combinations.append(
            {
                "inputs": VariadicInputConstraint(
                    [
                        InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                        InputShapeConstraint((2, 2, 2, 5, 2, 3)),
                    ]
                ),
                "axis": 3,
            }
        )

        return combinations

    def derive_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Derive additional properties for Concat operator testing.

        Args:
            properties: Base properties from parent class containing:
                - inputs_shape: tuple of shapes for each input tensor
                - attr_axis: axis value for concatenation

        Returns:
            Updated properties with Concat-specific derived values:
                - num_inputs: number of input tensors
                - input_ndim: number of dimensions (rank) of input tensors
                - axis_normalized: axis converted to positive index
        """
        item = properties.copy()

        # Get the shape tuple for variadic inputs
        inputs_shape = item["inputs_shape"]
        inputs_value = item["inputs_value"]
        axis = item["attr_axis"]

        # Not derive number of inputs for now as concat op support multiple inputs. Will adjust later if needed.
        # item["num_inputs"] = len(inputs_shape)

        if inputs_shape[0] is not None:
            item["inputs_dim"] = len(inputs_shape[0])
        else:
            array = np.array(inputs_value[0])
            item["inputs_dim"] = array.ndim

        normalized_axis = axis if axis >= 0 else item["inputs_dim"] + axis
        item["first_axis"] = normalized_axis == 0
        item["last_axis"] = normalized_axis == item["inputs_dim"] - 1

        return item

    def get_infinite_property_names(self) -> list[str]:
        """Returns names of infinite properties for Concat operator.

        Infinite properties are those with unbounded value sets:
        - inputs_shape: shapes of input tensors (unbounded combinations)

        The axis attribute is finite (limited by tensor rank) and already
        specified in each input combination.
        """
        return ["inputs_shape", "inputs_value", "attr_axis", "inputs_is_constant"]

    def get_qdq_config(self):
        return {
            "inputs": QDQParameterConfig(support_activation=True),
        }
