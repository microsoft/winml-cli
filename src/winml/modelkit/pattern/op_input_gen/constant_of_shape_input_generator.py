# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto

from ...onnx import SupportedONNXType
from .op_input_gen import (
    InputConstraint,
    InputValueConstraint,
    OpInputGenerator,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class ConstantOfShapeInputGenerator(OpInputGenerator):
    """Input generator for ConstantOfShape operator.

    ConstantOfShape documentation:
    - Input: input (1D tensor) - Shape of the expected output tensor.
    - Attribute: value (TensorProto, default 0.0 float32) - The value of the output elements.

    Coverage strategy:
    - Target shapes: 1D through 5D
    - Values: float32 (default), int64, bool
    """

    op_name = "ConstantOfShape"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute combinations."""
        # Create TensorProto for different values

        return {
            "value": [
                onnx.helper.make_tensor(
                    name="value",
                    data_type=SupportedONNXType.from_onnx_type(x).tensor_proto_type,
                    dims=[1],
                    vals=[2],
                )
                for x in self.onnx_types_to_check
            ]
        }

    def get_input_and_infinite_attribute_combinations(self) -> list[dict[str, InputConstraint]]:
        """Return input combinations for ConstantOfShape."""
        combinations = []

        # We want to test creating tensors of various shapes.
        # Common shapes from 1D to 5D
        target_shapes = [
            (),  # Scalar
            (5,),  # 1D
            (3, 4),  # 2D
            (2, 3, 4),  # 3D
            (2, 2, 3, 3),  # 4D
            (1, 2, 2, 3, 3),  # 5D
            (3, 5, 2, 1, 2, 3),  # 6D
        ]

        for shape in target_shapes:
            # The input to ConstantOfShape is a 1D tensor containing the shape dimensions
            shape_tensor = np.array(shape, dtype=np.int64)
            combinations.append({"input": InputValueConstraint(shape_tensor)})

        return combinations

    def infer_output_types(
        self, kwargs: dict[str, Any], tags: dict[str, Any], required_outputs_only: bool = True
    ) -> list[str]:
        """Infer output types based on 'value' attribute."""
        value_attr = kwargs.get("value")

        # Determine logic T2 type based on 'value' attribute
        t2_proto_type = TensorProto.FLOAT if value_attr is None else value_attr.data_type

        # Convert to supported ONNX type string
        supported_type = SupportedONNXType.from_tensor_proto_type(t2_proto_type)

        return [supported_type.annotation]

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties."""
        properties["input_dim"] = len(properties["input_value"])

        for t in properties["attr_value"]:
            if t[0] == "dataType":
                properties["value_type"] = SupportedONNXType.from_tensor_proto_type(t[1]).annotation

        return properties

    def get_infinite_property_names(self) -> list[str]:
        """Return names of infinite properties."""
        return ["input_value"]
