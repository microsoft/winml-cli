# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unsqueeze-Cast pattern for ONNX models.

This module provides a pattern for matching ``Unsqueeze -> Cast(to=FLOAT)``
subgraphs.  The exemplar case lives in google-t5/t5-small.onnx where
``/model/decoder/Unsqueeze_1`` is followed by ``/model/decoder/Cast_1`` to
promote an integer attention-mask tensor to float32 before being added to
attention scores.
"""

from typing import Any

import numpy as np
from onnx import TensorProto
from onnx.defs import OpSchema

from ..onnx import ONNXDomain
from .base import (
    Pattern,
    PatternInputGenerator,
    PatternMismatchedError,
    PatternSchema,
    Skeleton,
    register_pattern_input_generator,
)
from .match import SkeletonMatchResult
from .op_input_gen import InputShapeConstraint


# Cast `to` value this pattern is specialised for: TensorProto.FLOAT == 1.
_CAST_TO_FLOAT32 = int(TensorProto.FLOAT)


_UNSQUEEZE_CAST_SCHEMA = PatternSchema(
    name="UnsqueezeCastPattern",
    doc=(
        "Unsqueeze followed by Cast(to=FLOAT) pattern.\n"
        "Computes: output = Cast(Unsqueeze(data, axes), to=tensor(float))\n"
        "\n"
        "Attributes:\n"
        "- axes: int64 axes input to the Unsqueeze node (required to be constant).\n"
        "- to: Target dtype enum for the Cast node; constrained to "
        "TensorProto.FLOAT (1).\n"
    ),
    type_constraints=[
        OpSchema.TypeConstraintParam(
            type_param_str="T1",
            allowed_type_strs=[
                "tensor(float16)",
                "tensor(float)",
                "tensor(double)",
                "tensor(uint8)",
                "tensor(uint16)",
                "tensor(uint32)",
                "tensor(uint64)",
                "tensor(int8)",
                "tensor(int16)",
                "tensor(int32)",
                "tensor(int64)",
                "tensor(bfloat16)",
                "tensor(bool)",
            ],
            description="Constrain input type to all numeric tensor types.",
        ),
        OpSchema.TypeConstraintParam(
            type_param_str="T2",
            allowed_type_strs=["tensor(float)"],
            description="Constrain output type to tensor(float).",
        ),
    ],
    inputs=[
        OpSchema.FormalParameter(
            name="data",
            type_str="T1",
            description="Input tensor to be unsqueezed and cast to float.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            type_str="T2",
            description="Cast output tensor with an extra axis inserted.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        )
    ],
    attributes={
        "axes": OpSchema.Attribute(
            name="axes",
            description="Axes argument of the Unsqueeze node (constant int64 tensor).",
            type=OpSchema.AttrType.INTS,
            required=True,
        ),
        "to": OpSchema.Attribute(
            name="to",
            description="Cast `to` enum; constrained to TensorProto.FLOAT (1).",
            type=OpSchema.AttrType.INT,
            required=True,
        ),
    },
)


class UnsqueezeCastPattern(Pattern):
    """Pattern for Unsqueeze followed by Cast(to=FLOAT).

    Node topology:
    - Node 0 (Unsqueeze): Unsqueeze(data, axes)
    - Node 1 (Cast): Cast(unsqueeze_output, to=FLOAT)

    The ``axes`` input of the Unsqueeze node must be a constant (initializer
    or Constant-node output); otherwise the match is rejected.  The Cast
    ``to`` attribute is constrained to ``TensorProto.FLOAT`` (1).
    """

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for the UnsqueezeCast pattern."""
        node_op_types = ["Unsqueeze", "Cast"]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        edges = [
            (-1, 0, 0, 0),  # input data -> Unsqueeze[0]
            (0, 0, 1, 0),  # Unsqueeze output -> Cast[0]
        ]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=[1],
            n_inputs=1,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants for axes and attribute constraint for Cast.to."""
        internal_constants: list[tuple[int, int, np.ndarray]] = [
            (0, 1, np.array(attributes["axes"], dtype=np.int64)),
        ]
        internal_attributes: dict[tuple[int, str], Any] = {
            (1, "to"): _CAST_TO_FLOAT32,
        }
        return internal_constants, internal_attributes

    def _infer_schema_attributes(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> dict[str, Any]:
        """Infer ``axes`` (from Unsqueeze) and ``to`` (from Cast) attributes."""
        attributes: dict[str, Any] = {}
        matcher = skeleton_match_result.matcher
        matched_nodes = skeleton_match_result.matched_nodes

        unsqueeze_node = matched_nodes[0]
        if len(unsqueeze_node.input) <= 1:
            raise PatternMismatchedError("Unsqueeze node missing axes input")
        axes_input_name = unsqueeze_node.input[1]
        if axes_input_name not in matcher.tensor_values:
            raise PatternMismatchedError(
                f"Unsqueeze axes input '{axes_input_name}' is not a constant"
            )
        attributes["axes"] = tuple(matcher.tensor_values[axes_input_name].tolist())

        cast_node = matched_nodes[1]
        to_found = False
        for attr in cast_node.attribute:
            if attr.name == "to":
                attributes["to"] = int(attr.i)
                to_found = True
                break
        if not to_found:
            raise PatternMismatchedError("Cast node missing 'to' attribute")

        return attributes

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for the UnsqueezeCast pattern."""
        return _UNSQUEEZE_CAST_SCHEMA


@register_pattern_input_generator
class UnsqueezeCastPatternInputGenerator(PatternInputGenerator):
    """PatternInputGenerator for UnsqueezeCastPattern."""

    pattern = UnsqueezeCastPattern()
    registration_name = "UnsqueezeCastPattern"

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Return finite attribute sets (empty for this pattern)."""
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, Any]]:
        """Generate input/attribute combinations for testing."""
        return [
            {
                "data": InputShapeConstraint((2, 3)),
                "axes": (1,),
                "to": _CAST_TO_FLOAT32,
            },
            {
                "data": InputShapeConstraint((4, 5, 6)),
                "axes": (0,),
                "to": _CAST_TO_FLOAT32,
            },
        ]

    def derive_properties(self, properties: dict) -> dict:
        """Add convenience properties for parameterised testing."""
        item = properties.copy()
        item["axes_dim"] = len(item["attr_axes"])
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values."""
        return ["attr_axes", "data_shape"]
