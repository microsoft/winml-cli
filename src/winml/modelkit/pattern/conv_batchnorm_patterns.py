# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Patterns for folding inference BatchNormalization after Conv and Add."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.defs import OpSchema

from ..onnx import ONNXDomain, SupportedONNXType
from .base import Pattern, PatternMismatchedError, PatternSchema, Skeleton


if TYPE_CHECKING:
    from .match import PatternMatchResult, SkeletonMatchResult


CONV_ADD_BATCHNORM_SCHEMA = PatternSchema(
    name="ConvAddBatchNormalizationPattern",
    doc=(
        "Inference BatchNormalization applied to the sum of a Conv output and "
        "a static broadcast tensor."
    ),
    type_constraints=[
        OpSchema.TypeConstraintParam(
            type_param_str="T",
            allowed_type_strs=[
                "tensor(float16)",
                "tensor(float)",
                "tensor(double)",
            ],
            description="Constrain inputs and output to floating-point tensors.",
        )
    ],
    inputs=[
        OpSchema.FormalParameter(name, "T", description)
        for name, description in (
            ("X", "Conv input."),
            ("W", "Static Conv weights."),
            ("A", "Static tensor added to the Conv output."),
            ("scale", "BatchNormalization scale."),
            ("B", "BatchNormalization bias."),
            ("mean", "BatchNormalization input mean."),
            ("var", "BatchNormalization input variance."),
        )
    ],
    outputs=[
        OpSchema.FormalParameter(
            "Y",
            "T",
            "Folded Conv and Add output.",
        )
    ],
)


def _scale_broadcast_tensor(
    values: np.ndarray,
    output_shape: tuple[int, ...],
    gamma: np.ndarray,
) -> np.ndarray | None:
    """Scale a broadcastable Add operand along the Conv channel axis."""
    if values.ndim > len(output_shape):
        return None
    padded_shape = (1,) * (len(output_shape) - values.ndim) + tuple(values.shape)
    if any(dimension not in (1, output_shape[axis]) for axis, dimension in enumerate(padded_shape)):
        return None
    if padded_shape[1] not in (1, output_shape[1]):
        return None

    expanded_shape = list(padded_shape)
    expanded_shape[1] = output_shape[1]
    try:
        broadcast_values = np.broadcast_to(values.reshape(padded_shape), tuple(expanded_shape))
    except ValueError:
        return None
    factors = gamma.reshape((1, len(gamma)) + (1,) * (len(output_shape) - 2))
    return np.asarray(broadcast_values * factors, dtype=values.dtype)


class _ConvAddBatchNormalizationPatternBase(Pattern):
    """Shared validation for the two commutative Add input orders."""

    static_input_index: int

    @property
    def pattern_id(self) -> str:
        """Return the source pattern ID used by rewrite registration."""
        return "SUBGRAPH/Conv-Add-Batch-NormalizationPattern"

    def get_schema(self) -> PatternSchema:
        return CONV_ADD_BATCHNORM_SCHEMA

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        return [], {}

    def get_skeleton(self) -> Skeleton:
        conv_add_slot = 1 - self.static_input_index
        return Skeleton(
            node_op_types=["Conv", "Add", "BatchNormalization"],
            node_domains=[ONNXDomain.AI_ONNX] * 3,
            edges=[
                (-1, 0, 0, 0),
                (-2, 0, 0, 1),
                (0, 0, 1, conv_add_slot),
                (-3, 0, 1, self.static_input_index),
                (1, 0, 2, 0),
                (-4, 0, 2, 1),
                (-5, 0, 2, 2),
                (-6, 0, 2, 3),
                (-7, 0, 2, 4),
            ],
            exit_nodes=[2],
            n_inputs=7,
        )

    def check_skeleton_result(
        self,
        skeleton_match_result: SkeletonMatchResult,
    ) -> PatternMatchResult | None:
        result = super().check_skeleton_result(skeleton_match_result)
        if result is None:
            return None

        conv, add, batch_norm = skeleton_match_result.matched_nodes
        if (
            len(conv.input) not in (2, 3)
            or len(conv.output) != 1
            or len(add.input) != 2
            or len(add.output) != 1
            or len(batch_norm.input) != 5
            or len(batch_norm.output) != 1
        ):
            return None

        required_constants = ("W", "A", "scale", "B", "mean", "var")
        if any(
            name not in result.input_infos
            or not result.input_infos[name].is_constant
            or result.input_infos[name].value is None
            for name in required_constants
        ):
            return None

        values = {name: np.asarray(result.input_infos[name].value) for name in required_constants}
        weight = values["W"]
        add_tensor = values["A"]
        scale = values["scale"]
        beta = values["B"]
        mean = values["mean"]
        variance = values["var"]
        if weight.ndim < 1 or not np.issubdtype(weight.dtype, np.floating):
            return None

        matcher = skeleton_match_result.matcher
        output_shape_value = matcher.get_tensor_shape(conv.output[0])
        if (
            output_shape_value is None
            or len(output_shape_value) < 2
            or any(not isinstance(dimension, int) for dimension in output_shape_value)
        ):
            return None
        output_shape = tuple(int(dimension) for dimension in output_shape_value)
        channels = output_shape[1]
        if weight.shape[0] != channels:
            return None

        parameters = (scale, beta, mean, variance)
        if any(value.ndim != 1 or len(value) != channels for value in parameters):
            return None
        if any(value.dtype != weight.dtype for value in (*parameters, add_tensor)):
            return None

        bias = np.zeros(channels, dtype=weight.dtype)
        if len(conv.input) == 3 and conv.input[2]:
            bias_value = matcher.tensor_values.get(conv.input[2])
            if bias_value is None or conv.input[2] not in matcher.constant_and_initializer_names:
                return None
            bias = np.asarray(bias_value)
            if bias.ndim != 1 or len(bias) != channels or bias.dtype != weight.dtype:
                return None

        conv_attributes = {
            attribute.name: helper.get_attribute_value(attribute) for attribute in conv.attribute
        }
        batch_norm_attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in batch_norm.attribute
        }
        try:
            if int(batch_norm_attributes.get("training_mode", 0)) != 0:
                return None
            epsilon = float(batch_norm_attributes.get("epsilon", 1e-5))
        except (TypeError, ValueError):
            return None

        denominator = variance + epsilon
        if epsilon < 0 or np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
            return None
        gamma = scale / np.sqrt(denominator)
        if not np.all(np.isfinite(gamma)):
            return None

        scaled_add = _scale_broadcast_tensor(add_tensor, output_shape, gamma)
        if scaled_add is None:
            return None
        scaled_weight = weight * gamma.reshape((channels,) + (1,) * (weight.ndim - 1))
        folded_bias = bias * gamma + beta - gamma * mean

        result.attributes.update(
            {
                "conv_attributes": conv_attributes,
                "static_input_index": self.static_input_index,
                "folded_weight": np.asarray(scaled_weight, dtype=weight.dtype),
                "folded_bias": np.asarray(folded_bias, dtype=weight.dtype),
                "scaled_add": scaled_add,
            }
        )
        return result


class ConvAddBatchNormalizationPattern(_ConvAddBatchNormalizationPatternBase):
    """Match ``Add(Conv(X, W[, bias]), A) -> BatchNormalization``."""

    static_input_index = 1


class AddConvBatchNormalizationPattern(_ConvAddBatchNormalizationPatternBase):
    """Match ``Add(A, Conv(X, W[, bias])) -> BatchNormalization``."""

    static_input_index = 0


class FoldedConvAddPattern(Pattern):
    """Generate a Conv and Add with BatchNormalization folded into constants."""

    @property
    def pattern_id(self) -> str:
        """Return the folded target pattern ID used by rewrite registration."""
        return "SUBGRAPH/FoldedConvAddPattern"

    def get_schema(self) -> PatternSchema:
        """Return the schema shared with the source patterns."""
        return CONV_ADD_BATCHNORM_SCHEMA

    def get_skeleton(self) -> Skeleton:
        """Return the folded Conv-to-Add topology."""
        return Skeleton(
            node_op_types=["Conv", "Add"],
            node_domains=[ONNXDomain.AI_ONNX] * 2,
            edges=[
                (-1, 0, 0, 0),
                (-2, 0, 0, 1),
                (0, 0, 1, 0),
                (-3, 0, 1, 1),
            ],
            exit_nodes=[1],
            n_inputs=7,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return no static constraints because generation is customized."""
        return [], {}

    def get_onnx_model(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        output_dtypes: list[str],
        domain_versions: dict[ONNXDomain, int],
        prefix: str = "",
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> onnx.ModelProto:
        """Build the folded Conv and Add subgraph from captured constants."""
        required = (
            "conv_attributes",
            "static_input_index",
            "folded_weight",
            "folded_bias",
            "scaled_add",
        )
        if any(name not in attributes for name in required):
            raise PatternMismatchedError("Missing folded Conv/Add attributes")
        if input_names is None or output_names is None:
            raise PatternMismatchedError("Input and output names are required")

        weight_name = f"{prefix}weight"
        bias_name = f"{prefix}bias"
        add_name = f"{prefix}add"
        conv_output = f"{prefix}conv_output"
        conv = helper.make_node(
            "Conv",
            [input_names[0], weight_name, bias_name],
            [conv_output],
            name=f"{prefix}Conv",
            **attributes["conv_attributes"],
        )
        add_inputs = [conv_output, add_name]
        if attributes["static_input_index"] == 0:
            add_inputs.reverse()
        add = helper.make_node(
            "Add",
            add_inputs,
            [output_names[0]],
            name=f"{prefix}Add",
        )

        input_type = SupportedONNXType.from_np_type(inputs["X"].dtype).tensor_proto_type
        output_type = SupportedONNXType.from_onnx_type(output_dtypes[0]).tensor_proto_type
        graph = helper.make_graph(
            [conv, add],
            f"{prefix}FoldedConvAdd",
            [helper.make_tensor_value_info(input_names[0], input_type, inputs["X"].shape)],
            [helper.make_tensor_value_info(output_names[0], output_type, None)],
            initializer=[
                numpy_helper.from_array(attributes["folded_weight"], weight_name),
                numpy_helper.from_array(attributes["folded_bias"], bias_name),
                numpy_helper.from_array(attributes["scaled_add"], add_name),
            ],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid(domain.schema_domain, version)
                for domain, version in domain_versions.items()
            ],
        )
        model.ir_version = 11
        return model
