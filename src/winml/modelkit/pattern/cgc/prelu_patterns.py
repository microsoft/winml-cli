# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""PRelu decomposition for the missing IX ONNX legalization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from onnx import ModelProto, helper
from onnx.defs import get_schema

from ...onnx import ONNXDomain, SupportedONNXType
from .. import InputInfo, PatternMatchResult, make_single_op_pattern
from .utils import _static_tensor


if TYPE_CHECKING:
    from .. import SkeletonMatchResult


_PRELU_SCHEMA, _SinglePReluPattern = make_single_op_pattern(get_schema("PRelu", 9))
_RELU_TYPES = set(get_schema("Relu", 6).type_constraints[0].allowed_type_strs)


class PReluWithFiniteSlopePattern(_SinglePReluPattern):  # type: ignore[misc, valid-type]
    """Match floating-point PRelu with a finite, non-overridable constant slope."""

    def check_skeleton_result(
        self,
        skeleton_match_result: SkeletonMatchResult,
    ) -> PatternMatchResult | None:
        """Preserve broadcasting and reject slopes that would introduce NaNs."""
        node = skeleton_match_result.matched_nodes[0]
        matcher = skeleton_match_result.matcher
        if (
            len(node.input) != 2
            or len(node.output) != 1
            or matcher.domain_versions.get(ONNXDomain.AI_ONNX, 0) < 7
        ):
            return None
        data_name, slope_name = node.input
        data_type = matcher.get_tensor_type_str(data_name)
        if (
            data_type is None
            or data_type not in _RELU_TYPES
            or matcher.get_tensor_type_str(slope_name) != data_type
        ):
            return None
        slope = _static_tensor(slope_name, matcher)
        # With non-finite slopes, even x > 0 would evaluate slope * 0 to NaN.
        if slope is None or not np.all(np.isfinite(slope)):
            return None
        return PatternMatchResult(
            skeleton_match_result=skeleton_match_result,
            schema_input_to_value={"X": data_name, "slope": slope_name},
            schema_output_to_value={"Y": node.output[0]},
            type_param_to_type={"T": data_type},
            attributes={
                "_data_shape": matcher.get_tensor_shape(data_name),
                "_slope_shape": slope.shape,
                "_ir_version": matcher.model.ir_version,
            },
            input_infos={
                "X": InputInfo(name="X"),
                "slope": InputInfo(name="slope"),
            },
        )


class ExpandedPReluPattern(_SinglePReluPattern):  # type: ignore[misc, valid-type]
    """Generate Relu(x) - slope * Relu(-x) without folding or copying the slope."""

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
    ) -> ModelProto:
        """Build the floating-point decomposition with the original tensor boundaries."""
        del inputs, is_constant_map
        if input_names is None or len(input_names) != 2:
            raise ValueError("PRelu rewrite requires data and slope names")
        if output_names is None or len(output_names) != 1:
            raise ValueError("PRelu rewrite requires one output name")
        data_name, slope_name = input_names
        positive = f"{prefix}positive"
        negated = f"{prefix}negated"
        negative = f"{prefix}negative"
        scaled = f"{prefix}scaled"
        nodes = [
            helper.make_node("Relu", [data_name], [positive], name=f"{prefix}PositiveRelu"),
            helper.make_node("Neg", [data_name], [negated], name=f"{prefix}Neg"),
            helper.make_node("Relu", [negated], [negative], name=f"{prefix}NegativeRelu"),
            helper.make_node("Mul", [slope_name, negative], [scaled], name=f"{prefix}Mul"),
            helper.make_node("Sub", [positive, scaled], output_names, name=f"{prefix}Sub"),
        ]
        element_type = SupportedONNXType.from_onnx_type(output_dtypes[0]).tensor_proto_type
        graph = helper.make_graph(
            nodes,
            f"{prefix}ExpandedPRelu",
            [
                helper.make_tensor_value_info(data_name, element_type, attributes["_data_shape"]),
                helper.make_tensor_value_info(slope_name, element_type, attributes["_slope_shape"]),
            ],
            [
                helper.make_tensor_value_info(
                    output_names[0],
                    element_type,
                    attributes["_data_shape"],
                )
            ],
        )
        return helper.make_model(
            graph,
            producer_name="winmlcli-pattern-generator",
            opset_imports=[
                helper.make_opsetid(domain.schema_domain, version)
                for domain, version in domain_versions.items()
            ],
            ir_version=attributes["_ir_version"],
        )


__all__ = ["ExpandedPReluPattern", "PReluWithFiniteSlopePattern"]
