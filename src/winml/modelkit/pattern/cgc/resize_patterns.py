# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Explicit Resize compatibility rewrites for CGIR."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from onnx import ModelProto, TensorProto, helper
from onnx.defs import OpSchema, get_schema

from ...onnx import ONNXDomain, SupportedONNXType
from .. import (
    InputInfo,
    Pattern,
    PatternMatchResult,
    PatternSchema,
    Skeleton,
    opschema_to_pattern_schema,
)
from ..utils import get_attribute_proto_value


if TYPE_CHECKING:
    from .. import PatternMatcher, SkeletonMatchResult


_ONNX_RESIZE_SCHEMA = get_schema("Resize", 13)
_RESIZE_SCHEMA: PatternSchema = opschema_to_pattern_schema(_ONNX_RESIZE_SCHEMA)
_RESIZE_OMITTED_INPUTS_SCHEMA = PatternSchema(
    name=_RESIZE_SCHEMA.name,
    doc=_RESIZE_SCHEMA.doc,
    inputs=[
        _RESIZE_SCHEMA.inputs[0],
        OpSchema.FormalParameter(
            name="resize_parameter",
            type_str="TResizeParameter",
            description="The effective non-empty scales or sizes input.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
    ],
    outputs=_RESIZE_SCHEMA.outputs,
    type_constraints=[
        *_RESIZE_SCHEMA.type_constraints,
        OpSchema.TypeConstraintParam(
            type_param_str="TResizeParameter",
            allowed_type_strs=["tensor(float)", "tensor(int64)"],
            description="Constrain the effective Resize parameter.",
        ),
    ],
    attributes=_RESIZE_SCHEMA.attributes,
)


def _is_empty_shape(shape: tuple[int | str | None, ...] | None) -> bool:
    return shape is not None and any(dimension == 0 for dimension in shape)


def _is_empty_resize_input(name: str, matcher: PatternMatcher) -> bool:
    """Require static emptiness without relying on an overridable input's default."""
    if not name or not _is_empty_shape(matcher.get_tensor_shape(name)):
        return False
    pending = [name]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if not current or current in visited:
            continue
        visited.add(current)
        producer = matcher.producer_lookup.get(current)
        if producer is None or producer[2] == "GraphInput":
            return False
        node = matcher.node_lookup.get(producer[0])
        if node is not None:
            pending.extend(node.input)
    return True


class _ResizeOptionalInputsPattern(Pattern):
    def get_skeleton(self) -> Skeleton:
        return Skeleton(
            node_op_types=["Resize"],
            node_domains=[ONNXDomain.AI_ONNX],
            edges=[
                (-1, 0, 0, 0),
            ],
            exit_nodes=[0],
            n_inputs=1,
        )

    def get_schema(self) -> PatternSchema:
        return _RESIZE_OMITTED_INPUTS_SCHEMA

    def _infer_schema_attributes(
        self,
        skeleton_match_result: SkeletonMatchResult,
    ) -> dict[str, Any]:
        node = skeleton_match_result.matched_nodes[0]
        return {
            attribute.name: get_attribute_proto_value(
                attribute,
                replace_float_with_dummy=False,
            )
            for attribute in node.attribute
        }

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        return [], {(0, name): value for name, value in attributes.items()}


class ResizeWithEmptyOptionalInputsPattern(_ResizeOptionalInputsPattern):
    """Match Resize inputs represented by named empty tensors."""

    def check_skeleton_result(
        self,
        skeleton_match_result: SkeletonMatchResult,
    ) -> PatternMatchResult | None:
        """Omit statically empty ROI/scales while retaining the effective parameter."""
        node = skeleton_match_result.matched_nodes[0]
        matcher = skeleton_match_result.matcher
        if matcher.domain_versions.get(ONNXDomain.AI_ONNX, 0) < 11:
            return None
        attributes = self._infer_schema_attributes(skeleton_match_result)
        if attributes.get("coordinate_transformation_mode") == "tf_crop_and_resize":
            return None

        roi_name = node.input[1] if len(node.input) > 1 else ""
        scales_name = node.input[2] if len(node.input) > 2 else ""
        sizes_name = node.input[3] if len(node.input) > 3 else ""

        roi_is_empty = _is_empty_resize_input(roi_name, matcher)
        if roi_name and not roi_is_empty:
            return None

        scales_is_empty = _is_empty_resize_input(scales_name, matcher)
        if sizes_name:
            if scales_name and not scales_is_empty:
                return None
            resize_parameter_name = sizes_name
            resize_parameter_slot = 3
        elif scales_name and not scales_is_empty:
            resize_parameter_name = scales_name
            resize_parameter_slot = 2
        else:
            return None

        if not roi_is_empty and not scales_is_empty:
            return None

        skeleton_match_result = replace(
            skeleton_match_result,
            inputs=[node.input[0], resize_parameter_name],
        )
        type_param_to_type = self._infer_type_mapping(skeleton_match_result)
        if "T1" not in type_param_to_type:
            return None

        attributes["_resize_parameter_slot"] = resize_parameter_slot
        attributes["_ir_version"] = matcher.model.ir_version
        return PatternMatchResult(
            skeleton_match_result=skeleton_match_result,
            schema_input_to_value={
                "X": node.input[0],
                "resize_parameter": resize_parameter_name,
            },
            schema_output_to_value={"Y": node.output[0]},
            type_param_to_type=type_param_to_type,
            attributes=attributes,
            input_infos={
                "X": InputInfo(name="X"),
                "resize_parameter": InputInfo(name="resize_parameter"),
            },
        )


class ResizeWithOmittedOptionalInputsPattern(_ResizeOptionalInputsPattern):
    """Generate Resize with empty ROI and scales represented as omitted inputs."""

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
        """Build the equivalent Resize with empty optional inputs omitted."""
        del is_constant_map
        if input_names is None or len(input_names) != 2:
            raise ValueError("Resize rewrite requires X and sizes input names")
        if output_names is None or len(output_names) != 1:
            raise ValueError("Resize rewrite requires one output name")

        node_attributes = dict(attributes)
        ir_version = node_attributes.pop("_ir_version", None)
        resize_parameter_slot = int(node_attributes.pop("_resize_parameter_slot"))
        node_inputs = [input_names[0], "", "", ""]
        node_inputs[resize_parameter_slot] = input_names[1]
        while node_inputs[-1] == "":
            node_inputs.pop()
        node = helper.make_node(
            "Resize",
            node_inputs,
            output_names,
            name=f"{prefix}Resize",
            **node_attributes,
        )

        output_element_type = SupportedONNXType.from_onnx_type(output_dtypes[0]).tensor_proto_type
        parameter_element_type = (
            TensorProto.INT64 if resize_parameter_slot == 3 else TensorProto.FLOAT
        )
        input_specs = [
            (0, 0, output_element_type),
            (resize_parameter_slot, 1, parameter_element_type),
        ]
        graph_inputs = []
        for node_index, schema_index, fallback_element_type in input_specs:
            name = node_inputs[node_index]
            value = inputs.get(_RESIZE_OMITTED_INPUTS_SCHEMA.inputs[schema_index].name)
            element_type = (
                helper.np_dtype_to_tensor_dtype(value.dtype)
                if value is not None
                else fallback_element_type
            )
            shape = list(value.shape) if value is not None else None
            graph_inputs.append(helper.make_tensor_value_info(name, element_type, shape))

        graph = helper.make_graph(
            [node],
            f"{prefix}ResizeWithOmittedOptionalInputs",
            graph_inputs,
            [
                helper.make_tensor_value_info(
                    output_names[0],
                    output_element_type,
                    None,
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
            **({"ir_version": ir_version} if ir_version is not None else {}),
        )


class _ResizeAttributesPattern(_ResizeOptionalInputsPattern):
    """Preserve all input slots while rewriting only selected Resize attributes."""

    def get_schema(self) -> PatternSchema:
        return _RESIZE_SCHEMA

    def check_skeleton_result(
        self,
        skeleton_match_result: SkeletonMatchResult,
    ) -> PatternMatchResult | None:
        node = skeleton_match_result.matched_nodes[0]
        matcher = skeleton_match_result.matcher
        if (
            matcher.domain_versions.get(ONNXDomain.AI_ONNX, 0) < 11
            or not 3 <= len(node.input) <= len(_RESIZE_SCHEMA.inputs)
            or len(node.output) != 1
        ):
            return None
        skeleton_match_result = replace(
            skeleton_match_result,
            inputs=[*node.input, *([""] * (len(_RESIZE_SCHEMA.inputs) - len(node.input)))],
        )
        type_mapping = self._infer_type_mapping(skeleton_match_result)
        if "T1" not in type_mapping:
            return None
        input_types = []
        for name in skeleton_match_result.inputs:
            if not name:
                input_types.append(TensorProto.UNDEFINED)
                continue
            tensor_type = matcher.get_tensor_type_str(name)
            if tensor_type is None:
                return None
            input_types.append(SupportedONNXType.from_onnx_type(tensor_type).tensor_proto_type)
        attributes = self._infer_schema_attributes(skeleton_match_result)
        attributes["_ir_version"] = matcher.model.ir_version
        attributes["_input_element_types"] = input_types
        return PatternMatchResult(
            skeleton_match_result=skeleton_match_result,
            schema_input_to_value={
                parameter.name: name
                for parameter, name in zip(
                    _RESIZE_SCHEMA.inputs,
                    skeleton_match_result.inputs,
                    strict=True,
                )
            },
            schema_output_to_value={"Y": node.output[0]},
            type_param_to_type=type_mapping,
            attributes=attributes,
            # No dummy arrays: only input names and types are needed for an attribute edit.
            input_infos={
                parameter.name: InputInfo(name=parameter.name)
                for parameter in _RESIZE_SCHEMA.inputs
            },
        )

    def _rewrite_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        return attributes

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
        del inputs, is_constant_map
        if input_names is None or len(input_names) != len(_RESIZE_SCHEMA.inputs):
            raise ValueError("Resize attribute rewrite requires all optional input slots")
        if output_names is None or len(output_names) != 1:
            raise ValueError("Resize attribute rewrite requires one output")
        node_attributes = dict(attributes)
        ir_version = node_attributes.pop("_ir_version", None)
        input_types = node_attributes.pop("_input_element_types")
        node_attributes = self._rewrite_attributes(node_attributes)
        node_inputs = list(input_names)
        while node_inputs and not node_inputs[-1]:
            node_inputs.pop()
        output_type = SupportedONNXType.from_onnx_type(output_dtypes[0]).tensor_proto_type
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Resize", node_inputs, output_names, name=f"{prefix}Resize", **node_attributes
                )
            ],
            f"{prefix}ResizeAttributes",
            [
                helper.make_tensor_value_info(name, dtype, None)
                for name, dtype in zip(input_names, input_types, strict=True)
                if name
            ],
            [helper.make_tensor_value_info(output_names[0], output_type, None)],
        )
        return helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid(domain.schema_domain, version)
                for domain, version in domain_versions.items()
            ],
            **({"ir_version": ir_version} if ir_version is not None else {}),
        )


class ResizeWithTfHalfPixelForNNPattern(_ResizeAttributesPattern):
    """Match nearest/floor Resize where integer scales prove coordinate equivalence."""

    def check_skeleton_result(
        self,
        skeleton_match_result: SkeletonMatchResult,
    ) -> PatternMatchResult | None:
        """Require nearest/floor interpolation and readable non-overridable integer scales."""
        result = super().check_skeleton_result(skeleton_match_result)
        if result is None:
            return None
        attributes = result.attributes
        if (
            attributes.get("coordinate_transformation_mode") != "tf_half_pixel_for_nn"
            or attributes.get("mode", "nearest") != "nearest"
            or attributes.get("nearest_mode", "round_prefer_floor") != "floor"
        ):
            return None
        matcher = skeleton_match_result.matcher
        scales_name = result.schema_input_to_value["scales"]
        producer = matcher.producer_lookup.get(scales_name)
        if producer is None or producer[2] not in ("Initializer", "Constant"):
            return None
        if producer[2] == "Constant":
            node = matcher.node_lookup[producer[0]]
            if node.domain not in ("", "ai.onnx"):
                return None
        scales = matcher.tensor_values.get(scales_name)
        if (
            scales is None
            or scales.dtype != np.dtype(np.float32)
            or scales.ndim != 1
            or scales.size == 0
            or not np.all(np.isfinite(scales) & (scales > 0))
            or not np.all(scales == np.floor(scales))
        ):
            return None
        sizes_name = result.schema_input_to_value["sizes"]
        if sizes_name and not _is_empty_resize_input(sizes_name, matcher):
            return None
        input_shape = matcher.get_tensor_shape(result.schema_input_to_value["X"])
        axes = attributes.get("axes")
        rank = (
            len(axes) if axes is not None else len(input_shape) if input_shape is not None else None
        )
        if rank is None or scales.size != rank:
            return None
        return result


class ResizeWithAsymmetricCoordinatesPattern(_ResizeAttributesPattern):
    """Keep nearest interpolation and use asymmetric coordinates."""

    def _rewrite_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        return {**attributes, "coordinate_transformation_mode": "asymmetric"}


class ResizeWithCubicInterpolationPattern(_ResizeAttributesPattern):
    """Match cubic Resize without antialiasing, outside exclusion, or crop semantics."""

    def check_skeleton_result(
        self,
        skeleton_match_result: SkeletonMatchResult,
    ) -> PatternMatchResult | None:
        """Exclude cubic settings that need separate linear-approximation handling."""
        result = super().check_skeleton_result(skeleton_match_result)
        if result is None:
            return None
        attributes = result.attributes
        if (
            attributes.get("mode") != "cubic"
            or attributes.get("antialias", 0) != 0
            or attributes.get("exclude_outside", 0) != 0
            or attributes.get("coordinate_transformation_mode") == "tf_crop_and_resize"
        ):
            return None
        return result


class ResizeWithLinearInterpolationPattern(_ResizeAttributesPattern):
    """Approximate cubic interpolation with linear interpolation."""

    def _rewrite_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        return {**attributes, "mode": "linear"}


__all__ = [
    "ResizeWithAsymmetricCoordinatesPattern",
    "ResizeWithCubicInterpolationPattern",
    "ResizeWithEmptyOptionalInputsPattern",
    "ResizeWithLinearInterpolationPattern",
    "ResizeWithOmittedOptionalInputsPattern",
    "ResizeWithTfHalfPixelForNNPattern",
]
