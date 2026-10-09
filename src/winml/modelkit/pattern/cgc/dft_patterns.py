# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Static-axis DFT decomposition into real-valued matrix multiplications."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from onnx import ModelProto, helper, numpy_helper
from onnx.defs import get_schema

from ...onnx import ONNXDomain, SupportedONNXType
from .. import (
    InputInfo,
    Pattern,
    PatternMatchResult,
    PatternSchema,
    Skeleton,
    opschema_to_pattern_schema,
)
from .utils import _depends_on_overridable_initializer, _static_tensor


if TYPE_CHECKING:
    from .. import PatternMatcher, SkeletonMatchResult


_ONNX_DFT_SCHEMA = opschema_to_pattern_schema(get_schema("DFT", 20))
_DFT_SCHEMA = replace(_ONNX_DFT_SCHEMA, inputs=_ONNX_DFT_SCHEMA.inputs[:1])
_DFT_TYPES = set(get_schema("MatMul", 9).type_constraints[0].allowed_type_strs) & set(
    get_schema("DFT", 20).type_constraints[0].allowed_type_strs
)


def _static_integer(name: str, matcher: PatternMatcher) -> int | None:
    value = _static_tensor(name, matcher)
    if (
        value is None
        or value.ndim != 0
        or value.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
    ):
        return None
    return int(value)


class _DFTPattern(Pattern):
    def get_schema(self) -> PatternSchema:
        return _DFT_SCHEMA

    def get_skeleton(self) -> Skeleton:
        return Skeleton(
            node_op_types=["DFT"],
            node_domains=[ONNXDomain.AI_ONNX],
            edges=[(-1, 0, 0, 0)],
            exit_nodes=[0],
            n_inputs=1,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        return [], {}


class DFTWithStaticParametersPattern(_DFTPattern):
    """Match DFT with known signal length, component count, and transform axis."""

    def check_skeleton_result(
        self,
        skeleton_match_result: SkeletonMatchResult,
    ) -> PatternMatchResult | None:
        """Require static transform parameters without specializing batch dimensions."""
        node = skeleton_match_result.matched_nodes[0]
        matcher = skeleton_match_result.matcher
        opset = matcher.domain_versions.get(ONNXDomain.AI_ONNX, 0)
        if opset < 17 or not node.input or len(node.output) != 1:
            return None
        input_name = node.input[0]
        shape = matcher.get_tensor_shape(input_name)
        dtype = matcher.get_tensor_type_str(input_name)
        if (
            shape is None
            or len(shape) < 2
            or shape[-1] not in (1, 2)
            or dtype is None
            or dtype not in _DFT_TYPES
            or _depends_on_overridable_initializer(input_name, matcher)
        ):
            return None
        attributes = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        allowed_attributes = {"inverse", "onesided"} | ({"axis"} if opset < 20 else set())
        if attributes.keys() - allowed_attributes:
            return None
        inverse, onesided = attributes.get("inverse", 0), attributes.get("onesided", 0)
        if inverse not in (0, 1) or onesided not in (0, 1):
            return None
        if onesided and (inverse or shape[-1] != 1):
            return None
        if opset < 20:
            if len(node.input) > 2:
                return None
            axis = attributes.get("axis", 1)
        else:
            if len(node.input) > 3:
                return None
            axis = (
                _static_integer(node.input[2], matcher)
                if len(node.input) > 2 and node.input[2]
                else -2
            )
        if not isinstance(axis, int) or not -len(shape) <= axis < len(shape) - 1:
            return None
        axis %= len(shape)
        if axis == len(shape) - 1:
            return None
        input_length = shape[axis]
        if not isinstance(input_length, int) or input_length <= 0:
            return None
        length = (
            _static_integer(node.input[1], matcher)
            if len(node.input) > 1 and node.input[1]
            else input_length
        )
        if length is None or length <= 0:
            return None
        return PatternMatchResult(
            skeleton_match_result=skeleton_match_result,
            schema_input_to_value={"input": input_name},
            schema_output_to_value={"output": node.output[0]},
            type_param_to_type={"T1": dtype},
            attributes={
                "_shape": shape,
                "_axis": axis,
                "_length": length,
                "_inverse": inverse,
                "_onesided": onesided,
                "_ir_version": matcher.model.ir_version,
            },
            input_infos={"input": InputInfo(name="input")},
        )


class MatMulDFTPattern(_DFTPattern):
    """Generate the DFT definition using sine/cosine bases and complex arithmetic."""

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
        """Preserve transform direction, normalization, truncation, and output layout."""
        del inputs, is_constant_map
        if input_names is None or len(input_names) != 1:
            raise ValueError("DFT rewrite requires one signal input")
        if output_names is None or len(output_names) != 1:
            raise ValueError("DFT rewrite requires one output")
        shape = attributes["_shape"]
        axis, length = attributes["_axis"], attributes["_length"]
        inverse, onesided = attributes["_inverse"], attributes["_onesided"]
        input_length = min(shape[axis], length)
        output_length = length // 2 + 1 if onesided else length
        output_type = SupportedONNXType.from_onnx_type(output_dtypes[0])
        nodes, initializers = [], []

        def constant(name: str, value: np.ndarray) -> str:
            name = f"{prefix}{name}"
            initializers.append(numpy_helper.from_array(value, name))
            return name

        def op(op_type: str, names: list[str], suffix: str, **kwargs: Any) -> str:
            result = f"{prefix}{suffix}"
            nodes.append(helper.make_node(op_type, names, [result], name=result, **kwargs))
            return result

        # Only the retained input samples contribute. Missing samples are zero,
        # so zero-padding does not need an activation-sized temporary tensor.
        phase = (
            2
            * np.pi
            * (
                np.outer(
                    np.arange(input_length, dtype=np.float64),
                    np.arange(output_length, dtype=np.float64),
                )
                % length
            )
            / length
        )
        normalization = 1 / length if inverse else 1
        cosine = constant("cosine", (np.cos(phase) * normalization).astype(output_type.np_type))
        sine = constant(
            "sine",
            (np.sin(phase) * (1 if inverse else -1) * normalization).astype(output_type.np_type),
        )
        signal = input_names[0]
        if shape[axis] > length:
            signal = op(
                "Slice",
                [
                    signal,
                    constant("starts", np.asarray([0], dtype=np.int64)),
                    constant("ends", np.asarray([length], dtype=np.int64)),
                    constant("slice_axes", np.asarray([axis], dtype=np.int64)),
                ],
                "truncated",
            )
        permutation = [i for i in range(len(shape) - 1) if i != axis] + [axis, len(shape) - 1]
        transposed = permutation != list(range(len(shape)))
        if transposed:
            signal = op("Transpose", [signal], "transposed", perm=permutation)
        real = op(
            "Gather",
            [signal, constant("real_index", np.asarray(0, dtype=np.int64))],
            "real",
            axis=-1,
        )
        real_out = op("MatMul", [real, cosine], "real_cosine")
        imag_out = op("MatMul", [real, sine], "real_sine")
        if shape[-1] == 2:
            imag = op(
                "Gather",
                [signal, constant("imag_index", np.asarray(1, dtype=np.int64))],
                "imag",
                axis=-1,
            )
            real_out = op("Sub", [real_out, op("MatMul", [imag, sine], "imag_sine")], "real_output")
            imag_out = op(
                "Add", [imag_out, op("MatMul", [imag, cosine], "imag_cosine")], "imag_output"
            )
        component_axis = constant("component_axis", np.asarray([-1], dtype=np.int64))
        real_out = op("Unsqueeze", [real_out, component_axis], "real_component")
        imag_out = op("Unsqueeze", [imag_out, component_axis], "imag_component")
        result = op("Concat", [real_out, imag_out], "complex_output", axis=-1)
        if transposed:
            op("Transpose", [result], "restored", perm=np.argsort(permutation).tolist())
        nodes[-1].output[0] = output_names[0]
        output_shape = list(shape)
        output_shape[axis], output_shape[-1] = output_length, 2
        graph = helper.make_graph(
            nodes,
            f"{prefix}MatMulDFT",
            [helper.make_tensor_value_info(input_names[0], output_type.tensor_proto_type, shape)],
            [
                helper.make_tensor_value_info(
                    output_names[0], output_type.tensor_proto_type, output_shape
                )
            ],
            initializers,
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


__all__ = ["DFTWithStaticParametersPattern", "MatMulDFTPattern"]
