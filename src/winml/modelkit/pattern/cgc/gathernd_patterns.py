# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Equivalent GatherND-to-Reshape patterns for CGIR compatibility."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np
from onnx import ModelProto, helper, numpy_helper
from onnx.defs import get_schema

from ...onnx import ONNXDomain, SupportedONNXType
from .. import InputInfo, PatternMatchResult, make_single_op_pattern
from .utils import _depends_on_overridable_initializer, _static_tensor


if TYPE_CHECKING:
    from .. import SkeletonMatchResult


_GATHERND_SCHEMA, _SingleGatherNDPattern = make_single_op_pattern(get_schema("GatherND", 12))


class GatherNDWithIdentityIndicesPattern(_SingleGatherNDPattern):  # type: ignore[misc, valid-type]
    """Match unequal-rank GatherND that visits every input slice in storage order."""

    def check_skeleton_result(
        self,
        skeleton_match_result: SkeletonMatchResult,
    ) -> PatternMatchResult | None:
        """Require unequal ranks and complete, ordered indexing in every batch."""
        node = skeleton_match_result.matched_nodes[0]
        matcher = skeleton_match_result.matcher
        if len(node.input) != 2 or len(node.output) != 1:
            return None
        data_name, indices_name = node.input
        indices = _static_tensor(indices_name, matcher)
        data_shape = matcher.get_tensor_shape(data_name)
        if (
            indices is None
            or indices.dtype != np.dtype(np.int64)
            or indices.ndim == 0
            or indices.size == 0
            or data_shape is None
            or not data_shape
            or any(not isinstance(dim, int) or dim <= 0 for dim in data_shape)
        ):
            return None

        if _depends_on_overridable_initializer(data_name, matcher):
            return None

        batch_dims = next(
            (
                helper.get_attribute_value(attr)
                for attr in node.attribute
                if attr.name == "batch_dims"
            ),
            0,
        )
        depth = indices.shape[-1]
        if (
            not isinstance(batch_dims, int)
            or not 0 <= batch_dims < min(len(data_shape), indices.ndim)
            or not 1 <= depth <= len(data_shape) - batch_dims
            or indices.shape[:batch_dims] != data_shape[:batch_dims]
        ):
            return None
        indexed_shape = data_shape[batch_dims : batch_dims + depth]
        slice_count = prod(indexed_shape)
        if prod(indices.shape[batch_dims:-1]) != slice_count:
            return None
        output_shape = indices.shape[:-1] + data_shape[batch_dims + depth :]
        if len(data_shape) == indices.ndim == len(output_shape):
            return None

        # Compare coordinates, not data values: each batch must traverse all
        # indexed slices once in row-major order. Negative equivalents are valid.
        coordinates = indices.reshape(-1, slice_count, depth)
        positions = np.arange(slice_count, dtype=np.int64)
        stride = slice_count
        for axis, extent in enumerate(indexed_shape):
            stride //= extent
            expected = (positions // stride) % extent
            actual = coordinates[:, :, axis]
            if not np.all((actual == expected) | (actual == expected - extent)):
                return None

        type_mapping = self._infer_type_mapping(skeleton_match_result)
        if "T" not in type_mapping:
            return None
        return PatternMatchResult(
            skeleton_match_result=skeleton_match_result,
            schema_input_to_value={"data": data_name, "indices": indices_name},
            schema_output_to_value={"output": node.output[0]},
            type_param_to_type=type_mapping,
            attributes={
                "_data_shape": data_shape,
                "_output_shape": output_shape,
                "_ir_version": matcher.model.ir_version,
            },
            input_infos={
                "data": InputInfo(name="data"),
                "indices": InputInfo(name="indices"),
            },
        )


class ReshapedGatherNDPattern(_SingleGatherNDPattern):  # type: ignore[misc, valid-type]
    """Replace proven identity indexing with a statically shaped Reshape."""

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
        """Emit Reshape with a derived shape initializer and unchanged data type."""
        del inputs, is_constant_map
        if input_names is None or len(input_names) != 2:
            raise ValueError("GatherND rewrite requires data and indices names")
        if output_names is None or len(output_names) != 1:
            raise ValueError("GatherND rewrite requires one output name")
        data_shape = attributes["_data_shape"]
        output_shape = attributes["_output_shape"]
        element_type = SupportedONNXType.from_onnx_type(output_dtypes[0]).tensor_proto_type
        shape_name = f"{prefix}shape"
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Reshape",
                    [input_names[0], shape_name],
                    output_names,
                    name=f"{prefix}Reshape",
                )
            ],
            f"{prefix}ReshapedGatherND",
            [helper.make_tensor_value_info(input_names[0], element_type, data_shape)],
            [helper.make_tensor_value_info(output_names[0], element_type, output_shape)],
            [numpy_helper.from_array(np.asarray(output_shape, dtype=np.int64), shape_name)],
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


__all__ = ["GatherNDWithIdentityIndicesPattern", "ReshapedGatherNDPattern"]
