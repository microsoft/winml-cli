# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for UnsqueezeCastPattern.

The exemplar production case is the ``/model/decoder/Unsqueeze_1`` ->
``/model/decoder/Cast_1`` pair in google-t5/t5-small.onnx where a 4-D
attention-mask tensor is unsqueezed and then cast to float32.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from winml.modelkit.pattern import (
    PatternMatcher,
    UnsqueezeCastPattern,
)

from .conftest import TEST_DOMAIN_VERSIONS


_FLOAT = int(TensorProto.FLOAT)


def _build_unsqueeze_cast_model(
    *,
    data_shape: tuple[int, ...] = (2, 3),
    data_elem_type: int = TensorProto.INT64,
    axes: tuple[int, ...] = (1,),
    cast_to: int = _FLOAT,
    axes_as_initializer: bool = True,
) -> onnx.ModelProto:
    """Build a minimal ONNX model containing only Unsqueeze -> Cast."""
    out_rank = len(data_shape) + len(axes)
    norm_axes = sorted(a if a >= 0 else a + out_rank for a in axes)
    out_shape: list[int] = []
    data_iter = iter(data_shape)
    for i in range(out_rank):
        if i in norm_axes:
            out_shape.append(1)
        else:
            out_shape.append(next(data_iter))

    data = helper.make_tensor_value_info("data", data_elem_type, list(data_shape))
    output = helper.make_tensor_value_info("output", cast_to, out_shape)

    axes_arr = np.array(axes, dtype=np.int64)

    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []
    if axes_as_initializer:
        initializers.append(
            helper.make_tensor("axes", TensorProto.INT64, list(axes_arr.shape), axes_arr.tolist())
        )
        nodes.append(helper.make_node("Unsqueeze", ["data", "axes"], ["unsq_out"], name="unsq"))
    else:
        nodes.append(helper.make_node("Unsqueeze", ["data", "axes_dyn"], ["unsq_out"], name="unsq"))

    nodes.append(helper.make_node("Cast", ["unsq_out"], ["output"], name="cast", to=cast_to))

    graph_inputs = [data]
    if not axes_as_initializer:
        graph_inputs.append(
            helper.make_tensor_value_info("axes_dyn", TensorProto.INT64, [len(axes)])
        )

    graph = helper.make_graph(
        nodes=nodes,
        name="unsqueeze_cast_test",
        inputs=graph_inputs,
        outputs=[output],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _match(model: onnx.ModelProto) -> list:
    matcher = PatternMatcher(model)
    matcher.register_pattern(UnsqueezeCastPattern())
    return matcher.match()


class TestUnsqueezeCastPatternMatching:
    """Topology / constraint behaviour of UnsqueezeCastPattern."""

    def test_matches_float_cast(self) -> None:
        model = _build_unsqueeze_cast_model()
        results = _match(model)
        assert len(results) == 1
        attrs = results[0].attributes
        assert attrs["axes"] == (1,)
        assert attrs["to"] == _FLOAT

    @pytest.mark.parametrize("axes", [(0,), (1,), (-1,), (0, 2)])
    def test_matches_various_axes(self, axes: tuple[int, ...]) -> None:
        model = _build_unsqueeze_cast_model(data_shape=(2, 3, 4), axes=axes)
        results = _match(model)
        assert len(results) == 1
        assert results[0].attributes["axes"] == axes

    def test_rejects_non_float_cast(self) -> None:
        """Cast(to=int32) must not match."""
        model = _build_unsqueeze_cast_model(cast_to=TensorProto.INT32)
        results = _match(model)
        assert results == []

    def test_rejects_non_constant_axes(self) -> None:
        """Unsqueeze with a dynamic (graph input) axes input must not match."""
        model = _build_unsqueeze_cast_model(axes_as_initializer=False)
        results = _match(model)
        assert results == []

    def test_matches_float_input_too(self) -> None:
        """Pattern is dtype-agnostic on the input side: float input also matches."""
        model = _build_unsqueeze_cast_model(data_elem_type=TensorProto.FLOAT)
        results = _match(model)
        assert len(results) == 1


class TestUnsqueezeCastPatternRoundTrip:
    """Self-matching via Pattern.get_onnx_model."""

    def test_get_onnx_model_self_matches(self) -> None:
        pattern = UnsqueezeCastPattern()
        inputs = {"data": np.random.randn(2, 3).astype(np.float32)}
        attributes = {"axes": (1,), "to": _FLOAT}
        model = pattern.get_onnx_model(
            inputs,
            attributes,
            is_constant_map={"data": False},
            output_dtypes=["tensor(float)"],
            domain_versions=TEST_DOMAIN_VERSIONS,
        )
        onnx.checker.check_model(model)

        results = _match(model)
        assert len(results) == 1
        assert results[0].attributes["axes"] == (1,)
        assert results[0].attributes["to"] == _FLOAT
