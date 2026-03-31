# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ONNX model inspection utilities."""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from winml.modelkit.onnx.inspection import (
    find_undefined_types,
    format_model_type_summary,
    get_qdq_param_info,
    get_value_info_dims,
    get_value_info_elem_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_model() -> onnx.ModelProto:
    """Build a simple model with known types."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph([node], "simple", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _make_model_with_undefined() -> onnx.ModelProto:
    """Build a model with UNDEFINED type entries."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    undef = helper.make_tensor_value_info("bad_tensor", TensorProto.UNDEFINED, None)
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["X"], ["Y"], name="relu")
    graph = helper.make_graph([node], "undef_test", [x, undef], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _make_qdq_model() -> onnx.ModelProto:
    """Build a QDQ model for param info testing."""
    scale_init = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.05])
    zp_init = helper.make_tensor("zp", TensorProto.UINT8, [], [128])

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    scale_vi = helper.make_tensor_value_info("scale", TensorProto.UNDEFINED, None)
    zp_vi = helper.make_tensor_value_info("zp", TensorProto.UINT8, [])

    q = helper.make_node("QuantizeLinear", ["X", "scale", "zp"], ["Q_out"], name="Q")
    dq = helper.make_node("DequantizeLinear", ["Q_out", "scale", "zp"], ["Y"], name="DQ")

    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [q, dq],
        "qdq_test",
        [x, scale_vi, zp_vi],
        [y],
        initializer=[scale_init, zp_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


# ---------------------------------------------------------------------------
# Tests: get_value_info_elem_type
# ---------------------------------------------------------------------------


class TestGetValueInfoElemType:
    def test_returns_float_for_float_tensor(self):
        vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        assert get_value_info_elem_type(vi) == TensorProto.FLOAT

    def test_returns_undefined_for_undefined(self):
        vi = helper.make_tensor_value_info("x", TensorProto.UNDEFINED, None)
        assert get_value_info_elem_type(vi) == TensorProto.UNDEFINED

    def test_returns_uint8(self):
        vi = helper.make_tensor_value_info("x", TensorProto.UINT8, [])
        assert get_value_info_elem_type(vi) == TensorProto.UINT8


# ---------------------------------------------------------------------------
# Tests: get_value_info_dims
# ---------------------------------------------------------------------------


class TestGetValueInfoDims:
    def test_returns_static_dims(self):
        vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 224, 224])
        assert get_value_info_dims(vi) == [1, 3, 224, 224]

    def test_returns_symbolic_dims(self):
        vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
        assert get_value_info_dims(vi) == ["batch", 4]

    def test_returns_none_for_no_shape(self):
        vi = helper.make_tensor_value_info("x", TensorProto.UNDEFINED, None)
        # UNDEFINED type with no shape
        result = get_value_info_dims(vi)
        # With UNDEFINED type, tensor_type field may not have shape
        assert result is None or result == []

    def test_scalar_returns_empty_list(self):
        vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [])
        assert get_value_info_dims(vi) == []


# ---------------------------------------------------------------------------
# Tests: find_undefined_types
# ---------------------------------------------------------------------------


class TestFindUndefinedTypes:
    def test_finds_undefined_in_graph_input(self):
        model = _make_model_with_undefined()
        results = find_undefined_types(model)
        names = [r["name"] for r in results]
        assert "bad_tensor" in names

    def test_empty_when_all_defined(self):
        model = _make_simple_model()
        results = find_undefined_types(model)
        assert len(results) == 0

    def test_source_is_reported(self):
        model = _make_model_with_undefined()
        results = find_undefined_types(model)
        bad = next(r for r in results if r["name"] == "bad_tensor")
        assert bad["source"] == "graph_input"


# ---------------------------------------------------------------------------
# Tests: get_qdq_param_info
# ---------------------------------------------------------------------------


class TestGetQdqParamInfo:
    def test_reports_scale_and_zp(self):
        model = _make_qdq_model()
        info_list = get_qdq_param_info(model)
        assert len(info_list) == 2  # Q + DQ nodes

    def test_detects_undefined_scale(self):
        model = _make_qdq_model()
        info_list = get_qdq_param_info(model)
        # Scale has UNDEFINED type in graph.input
        q_info = next(i for i in info_list if i["op_type"] == "QuantizeLinear")
        assert q_info["scale_vi_type"] == TensorProto.UNDEFINED
        assert any("UNDEFINED" in issue for issue in q_info["issues"])

    def test_reports_correct_initializer_type(self):
        model = _make_qdq_model()
        info_list = get_qdq_param_info(model)
        q_info = info_list[0]
        assert q_info["scale_init_type"] == TensorProto.FLOAT
        assert q_info["zp_init_type"] == TensorProto.UINT8

    def test_no_qdq_returns_empty(self):
        model = _make_simple_model()
        assert get_qdq_param_info(model) == []


# ---------------------------------------------------------------------------
# Tests: format_model_type_summary
# ---------------------------------------------------------------------------


class TestFormatModelTypeSummary:
    def test_returns_string(self):
        model = _make_simple_model()
        summary = format_model_type_summary(model)
        assert isinstance(summary, str)
        assert "Nodes:" in summary

    def test_shows_undefined_types(self):
        model = _make_model_with_undefined()
        summary = format_model_type_summary(model)
        assert "UNDEFINED" in summary

    def test_shows_qdq_issues(self):
        model = _make_qdq_model()
        summary = format_model_type_summary(model)
        assert "QDQ Issues" in summary
