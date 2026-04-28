# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.onnx.utils — check_onnx_model and has_unloaded_external_data."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from winml.modelkit.onnx import check_onnx_model, has_unloaded_external_data


def _make_simple_model() -> onnx.ModelProto:
    """Minimal valid ONNX model with an inline initializer."""
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    weight = numpy_helper.from_array(np.ones((4,), dtype=np.float32), name="W")
    node = helper.make_node("Add", ["X", "W"], ["Y"], name="add")
    graph = helper.make_graph([node], "g", [x_info], [y_info], initializer=[weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_model_with_unloaded_external_data() -> onnx.ModelProto:
    """Model whose initializer mimics load_external_data=False: data_location=EXTERNAL, raw_data empty."""
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    weight = numpy_helper.from_array(np.ones((4,), dtype=np.float32), name="W")
    # Simulate what onnx.load(..., load_external_data=False) produces for large tensors.
    weight.data_location = TensorProto.EXTERNAL
    weight.ClearField("raw_data")
    entry = weight.external_data.add()
    entry.key = "location"
    entry.value = "model.onnx.data"

    node = helper.make_node("Add", ["X", "W"], ["Y"], name="add")
    graph = helper.make_graph([node], "g", [x_info], [y_info], initializer=[weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestHasUnloadedExternalData:
    def test_inline_model_returns_false(self):
        assert has_unloaded_external_data(_make_simple_model()) is False

    def test_unloaded_external_tensor_returns_true(self):
        assert has_unloaded_external_data(_make_model_with_unloaded_external_data()) is True

    def test_loaded_external_data_returns_false(self):
        """After raw_data is populated the tensor is considered loaded."""
        model = _make_model_with_unloaded_external_data()
        init = model.graph.initializer[0]
        init.raw_data = np.ones((4,), dtype=np.float32).tobytes()
        assert has_unloaded_external_data(model) is False


class TestCheckOnnxModel:
    def test_valid_inline_model_does_not_raise(self):
        check_onnx_model(_make_simple_model())

    def test_unloaded_external_data_does_not_raise_when_skip_enabled(self):
        """check_onnx_model must not raise when skip_if_unloaded_external_data=True."""
        check_onnx_model(
            _make_model_with_unloaded_external_data(),
            skip_if_unloaded_external_data=True,
        )

    def test_unloaded_external_data_raises_by_default(self):
        """check_onnx_model raises by default when external data is missing on disk."""
        with pytest.raises(onnx.checker.ValidationError):
            check_onnx_model(_make_model_with_unloaded_external_data())

    def test_invalid_model_raises(self):
        """check_onnx_model should still raise for structurally invalid models."""
        model = _make_simple_model()
        # Corrupt the opset version to something invalid.
        model.opset_import[0].version = 0
        with pytest.raises(onnx.checker.ValidationError):
            check_onnx_model(model)
