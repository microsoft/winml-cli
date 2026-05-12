# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``winml.modelkit.onnx.detection``.

Covers ``is_quantized_onnx`` for both QDQ and QOperator formats and
``is_compiled_onnx`` for EPContext detection. Builds tiny synthetic
ONNX models with the relevant ops so no network or large fixtures
are required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
from onnx import TensorProto, helper

from winml.modelkit.onnx.detection import is_compiled_onnx, is_quantized_onnx


if TYPE_CHECKING:
    from pathlib import Path


def _save(graph: onnx.GraphProto, path: Path, *, opset: int = 17) -> Path:
    """Save a graph as a minimal ONNX model."""
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


class TestIsQuantizedOnnx:
    """Both QDQ and QOperator quantization formats are detected."""

    def test_float_model_is_not_quantized(self, tmp_path: Path) -> None:
        """A plain float MatMul model is not flagged as quantized."""
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        w = helper.make_tensor(
            "W", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten().tolist()
        )
        node = helper.make_node("MatMul", ["X", "W"], ["Y"])
        graph = helper.make_graph([node], "g", [x], [y], [w])
        path = _save(graph, tmp_path / "float.onnx")
        assert is_quantized_onnx(path) is False

    def test_qdq_quantizelinear_is_detected(self, tmp_path: Path) -> None:
        """A graph containing QuantizeLinear is recognized as quantized."""
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
        y = helper.make_tensor_value_info("Y", TensorProto.UINT8, [4])
        scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.1])
        zp = helper.make_tensor("zp", TensorProto.UINT8, [], [128])
        node = helper.make_node("QuantizeLinear", ["X", "scale", "zp"], ["Y"])
        graph = helper.make_graph([node], "g", [x], [y], [scale, zp])
        path = _save(graph, tmp_path / "qdq.onnx")
        assert is_quantized_onnx(path) is True

    def test_qoperator_matmulinteger_is_detected(self, tmp_path: Path) -> None:
        """A graph containing MatMulInteger is recognized as quantized.

        Regression test for the SAM 3 ``vision_encoder_int8.onnx`` case:
        the encoder uses ``QuantFormat.QOperator`` (no QDQ pairs), so the
        old QDQ-only check returned False and the build pipeline tried to
        run the optimizer over already-quantized integer ops.
        """
        a = helper.make_tensor_value_info("A", TensorProto.UINT8, [1, 4])
        b = helper.make_tensor_value_info("B", TensorProto.UINT8, [4, 4])
        y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4])
        node = helper.make_node("MatMulInteger", ["A", "B"], ["Y"])
        graph = helper.make_graph([node], "g", [a, b], [y])
        path = _save(graph, tmp_path / "qop_matmul.onnx")
        assert is_quantized_onnx(path) is True

    def test_qoperator_convinteger_is_detected(self, tmp_path: Path) -> None:
        """A graph containing ConvInteger is recognized as quantized."""
        x = helper.make_tensor_value_info("X", TensorProto.UINT8, [1, 1, 4, 4])
        w = helper.make_tensor(
            "W", TensorProto.UINT8, [1, 1, 1, 1], np.array([1], dtype=np.uint8).tobytes(), raw=True
        )
        y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 1, 4, 4])
        node = helper.make_node("ConvInteger", ["X", "W"], ["Y"])
        graph = helper.make_graph([node], "g", [x], [y], [w])
        path = _save(graph, tmp_path / "qop_conv.onnx")
        assert is_quantized_onnx(path) is True

    def test_qoperator_qlinearmatmul_is_detected(self, tmp_path: Path) -> None:
        """A graph containing QLinearMatMul is recognized as quantized."""
        a = helper.make_tensor_value_info("A", TensorProto.UINT8, [1, 4])
        b = helper.make_tensor_value_info("B", TensorProto.UINT8, [4, 4])
        y = helper.make_tensor_value_info("Y", TensorProto.UINT8, [1, 4])
        a_scale = helper.make_tensor("a_scale", TensorProto.FLOAT, [], [0.1])
        a_zp = helper.make_tensor("a_zp", TensorProto.UINT8, [], [128])
        b_scale = helper.make_tensor("b_scale", TensorProto.FLOAT, [], [0.1])
        b_zp = helper.make_tensor("b_zp", TensorProto.UINT8, [], [128])
        y_scale = helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.1])
        y_zp = helper.make_tensor("y_zp", TensorProto.UINT8, [], [128])
        node = helper.make_node(
            "QLinearMatMul",
            ["A", "a_scale", "a_zp", "B", "b_scale", "b_zp", "y_scale", "y_zp"],
            ["Y"],
        )
        graph = helper.make_graph(
            [node], "g", [a, b], [y], [a_scale, a_zp, b_scale, b_zp, y_scale, y_zp]
        )
        path = _save(graph, tmp_path / "qop_qlinear.onnx", opset=15)
        assert is_quantized_onnx(path) is True


class TestIsCompiledOnnx:
    """EPContext detection."""

    def test_float_model_is_not_compiled(self, tmp_path: Path) -> None:
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
        node = helper.make_node("Identity", ["X"], ["Y"])
        graph = helper.make_graph([node], "g", [x], [y])
        path = _save(graph, tmp_path / "float.onnx")
        assert is_compiled_onnx(path) is False
