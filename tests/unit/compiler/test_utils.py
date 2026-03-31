# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for compiler detection utilities and transform registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx
from onnx import TensorProto, helper

from winml.modelkit.compiler.transforms import (
    clear_transforms,
    get_transforms_for_ep,
    register_transform,
)
from winml.modelkit.compiler.utils import needs_format_conversion
from winml.modelkit.onnx import is_compiled_onnx, is_quantized_onnx


if TYPE_CHECKING:
    from pathlib import Path


def _make_simple_model(op_types: list[str]) -> onnx.ModelProto:
    """Build a minimal ONNX model with given op types."""
    nodes = []
    for i, op_type in enumerate(op_types):
        node = helper.make_node(
            op_type,
            inputs=[f"input_{i}"],
            outputs=[f"output_{i}"],
        )
        nodes.append(node)

    input_info = helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [1, 1])
    output_info = helper.make_tensor_value_info(
        f"output_{len(nodes) - 1}", TensorProto.FLOAT, [1, 1]
    )
    graph = helper.make_graph(nodes, "test_graph", [input_info], [output_info])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestIsQuantizedOnnx:
    def test_model_with_qdq(self, tmp_path: Path) -> None:
        model = _make_simple_model(["QuantizeLinear", "DequantizeLinear", "Conv"])
        path = tmp_path / "qdq.onnx"
        onnx.save(model, str(path))
        assert is_quantized_onnx(path) is True

    def test_model_without_qdq(self, tmp_path: Path) -> None:
        model = _make_simple_model(["Conv", "Relu"])
        path = tmp_path / "plain.onnx"
        onnx.save(model, str(path))
        assert is_quantized_onnx(path) is False

    def test_model_with_only_quantize(self, tmp_path: Path) -> None:
        model = _make_simple_model(["QuantizeLinear"])
        path = tmp_path / "q_only.onnx"
        onnx.save(model, str(path))
        assert is_quantized_onnx(path) is True


class TestIsCompiledOnnx:
    def test_model_with_ep_context(self, tmp_path: Path) -> None:
        model = _make_simple_model(["EPContext"])
        path = tmp_path / "ctx.onnx"
        onnx.save(model, str(path))
        assert is_compiled_onnx(path) is True

    def test_model_without_ep_context(self, tmp_path: Path) -> None:
        model = _make_simple_model(["Conv", "Relu"])
        path = tmp_path / "no_ctx.onnx"
        onnx.save(model, str(path))
        assert is_compiled_onnx(path) is False


class TestNeedsFormatConversion:
    def test_qlinear_for_qnn(self, tmp_path: Path) -> None:
        model = _make_simple_model(["QLinearConv", "Relu"])
        path = tmp_path / "qlinear.onnx"
        onnx.save(model, str(path))
        assert needs_format_conversion(path, "qnn") is True

    def test_qdq_for_qnn(self, tmp_path: Path) -> None:
        model = _make_simple_model(["QuantizeLinear", "DequantizeLinear"])
        path = tmp_path / "qdq.onnx"
        onnx.save(model, str(path))
        assert needs_format_conversion(path, "qnn") is False

    def test_plain_for_qnn(self, tmp_path: Path) -> None:
        model = _make_simple_model(["Conv"])
        path = tmp_path / "plain.onnx"
        onnx.save(model, str(path))
        assert needs_format_conversion(path, "qnn") is False

    def test_any_for_cpu(self, tmp_path: Path) -> None:
        model = _make_simple_model(["QLinearConv"])
        path = tmp_path / "qlinear.onnx"
        onnx.save(model, str(path))
        assert needs_format_conversion(path, "cpu") is False  # Not implemented yet


class TestTransformRegistry:
    def setup_method(self) -> None:
        clear_transforms()

    def teardown_method(self) -> None:
        clear_transforms()

    def test_register_and_get(self) -> None:
        class QnnTransform:
            def applies_to(self, ep: str) -> bool:
                return ep == "qnn"

            def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
                return model

        register_transform(QnnTransform())
        assert len(get_transforms_for_ep("qnn")) == 1
        assert len(get_transforms_for_ep("dml")) == 0

    def test_clear_transforms(self) -> None:
        class AnyTransform:
            def applies_to(self, ep: str) -> bool:
                return True

            def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
                return model

        register_transform(AnyTransform())
        assert len(get_transforms_for_ep("qnn")) == 1
        clear_transforms()
        assert len(get_transforms_for_ep("qnn")) == 0

    def test_multiple_transforms(self) -> None:
        class T1:
            def applies_to(self, ep: str) -> bool:
                return ep == "qnn"

            def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
                return model

        class T2:
            def applies_to(self, ep: str) -> bool:
                return ep == "qnn"

            def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
                return model

        register_transform(T1())
        register_transform(T2())
        assert len(get_transforms_for_ep("qnn")) == 2
