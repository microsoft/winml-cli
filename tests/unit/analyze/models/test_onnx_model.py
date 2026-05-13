# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Unit tests for ONNXModel Pydantic validation.

Tests verify:
- opset_version validation (>= 13)
- graph serialization and deserialization
- from_onnx_model class method
- get_graph method
"""

import pytest
from onnx import TensorProto, helper
from pydantic import ValidationError

from winml.modelkit.analyze import ONNXModel


class TestONNXModelValidation:
    """Test ONNXModel Pydantic validation rules."""

    @pytest.mark.parametrize("opset", [12, 13, 14, 15, 16])
    def test_opset_version_must_be_at_least_12(self, opset):
        """Test that opset_version must be >= 12."""
        # Valid opset
        model = ONNXModel(
            model_path="test.onnx",
            opset_version=opset,
            node_count=1,
            initializer_count=0,
            input_count=1,
            output_count=1,
        )
        assert model.opset_version == opset

    def test_opset_version_below_12_invalid(self):
        """Test that opset_version < 12 raises ValidationError."""
        # Invalid opset (< 12)
        with pytest.raises(ValidationError, match=r"Opset version .* < 12"):
            ONNXModel(
                model_path="test.onnx",
                opset_version=11,
                node_count=1,
                initializer_count=0,
                input_count=1,
                output_count=1,
            )

    def test_graph_must_not_be_empty(self):
        """Test that node_count must not be zero."""
        # Valid non-empty graph
        ONNXModel(
            model_path="test.onnx",
            opset_version=12,
            node_count=1,
            initializer_count=0,
            input_count=1,
            output_count=1,
        )

        # Invalid empty graph (node_count = 0)
        with pytest.raises(ValidationError, match="Graph must contain at least one node"):
            ONNXModel(
                model_path="test.onnx",
                opset_version=12,
                node_count=0,
                initializer_count=0,
                input_count=1,
                output_count=1,
            )

    def test_from_onnx_model_creates_instance(self):
        """Test that from_onnx_model class method creates valid ONNXModel."""
        # Create a minimal ONNX model
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

        node = helper.make_node("Identity", ["input"], ["output"])
        graph_def = helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])

        onnx_model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 12)])

        # Convert to ONNXModel
        model = ONNXModel.from_onnx_model(onnx_model, "test.onnx")

        assert model.model_path == "test.onnx"
        assert model.opset_version == 12
        assert model.node_count == 1
        assert model.input_count == 1
        assert model.output_count == 1

    def test_get_graph_deserializes_correctly(self):
        """Test that get_graph() returns a valid GraphProto."""
        # Create a minimal ONNX model
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

        node = helper.make_node("Conv", ["input", "weights"], ["output"])
        graph_def = helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])

        onnx_model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 12)])

        # Convert to ONNXModel and back
        model = ONNXModel.from_onnx_model(onnx_model, "conv.onnx")
        graph_proto = model.get_graph()

        assert graph_proto.name == "test_graph"
        assert len(graph_proto.node) == 1
        assert graph_proto.node[0].op_type == "Conv"

    def test_round_trip_serialization(self):
        """Test that graph can be serialized and deserialized without data loss."""
        # Create a more complex graph
        input_tensor = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64, 28, 28])
        output_tensor = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 128, 14, 14])

        nodes = [
            helper.make_node(
                "Conv",
                ["x", "w1"],
                ["conv_out"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            ),
            helper.make_node("Relu", ["conv_out"], ["relu_out"]),
            helper.make_node("MaxPool", ["relu_out"], ["y"], kernel_shape=[2, 2], strides=[2, 2]),
        ]

        graph_def = helper.make_graph(nodes, "complex_graph", [input_tensor], [output_tensor])
        onnx_model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 12)])

        # Round-trip conversion
        model = ONNXModel.from_onnx_model(onnx_model, "complex.onnx")
        reconstructed_graph = model.get_graph()

        assert reconstructed_graph.name == "complex_graph"
        assert len(reconstructed_graph.node) == 3
        assert reconstructed_graph.node[0].op_type == "Conv"
        assert reconstructed_graph.node[1].op_type == "Relu"
        assert reconstructed_graph.node[2].op_type == "MaxPool"

    def test_model_path_preserved(self):
        """Test that model_path is correctly stored and retrieved."""
        model = ONNXModel(
            model_path="/path/to/model.onnx",
            opset_version=15,
            node_count=1,
            initializer_count=0,
            input_count=1,
            output_count=1,
        )

        assert model.model_path == "/path/to/model.onnx"

    @pytest.mark.parametrize("opset", [14, 15, 16, 17, 18, 19, 20])
    def test_opset_version_higher_than_13(self, opset):
        """Test that opset versions > 13 are accepted."""
        model = ONNXModel(
            model_path="test.onnx",
            opset_version=opset,
            node_count=1,
            initializer_count=0,
            input_count=1,
            output_count=1,
        )
        assert model.opset_version == opset

    def test_stable_node_key_sidecar_for_unnamed_nodes(self):
        """Test unnamed nodes get stable sidecar keys and can be reverse-looked-up."""
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

        add_node = helper.make_node("Add", ["input", "input"], ["mid"])
        relu_node = helper.make_node("Relu", ["mid"], ["output"])

        graph_def = helper.make_graph(
            [add_node, relu_node],
            "unnamed_graph",
            [input_tensor],
            [output_tensor],
        )
        onnx_model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 12)])

        model = ONNXModel.from_onnx_model(onnx_model, "unnamed.onnx")
        graph = model.get_graph()

        assert model.get_node_key(graph.node[0]) == "node_0"
        assert model.get_node_key(graph.node[1]) == "node_1"
        assert model.get_node_by_key("node_0") is graph.node[0]
        assert model.get_node_by_key("node_1") is graph.node[1]

    def test_get_node_key_rejects_unknown_unnamed_node(self):
        """Unknown unnamed nodes should raise instead of using node_obj fallback."""
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

        relu_node = helper.make_node("Relu", ["input"], ["output"])
        graph_def = helper.make_graph([relu_node], "single_graph", [input_tensor], [output_tensor])
        onnx_model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 12)])
        model = ONNXModel.from_onnx_model(onnx_model, "single.onnx")

        unknown_unnamed_node = helper.make_node("Relu", ["x"], ["y"])
        with pytest.raises(KeyError, match="unnamed node outside ONNXModel graph"):
            model.get_node_key(unknown_unnamed_node)
