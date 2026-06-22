# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for core/onnx_utils.py - ONNX utility functions."""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from winml.modelkit.core.onnx_utils import get_io_config


class TestGetIoConfig:
    """Tests for get_io_config() function."""

    def test_single_input_single_output(self) -> None:
        """Test simple model with one input and one output."""
        # Create simple model: input -> Identity -> output
        x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

        node = helper.make_node("Identity", ["input"], ["output"])
        graph = helper.make_graph([node], "test_graph", [x_info], [y_info])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        config = get_io_config(model)

        assert config["input_names"] == ["input"]
        assert config["output_names"] == ["output"]
        assert config["input_shapes"] == [[1, 3, 224, 224]]
        assert config["output_shapes"] == [[1, 3, 224, 224]]
        assert config["input_types"] == [np.float32]
        assert config["output_types"] == [np.float32]

    def test_multiple_inputs_outputs(self) -> None:
        """Test model with multiple inputs and outputs."""
        # Create model: two inputs -> Concat -> two outputs via Split
        in_a = helper.make_tensor_value_info("input_a", TensorProto.FLOAT, [1, 10])
        in_b = helper.make_tensor_value_info("input_b", TensorProto.FLOAT, [1, 10])
        out_x = helper.make_tensor_value_info("output_x", TensorProto.FLOAT, [1, 10])
        out_y = helper.make_tensor_value_info("output_y", TensorProto.FLOAT, [1, 10])

        # Identity nodes for simplicity
        node1 = helper.make_node("Identity", ["input_a"], ["output_x"])
        node2 = helper.make_node("Identity", ["input_b"], ["output_y"])

        graph = helper.make_graph([node1, node2], "test_graph", [in_a, in_b], [out_x, out_y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        config = get_io_config(model)

        assert config["input_names"] == ["input_a", "input_b"]
        assert config["output_names"] == ["output_x", "output_y"]
        assert len(config["input_shapes"]) == 2
        assert len(config["output_shapes"]) == 2

    def test_dynamic_dimensions(self) -> None:
        """Test model with dynamic batch dimension."""
        # Create model with dynamic batch size (None in shape)
        x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 224, 224])
        y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 1000])

        node = helper.make_node("Identity", ["input"], ["output"])
        graph = helper.make_graph([node], "test_graph", [x_info], [y_info])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        config = get_io_config(model)

        # Dynamic dims should be None
        assert config["input_shapes"] == [[None, 3, 224, 224]]
        assert config["output_shapes"] == [[None, 1000]]

    def test_various_dtypes(self) -> None:
        """Test model with various data types."""
        # Create inputs with different dtypes
        float32_input = helper.make_tensor_value_info("float32_in", TensorProto.FLOAT, [1, 10])
        int64_input = helper.make_tensor_value_info("int64_in", TensorProto.INT64, [1, 10])
        float16_output = helper.make_tensor_value_info("float16_out", TensorProto.FLOAT16, [1, 10])
        int32_output = helper.make_tensor_value_info("int32_out", TensorProto.INT32, [1, 10])

        node1 = helper.make_node("Identity", ["float32_in"], ["float16_out"])
        node2 = helper.make_node("Identity", ["int64_in"], ["int32_out"])

        graph = helper.make_graph(
            [node1, node2],
            "test_graph",
            [float32_input, int64_input],
            [float16_output, int32_output],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        config = get_io_config(model)

        assert config["input_names"] == ["float32_in", "int64_in"]
        assert config["output_names"] == ["float16_out", "int32_out"]
        assert config["input_types"][0] == np.float32
        assert config["input_types"][1] == np.int64
        assert config["output_types"][0] == np.float16
        assert config["output_types"][1] == np.int32

    def test_scalar_tensors(self) -> None:
        """Test model with scalar (0-dimensional) tensors."""
        x_info = helper.make_tensor_value_info("scalar_in", TensorProto.FLOAT, [])
        y_info = helper.make_tensor_value_info("scalar_out", TensorProto.FLOAT, [])

        node = helper.make_node("Identity", ["scalar_in"], ["scalar_out"])
        graph = helper.make_graph([node], "test_graph", [x_info], [y_info])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        config = get_io_config(model)

        assert config["input_shapes"] == [[]]
        assert config["output_shapes"] == [[]]

    def test_1d_tensors(self) -> None:
        """Test model with 1D tensors."""
        x_info = helper.make_tensor_value_info("vector_in", TensorProto.FLOAT, [128])
        y_info = helper.make_tensor_value_info("vector_out", TensorProto.FLOAT, [64])

        node = helper.make_node("Identity", ["vector_in"], ["vector_out"])
        graph = helper.make_graph([node], "test_graph", [x_info], [y_info])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        config = get_io_config(model)

        assert config["input_shapes"] == [[128]]
        assert config["output_shapes"] == [[64]]

    def test_mixed_static_dynamic_dims(self) -> None:
        """Test model with mix of static and dynamic dimensions."""
        # Common pattern: dynamic batch, static spatial dims
        x_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["batch", 3, "height", "width"]
        )
        y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 1000])

        node = helper.make_node("Identity", ["input"], ["output"])
        graph = helper.make_graph([node], "test_graph", [x_info], [y_info])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        config = get_io_config(model)

        # Named dynamic dims should be None
        input_shape = config["input_shapes"][0]
        assert input_shape[0] is None  # batch
        assert input_shape[1] == 3  # static
        assert input_shape[2] is None  # height
        assert input_shape[3] is None  # width

    def test_classification_model_pattern(self) -> None:
        """Test typical image classification model I/O pattern."""
        # pixel_values -> logits pattern
        pixel_values = helper.make_tensor_value_info(
            "pixel_values", TensorProto.FLOAT, [1, 3, 224, 224]
        )
        logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1000])

        node = helper.make_node("Identity", ["pixel_values"], ["logits"])
        graph = helper.make_graph([node], "classifier", [pixel_values], [logits])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        config = get_io_config(model)

        assert "pixel_values" in config["input_names"]
        assert "logits" in config["output_names"]
        assert config["input_shapes"][0] == [1, 3, 224, 224]
        assert config["output_shapes"][0] == [1, 1000]

    def test_text_model_pattern(self) -> None:
        """Test typical text/NLP model I/O pattern."""
        # input_ids, attention_mask -> last_hidden_state pattern
        input_ids = helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, ["batch", "seq_len"]
        )
        attention_mask = helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch", "seq_len"]
        )
        hidden_state = helper.make_tensor_value_info(
            "last_hidden_state", TensorProto.FLOAT, ["batch", "seq_len", 768]
        )

        node = helper.make_node("Identity", ["input_ids"], ["last_hidden_state"])
        graph = helper.make_graph([node], "encoder", [input_ids, attention_mask], [hidden_state])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        config = get_io_config(model)

        assert config["input_names"] == ["input_ids", "attention_mask"]
        assert config["output_names"] == ["last_hidden_state"]
        assert config["input_types"][0] == np.int64
        assert config["input_types"][1] == np.int64
        assert config["output_types"][0] == np.float32

    def test_returns_correct_structure(self) -> None:
        """Test that returned dict has all expected keys."""
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])

        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph([node], "test", [x_info], [y_info])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        config = get_io_config(model)

        expected_keys = {
            "input_names",
            "input_shapes",
            "input_symbolic_shapes",
            "input_types",
            "output_names",
            "output_shapes",
            "output_types",
        }
        assert set(config.keys()) == expected_keys

    def test_lists_have_matching_lengths(self) -> None:
        """Test that input/output lists have consistent lengths."""
        in_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1])
        in_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1])
        out_c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [1])

        node1 = helper.make_node("Identity", ["a"], ["c"])
        graph = helper.make_graph([node1], "test", [in_a, in_b], [out_c])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        config = get_io_config(model)

        # Input lists should all have same length
        num_inputs = len(config["input_names"])
        assert len(config["input_shapes"]) == num_inputs
        assert len(config["input_types"]) == num_inputs

        # Output lists should all have same length
        num_outputs = len(config["output_names"])
        assert len(config["output_shapes"]) == num_outputs
        assert len(config["output_types"]) == num_outputs
