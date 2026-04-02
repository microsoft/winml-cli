# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for ORTFusionPipe that are slow.

Extracted from tests/unit/optim/pipes/test_pipe_fusion.py.
"""

import onnx
import pytest
from onnx import TensorProto, helper

from winml.modelkit.optim.pipes import ORTFusionPipe, ORTFusionPipeConfig


def _make_simple_model() -> onnx.ModelProto:
    """Create a minimal ONNX model with a single Add op."""
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
    add_node = helper.make_node("Add", ["input1", "input2"], ["output"], name="add_node")
    graph = helper.make_graph([add_node], "test_graph", [input1, input2], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


@pytest.mark.slow
class TestORTFusionPipeProcessSlow:
    """Slow tests for ORTFusionPipe processing."""

    def test_process_different_model_types(self) -> None:
        """Test processing with different model types."""
        model = _make_simple_model()
        pipe = ORTFusionPipe()

        # Test common model types
        for model_type in ["bert", "gpt2", "t5"]:
            config = ORTFusionPipeConfig(model_type=model_type)
            result = pipe.process(model, config)
            assert isinstance(result, onnx.ModelProto)
