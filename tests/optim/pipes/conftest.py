"""Pipe-specific fixtures for pipe implementation tests.

This conftest provides fixtures specific to Pipe class testing.
For general optim fixtures, see tests/optim/conftest.py.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


# Re-export commonly-needed fixtures from parent conftest
# These are automatically available to tests in this directory


# =============================================================================
# BUILDERS
# =============================================================================


def build_causal_mask_model(
    seq_len: int = 16,
    mask_value: float = -3.4028235e38,
) -> onnx.ModelProto:
    """Build a causal mask model with extreme float constants.

    This model structure is based on real attention mask patterns from transformer
    models like CLIP. The causal_mask and mask_value initializers contain extreme
    values (torch.finfo(float32).min ≈ -3.4e38) that cause quantization issues.

    Model structure:
        attention_mask [1, seq_len]
            -> Unsqueeze (axes=[1])
            -> Unsqueeze (axes=[2])
            -> Cast (to bool)
            -> Where(condition, causal_mask, mask_value)
        -> output [1, 1, seq_len, seq_len]

    Args:
        seq_len: Sequence length for the causal mask (default: 16)
        mask_value: Value for masked positions (default: -3.4e38)

    Returns:
        ONNX model with causal mask pattern
    """
    # Build causal mask (lower triangular with zeros, upper with mask_value)
    causal_mask_values = np.triu(
        np.full((seq_len, seq_len), mask_value, dtype=np.float32), k=1
    )
    causal_mask_values = causal_mask_values.reshape(1, 1, seq_len, seq_len)

    # Initializers
    causal_mask_init = numpy_helper.from_array(causal_mask_values, "causal_mask.1")
    mask_value_init = numpy_helper.from_array(
        np.array(mask_value, dtype=np.float32), "mask_value"
    )

    # Input/Output
    input_tensor = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, [1, seq_len]
    )
    output_tensor = helper.make_tensor_value_info(
        "causal_mask", TensorProto.FLOAT, [1, 1, seq_len, seq_len]
    )

    # Nodes
    nodes = [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["/Constant_output_0"],
            name="/Constant",
            value=helper.make_tensor("axes_1", TensorProto.INT64, [1], [1]),
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=["attention_mask", "/Constant_output_0"],
            outputs=["/Unsqueeze_output_0"],
            name="/Unsqueeze",
        ),
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["/Constant_1_output_0"],
            name="/Constant_1",
            value=helper.make_tensor("axes_2", TensorProto.INT64, [1], [2]),
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=["/Unsqueeze_output_0", "/Constant_1_output_0"],
            outputs=["/Unsqueeze_1_output_0"],
            name="/Unsqueeze_1",
        ),
        helper.make_node(
            "Cast",
            inputs=["/Unsqueeze_1_output_0"],
            outputs=["/Cast_output_0"],
            name="/Cast",
            to=TensorProto.BOOL,
        ),
        helper.make_node(
            "Where",
            inputs=["/Cast_output_0", "causal_mask.1", "mask_value"],
            outputs=["causal_mask"],
            name="/Where",
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="main_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[causal_mask_init, mask_value_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def build_model_with_normal_constants() -> onnx.ModelProto:
    """Build a model with only normal float constants (no extreme values)."""
    normal_values = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
    normal_const = numpy_helper.from_array(normal_values, "normal_const")

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 2])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])

    add_node = helper.make_node(
        "Add", inputs=["input", "normal_const"], outputs=["output"], name="add"
    )

    graph = helper.make_graph(
        nodes=[add_node],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[normal_const],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def causal_mask_model() -> onnx.ModelProto:
    """Create causal mask model with extreme float constants (-3.4e38)."""
    return build_causal_mask_model()


@pytest.fixture
def model_with_normal_constants() -> onnx.ModelProto:
    """Create a model with only normal float constants (no extreme values)."""
    return build_model_with_normal_constants()
