"""Generate minimal ONNX test fixtures for static analyzer testing.

This script creates minimal valid ONNX models used in unit and integration tests.
Each model represents a specific test scenario.

Usage:
    python tests/fixtures/create_test_models.py
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def create_simple_conv_model() -> onnx.ModelProto:
    """Create a minimal Conv model for basic runtime check testing.

    Model structure:
    - Input: [1, 3, 224, 224] (NCHW format)
    - Conv2D: 3 input channels -> 64 output channels, 3x3 kernel
    - Output: [1, 64, 222, 222]

    This model tests:
    - Basic Conv operator support
    - Operator attribute extraction (kernel_shape, pads, strides)
    - Single operator pattern matching
    """
    # Define inputs
    input_tensor = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 3, 224, 224],
    )

    # Define outputs
    output_tensor = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 64, 222, 222],
    )

    # Create Conv weight initializer (64, 3, 3, 3)
    # Shape: [out_channels, in_channels, kernel_h, kernel_w]
    conv_weight = helper.make_tensor(
        name="conv_weight",
        data_type=TensorProto.FLOAT,
        dims=[64, 3, 3, 3],
        vals=[0.1] * (64 * 3 * 3 * 3),  # Dummy weights
    )

    # Create Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv_weight"],
        outputs=["output"],
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    # Create graph
    graph = helper.make_graph(
        nodes=[conv_node],
        name="SimpleConv",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[conv_weight],
    )

    # Create model with opset 13 (widely supported)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="ModelKit Test Fixture Generator",
    )

    # Check model validity
    onnx.checker.check_model(model)

    return model


def create_multi_op_model() -> onnx.ModelProto:
    """Create a model with multiple operators for comprehensive testing.

    Model structure:
    - Input: [1, 3, 224, 224]
    - Conv2D: 3 -> 64 channels
    - Add: bias addition
    - Relu: activation
    - MatMul: linear transformation
    - Output: [1, 100]

    This model tests:
    - Multiple operator type extraction
    - Sequential operator pattern
    - Mixed operator support classification
    """
    # Input
    input_tensor = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 3, 224, 224],
    )

    # Output
    output_tensor = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 100],
    )

    # Initializers
    conv_weight = helper.make_tensor(
        "conv_weight",
        TensorProto.FLOAT,
        [64, 3, 3, 3],
        [0.1] * (64 * 3 * 3 * 3),
    )

    bias = helper.make_tensor(
        "bias",
        TensorProto.FLOAT,
        [1, 64, 1, 1],
        [0.01] * 64,
    )

    reshape_shape = helper.make_tensor(
        "reshape_shape",
        TensorProto.INT64,
        [2],
        [1, 64 * 222 * 222],
    )

    matmul_weight = helper.make_tensor(
        "matmul_weight",
        TensorProto.FLOAT,
        [64 * 222 * 222, 100],
        [0.001] * (64 * 222 * 222 * 100),
    )

    # Nodes
    conv_node = helper.make_node(
        "Conv",
        ["input", "conv_weight"],
        ["conv_output"],
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    add_node = helper.make_node(
        "Add",
        ["conv_output", "bias"],
        ["add_output"],
    )

    relu_node = helper.make_node(
        "Relu",
        ["add_output"],
        ["relu_output"],
    )

    reshape_node = helper.make_node(
        "Reshape",
        ["relu_output", "reshape_shape"],
        ["reshape_output"],
    )

    matmul_node = helper.make_node(
        "MatMul",
        ["reshape_output", "matmul_weight"],
        ["output"],
    )

    # Create graph
    graph = helper.make_graph(
        nodes=[conv_node, add_node, relu_node, reshape_node, matmul_node],
        name="MultiOp",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[conv_weight, bias, reshape_shape, matmul_weight],
    )

    # Create model
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="ModelKit Test Fixture Generator",
    )

    onnx.checker.check_model(model)

    return model


def main() -> None:
    """Generate all test fixture models."""
    fixtures_dir = Path(__file__).parent
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ONNX test fixtures...")

    # Generate simple_conv.onnx
    simple_conv = create_simple_conv_model()
    simple_conv_path = fixtures_dir / "simple_conv.onnx"
    onnx.save(simple_conv, str(simple_conv_path))
    print(f"✓ Created {simple_conv_path}")

    # Generate multi_op.onnx
    multi_op = create_multi_op_model()
    multi_op_path = fixtures_dir / "multi_op.onnx"
    onnx.save(multi_op, str(multi_op_path))
    print(f"✓ Created {multi_op_path}")

    print("\nAll test fixtures generated successfully!")


if __name__ == "__main__":
    main()
