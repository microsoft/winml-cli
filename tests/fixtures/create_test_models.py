# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Generate minimal ONNX test fixtures for static analyzer testing.

This script creates minimal valid ONNX models used in unit and integration tests.
Each model represents a specific test scenario.

Usage:
    python tests/fixtures/create_test_models.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
        producer_name="WinML CLI Test Fixture Generator",
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
        producer_name="WinML CLI Test Fixture Generator",
    )

    onnx.checker.check_model(model)

    return model


# Image-segmentation I/O contract shared by HF semantic-segmentation exports
# (e.g. nvidia/segformer-*-ade-*): pixel_values [batch, 3, height, width] ->
# logits [batch, num_labels, height/4, width/4]. 150 = ADE20K class count.
SEG_NUM_CHANNELS = 3
SEG_NUM_LABELS = 150


def create_fake_segmentation_model() -> onnx.ModelProto:
    """Create a tiny FP32 semantic-segmentation model with random weights.

    Stands in for a real HuggingFace semantic-segmentation export (e.g.
    ``nvidia/segformer-b0-finetuned-ade-512-512``) whose heavy backbone can
    randomly hang on QNN hosts during quantization calibration. It keeps the
    same I/O contract so calibration datasets and the quantizer treat it
    identically to the real model:

    - Input:  ``pixel_values`` [batch, 3, height, width] (FLOAT)
    - Output: ``logits`` [batch, num_labels, height/4, width/4] (FLOAT)

    Two stride-2 convs reproduce the ``/4`` logits resolution; a 1x1 conv acts
    as the classifier head. Spatial dims stay dynamic so the model accepts both
    calibration inputs (e.g. 512x512) and a degenerate 1x1 inference probe.
    Weights are seeded-random so regeneration stays deterministic.
    """
    rng = np.random.default_rng(1234)

    pixel_values = helper.make_tensor_value_info(
        "pixel_values",
        TensorProto.FLOAT,
        ["batch_size", SEG_NUM_CHANNELS, "height", "width"],
    )
    logits = helper.make_tensor_value_info(
        "logits",
        TensorProto.FLOAT,
        ["batch_size", SEG_NUM_LABELS, "height_out", "width_out"],
    )

    def _weight(shape: tuple[int, ...], name: str) -> onnx.TensorProto:
        return onnx.numpy_helper.from_array(
            (rng.standard_normal(shape) * 0.1).astype(np.float32), name
        )

    w1 = _weight((8, SEG_NUM_CHANNELS, 3, 3), "seg_W1")
    b1 = _weight((8,), "seg_B1")
    w2 = _weight((16, 8, 3, 3), "seg_W2")
    b2 = _weight((16,), "seg_B2")
    w3 = _weight((SEG_NUM_LABELS, 16, 1, 1), "seg_W3")
    b3 = _weight((SEG_NUM_LABELS,), "seg_B3")

    nodes = [
        helper.make_node(
            "Conv",
            ["pixel_values", "seg_W1", "seg_B1"],
            ["c1"],
            name="Conv_1",
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", ["c1"], ["r1"], name="Relu_1"),
        helper.make_node(
            "Conv",
            ["r1", "seg_W2", "seg_B2"],
            ["c2"],
            name="Conv_2",
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", ["c2"], ["r2"], name="Relu_2"),
        helper.make_node(
            "Conv",
            ["r2", "seg_W3", "seg_B3"],
            ["logits"],
            name="Classifier",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="FakeSegmentation",
        inputs=[pixel_values],
        outputs=[logits],
        initializer=[w1, b1, w2, b2, w3, b3],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="WinML CLI Test Fixture Generator",
    )
    # Match the quantize e2e fixtures (ir_version 8) so onnxruntime's quantizer
    # loads it identically to the other tiny models in that suite.
    model.ir_version = 8

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

    # Generate fake_segmentation.onnx
    fake_segmentation = create_fake_segmentation_model()
    fake_segmentation_path = fixtures_dir / "fake_segmentation.onnx"
    onnx.save(fake_segmentation, str(fake_segmentation_path))
    print(f"✓ Created {fake_segmentation_path}")

    print("\nAll test fixtures generated successfully!")


if __name__ == "__main__":
    main()
