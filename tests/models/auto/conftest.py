# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Pytest fixtures for WinML AutoModel tests.

This module provides common fixtures for testing the modelkit/models/winml/ implementation
following the design specifications in docs/design/automodel/.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


if TYPE_CHECKING:
    from collections.abc import Generator


# Test model IDs - small real models for integration testing
# Per design: use small real models (microsoft/resnet-18, prajjwal1/bert-tiny)
TEST_IMAGE_CLASSIFICATION_MODEL = "microsoft/resnet-18"
TEST_SEQUENCE_CLASSIFICATION_MODEL = "prajjwal1/bert-tiny"
TEST_IMAGE_SEGMENTATION_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

# Fallback tiny models for faster unit tests
TINY_IMAGE_MODEL = "hf-internal-testing/tiny-random-convnext"
TINY_TEXT_MODEL = "hf-internal-testing/tiny-random-BertModel"


@pytest.fixture(scope="session")
def test_cache_dir() -> Generator[Path, None, None]:
    """
    Create a temporary cache directory for test artifacts.

    Cleaned up after test session.
    """
    cache_dir = Path(tempfile.mkdtemp(prefix="winml_test_cache_"))
    yield cache_dir

    # Cleanup after session
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for individual tests."""
    return tmp_path


@pytest.fixture
def simple_matmul_onnx(tmp_path: Path) -> Path:
    """
    Create a simple MatMul ONNX model for basic testing.

    Graph: A @ B = C
    Where A is input (1, 4), B is constant (4, 4), C is output (1, 4)

    This follows the test design from ARCHITECTURE_PRINCIPLES.md Section 8.
    """
    # Input
    inp_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])

    # Output
    out_c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])

    # Constant weights
    b_values = np.random.randn(4, 4).astype(np.float32)
    b_tensor = helper.make_tensor("B", TensorProto.FLOAT, [4, 4], b_values.flatten())

    # MatMul node
    matmul_node = helper.make_node("MatMul", ["A", "B"], ["C"], name="matmul")

    # Graph
    graph = helper.make_graph(
        [matmul_node],
        "test_matmul",
        [inp_a],
        [out_c],
        [b_tensor],
    )

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    # Save
    output_path = tmp_path / "test_matmul.onnx"
    onnx.save(model, str(output_path))

    return output_path


@pytest.fixture
def sample_matmul_input() -> dict[str, np.ndarray]:
    """Sample input for simple_matmul_onnx model."""
    return {"A": np.random.randn(1, 4).astype(np.float32)}


@pytest.fixture
def image_classification_onnx(tmp_path: Path) -> Path:
    """
    Create a simple image classification ONNX model.

    Graph: Conv -> Flatten -> MatMul -> logits
    Input: pixel_values (1, 3, 224, 224)
    Output: logits (1, 1000)
    """
    # Input
    pixel_values = helper.make_tensor_value_info(
        "pixel_values", TensorProto.FLOAT, [1, 3, 224, 224]
    )

    # Output
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1000])

    # Simple conv weights (just for structure, not real weights)
    conv_w = helper.make_tensor(
        "conv_w", TensorProto.FLOAT, [64, 3, 7, 7],
        np.random.randn(64, 3, 7, 7).astype(np.float32).flatten()
    )

    # FC weights
    fc_w = helper.make_tensor(
        "fc_w", TensorProto.FLOAT, [50176, 1000],  # 64*28*28 = 50176
        np.random.randn(50176, 1000).astype(np.float32).flatten()
    )

    # Nodes
    nodes = [
        helper.make_node("Conv", ["pixel_values", "conv_w"], ["conv_out"],
                        kernel_shape=[7, 7], strides=[8, 8], pads=[3, 3, 3, 3]),
        helper.make_node("Flatten", ["conv_out"], ["flat_out"], axis=1),
        helper.make_node("MatMul", ["flat_out", "fc_w"], ["logits"]),
    ]

    # Graph
    graph = helper.make_graph(
        nodes,
        "image_classification",
        [pixel_values],
        [logits],
        [conv_w, fc_w],
    )

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    # Save
    output_path = tmp_path / "image_classification.onnx"
    onnx.save(model, str(output_path))

    return output_path


@pytest.fixture
def sample_image_input() -> dict[str, np.ndarray]:
    """Sample input for image classification models."""
    return {"pixel_values": np.random.randn(1, 3, 224, 224).astype(np.float32)}


@pytest.fixture
def sequence_classification_onnx(tmp_path: Path) -> Path:
    """
    Create a simple sequence classification ONNX model.

    Graph: Embedding -> Reduce -> MatMul -> logits
    Input: input_ids (1, 128), attention_mask (1, 128)
    Output: logits (1, 2)
    """
    # Inputs
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, [1, 128]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, [1, 128]
    )

    # Output
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])

    # Embedding weights (vocab_size=1000, hidden_size=64)
    embed_w = helper.make_tensor(
        "embed_w", TensorProto.FLOAT, [1000, 64],
        np.random.randn(1000, 64).astype(np.float32).flatten()
    )

    # FC weights
    fc_w = helper.make_tensor(
        "fc_w", TensorProto.FLOAT, [64, 2],
        np.random.randn(64, 2).astype(np.float32).flatten()
    )

    # Nodes
    nodes = [
        # Cast input_ids to int32 for Gather
        helper.make_node("Cast", ["input_ids"], ["input_ids_i32"], to=TensorProto.INT32),
        # Gather embeddings
        helper.make_node("Gather", ["embed_w", "input_ids_i32"], ["embeddings"], axis=0),
        # Mean pooling (axis=1)
        helper.make_node("ReduceMean", ["embeddings"], ["pooled"], axes=[1]),
        # Classification head
        helper.make_node("MatMul", ["pooled", "fc_w"], ["logits"]),
    ]

    # Graph
    graph = helper.make_graph(
        nodes,
        "sequence_classification",
        [input_ids, attention_mask],
        [logits],
        [embed_w, fc_w],
    )

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    # Save
    output_path = tmp_path / "sequence_classification.onnx"
    onnx.save(model, str(output_path))

    return output_path


@pytest.fixture
def sample_text_input() -> dict[str, np.ndarray]:
    """Sample input for sequence classification models."""
    return {
        "input_ids": np.random.randint(0, 1000, (1, 128)).astype(np.int64),
        "attention_mask": np.ones((1, 128), dtype=np.int64),
    }


# Skip markers for conditional tests
requires_npu = pytest.mark.skipif(
    os.environ.get("WINML_TEST_NPU", "0") != "1",
    reason="NPU tests require WINML_TEST_NPU=1 and NPU hardware"
)

requires_gpu = pytest.mark.skipif(
    os.environ.get("WINML_TEST_GPU", "0") != "1",
    reason="GPU tests require WINML_TEST_GPU=1 and GPU hardware"
)

requires_hf_hub = pytest.mark.skipif(
    os.environ.get("WINML_TEST_OFFLINE", "0") == "1",
    reason="HF Hub tests disabled in offline mode"
)


@pytest.fixture
def mock_hf_config() -> dict[str, Any]:
    """Mock HuggingFace config for testing without network."""
    return {
        "_name_or_path": "test-model",
        "architectures": ["ConvNextForImageClassification"],
        "model_type": "convnext",
        "num_labels": 1000,
        "id2label": {"0": "class_0", "1": "class_1"},
        "label2id": {"class_0": 0, "class_1": 1},
        "hidden_sizes": [96, 192, 384, 768],
        "winml": {
            "export": {
                "opset_version": 17,
                "batch_size": 1,
            },
            "quantization": {
                "enabled": False,
            },
            "optimization": {
                "gelu_fusion": True,
            },
            "compile": {
                "device": "auto",
                "persist_jit": False,
            },
        },
    }
