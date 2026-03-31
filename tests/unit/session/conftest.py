# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared fixtures for WinMLSession tests.

EP Markers:
    Use @pytest.mark.ep("qnn") or @pytest.mark.ep("openvino") to mark tests
    that require a specific execution provider. Tests will be skipped if the
    EP is not available.

    Example:
        @pytest.mark.ep("qnn")
        def test_qnn_inference(self, simple_matmul_onnx):
            session = WinMLSession(onnx_path=simple_matmul_onnx, device="npu")
            ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path
import onnx
import pytest
from onnx import TensorProto, helper


# =============================================================================
# EP MARKERS - Skip tests if required EP is not available
# =============================================================================

# EP name mapping: marker name -> ORT provider name
EP_NAME_MAP = {
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "directml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "tensorrt_rtx": "NvTensorRTRTXExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}


def pytest_configure(config: pytest.Config) -> None:
    """Register EP markers."""
    config.addinivalue_line(
        "markers",
        "ep(name): mark test to run only when specific EP is available "
        "(qnn, openvino, directml, cuda, tensorrt, tensorrt_rtx, vitisai, coreml, rocm)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests if required EP is not available.

    Note: EP discovery is only performed when tests with @pytest.mark.ep() markers
    are found. This avoids slow WinML initialization for pure unit tests.
    """
    # Only consider non-e2e items with EP markers. E2e tests handle their own
    # EP discovery. This hook runs before -m filtering, so e2e items are still
    # in the list — skip them to avoid triggering WinML SDK initialization.
    items_with_ep_markers = [
        item
        for item in items
        if any(item.iter_markers(name="ep")) and not any(item.iter_markers(name="e2e"))
    ]
    if not items_with_ep_markers:
        return  # No non-e2e EP markers, skip expensive WinML discovery

    import onnxruntime as ort

    from winml.modelkit.session import WinMLEPRegistry

    # Register WinML EPs so ort.get_ep_devices() includes them
    registry = WinMLEPRegistry.get_instance()
    registry.register_to_ort()

    # Use ort.get_ep_devices() for hardware-accurate availability (only returns
    # EPs backed by actual hardware, not just library-present registrations).
    try:
        available_providers = {d.ep_name for d in ort.get_ep_devices()}
    except Exception:
        available_providers = set()
    # Only add CPUExecutionProvider as guaranteed fallback.
    # Do NOT add all ORT providers here — get_ort_available_providers() returns
    # library-present EPs (e.g., DmlExecutionProvider) even without hardware.
    available_providers.add("CPUExecutionProvider")

    for item in items_with_ep_markers:
        for marker in item.iter_markers(name="ep"):
            ep_name = marker.args[0] if marker.args else None
            if ep_name is None:
                continue

            provider_name = EP_NAME_MAP.get(ep_name.lower())
            if provider_name is None:
                item.add_marker(pytest.mark.skip(reason=f"Unknown EP marker: {ep_name}"))
            elif provider_name not in available_providers:
                item.add_marker(pytest.mark.skip(reason=f"EP not available: {provider_name}"))


def create_matmul_onnx(output_path: Path) -> Path:
    """
    Create a simple MatMul ONNX model for testing.

    Graph: A @ B = C
    Where A is input (1, 4), B is constant (4, 4), C is output (1, 4)

    This is the simplest possible ONNX model that can run on any EP.
    """
    # Input tensor info
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])  # noqa: N806

    # Output tensor info
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])  # noqa: N806

    # Constant weights (4x4 matrix)
    np.random.seed(42)  # Reproducible
    B_values = np.random.randn(4, 4).astype(np.float32)  # noqa: N806
    B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [4, 4], B_values.flatten().tolist())  # noqa: N806

    # MatMul node
    matmul_node = helper.make_node("MatMul", ["A", "B"], ["C"], name="matmul")

    # Graph
    graph = helper.make_graph(
        nodes=[matmul_node],
        name="test_matmul",
        inputs=[A],
        outputs=[C],
        initializer=[B_tensor],
    )

    # Model with explicit IR version for ORT compatibility
    # Use opset 13 and IR version 7 for broad compatibility
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    # Set IR version explicitly (ORT supports up to IR version 9)
    model.ir_version = 7

    # Validate
    onnx.checker.check_model(model)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    return output_path


@pytest.fixture
def simple_matmul_onnx(tmp_path: Path) -> Path:
    """Create simple MatMul ONNX model for testing."""
    return create_matmul_onnx(tmp_path / "test_matmul.onnx")


def create_static_batch_onnx(output_path: Path, batch_size: int = 1) -> Path:
    """Create ONNX model with static batch size for re-batching tests.

    Graph: A @ B = C
    Where A is input (batch_size, 4), B is constant (4, 4), C is output (batch_size, 4)

    Args:
        output_path: Where to save the model
        batch_size: Static batch size (default: 1)
    """
    # Input tensor info with STATIC batch size
    a_input = helper.make_tensor_value_info("A", TensorProto.FLOAT, [batch_size, 4])

    # Output tensor info with STATIC batch size
    c_output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [batch_size, 4])

    # Constant weights (4x4 matrix)
    np.random.seed(42)  # Reproducible
    b_values = np.random.randn(4, 4).astype(np.float32)
    b_tensor = helper.make_tensor("B", TensorProto.FLOAT, [4, 4], b_values.flatten().tolist())

    # MatMul node
    matmul_node = helper.make_node("MatMul", ["A", "B"], ["C"], name="matmul")

    # Graph
    graph = helper.make_graph(
        nodes=[matmul_node],
        name="test_static_batch_matmul",
        inputs=[a_input],
        outputs=[c_output],
        initializer=[b_tensor],
    )

    # Model
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 7

    # Validate
    onnx.checker.check_model(model)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    return output_path


@pytest.fixture
def static_batch1_onnx(tmp_path: Path) -> Path:
    """Create ONNX model with static batch=1 for re-batching tests."""
    return create_static_batch_onnx(tmp_path / "static_batch1.onnx", batch_size=1)


@pytest.fixture
def static_batch2_onnx(tmp_path: Path) -> Path:
    """Create ONNX model with static batch=2 for re-batching tests."""
    return create_static_batch_onnx(tmp_path / "static_batch2.onnx", batch_size=2)


@pytest.fixture
def sample_input() -> dict[str, np.ndarray]:
    """Create sample input for MatMul model."""
    np.random.seed(123)
    return {"A": np.random.randn(1, 4).astype(np.float32)}
