# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for WinMLSession.is_compatible()."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from onnx import TensorProto, helper

from winml.modelkit.session import WinMLSession


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def cpu_session(tmp_path: Path) -> WinMLSession:
    """Create a WinMLSession bound to CPUExecutionProvider using a minimal ONNX model.

    Uses the real OrtEpDevice for CPU so that is_compatible() can call
    ORT with actual shapes. WinMLEPRegistry is mocked to return the real
    CPU OrtEpDevice, bypassing slow PowerShell/WMI queries on CI.
    """
    from unittest.mock import patch

    import onnx
    import onnxruntime as _ort

    from winml.modelkit.session import WinMLEPDevice

    # Discover the real CPU OrtEpDevice so add_provider_for_devices() gets
    # a genuine handle and ORT can run inference.
    cpu_ort_devs = [d for d in _ort.get_ep_devices() if d.ep_name == "CPUExecutionProvider"]
    if not cpu_ort_devs:
        pytest.skip("CPUExecutionProvider not available in ort.get_ep_devices()")
    real_cpu_dev = cpu_ort_devs[0]

    cpu_ep_device = WinMLEPDevice(
        ep="CPUExecutionProvider",
        device="cpu",
        vendor_id=real_cpu_dev.device.vendor_id,
        device_id=real_cpu_dev.device.device_id,
    )

    # Build minimal Relu model
    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph(
        [node],
        "stub",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    model_path = tmp_path / "stub.onnx"
    onnx.save(model, str(model_path))

    # Mock EP registry to return the real CPU OrtEpDevice, avoiding WMI queries
    with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = [real_cpu_dev]
        return WinMLSession(onnx_path=model_path, ep_device=cpu_ep_device)


class TestIsCompatible:
    """Test WinMLSession.is_compatible() utility."""

    def test_relu_compatible_with_cpu(self, cpu_session: WinMLSession) -> None:
        """Relu should be compatible with CPU EP."""
        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
        )

        assert cpu_session.is_compatible(node, graph) is True

    def test_add_compatible_with_cpu(self, cpu_session: WinMLSession) -> None:
        """Add should be compatible with CPU EP."""
        node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])

        graph = helper.make_graph(
            [node],
            "test",
            [
                helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4]),
            ],
            [helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])],
        )

        assert cpu_session.is_compatible(node, graph) is True

    def test_unknown_op_incompatible(self, cpu_session: WinMLSession) -> None:
        """Nonexistent op should be incompatible."""
        node = helper.make_node(
            "CompletelyFakeOp12345",
            inputs=["X"],
            outputs=["Y"],
            domain="fake.domain",
        )

        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
        )

        assert cpu_session.is_compatible(node, graph) is False

    def test_without_graph_context_warns(
        self, cpu_session: WinMLSession, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Calling without graph emits warning and still returns a result."""
        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        with caplog.at_level(logging.WARNING):
            result = cpu_session.is_compatible(node, graph=None)

        assert "dummy shapes" in caplog.text
        # Relu with dummy shapes should still work on CPU
        assert result is True

    def test_without_graph_context_still_works_for_valid_op(
        self, cpu_session: WinMLSession
    ) -> None:
        """Valid op without graph context should still be compatible (less accurate)."""
        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        assert cpu_session.is_compatible(node) is True

    def test_node_with_no_inputs_returns_false(self, cpu_session: WinMLSession) -> None:
        """Node with empty input list should return False."""
        node = helper.make_node("Relu", inputs=[], outputs=["Y"])

        assert cpu_session.is_compatible(node) is False

    def test_node_with_no_outputs_returns_false(self, cpu_session: WinMLSession) -> None:
        """Node with empty output list should return False."""
        node = helper.make_node("Relu", inputs=["X"], outputs=[])

        assert cpu_session.is_compatible(node) is False

    def test_graph_context_resolves_shapes(self, cpu_session: WinMLSession) -> None:
        """When graph is provided, shapes come from graph value_info."""
        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8])
        y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 8])
        graph = helper.make_graph([node], "test", [x_info], [y_info])

        # Should work with the real shapes from graph
        assert cpu_session.is_compatible(node, graph) is True
