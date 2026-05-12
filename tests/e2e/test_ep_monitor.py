# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for HWMonitor requiring real NPU hardware.

Extracted from tests/unit/session/test_ep_monitor.py.
"""

from __future__ import annotations

import json
import sys
import time

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
class TestHWMonitorIntegration:
    """Integration tests requiring a real NPU."""

    @pytest.fixture
    def simple_onnx_model(self, tmp_path):
        """Create a minimal ONNX model (MatMul) for NPU integration testing.

        This is a tiny graph that compiles to any NPU EP and runs fast
        enough for integration testing without downloading large models.
        """
        import onnx
        from onnx import TensorProto, helper

        # MatMul: (1, 64) x (64, 32) -> (1, 32)
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [64, 32])
        z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 32])

        matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
        graph = helper.make_graph([matmul], "test_matmul", [x, y], [z])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        model_path = tmp_path / "test_matmul.onnx"
        onnx.save(model, str(model_path))
        return model_path

    def test_npu_monitor_captures_metrics(self, simple_onnx_model):
        """HWMonitor captures metrics during real NPU inference."""
        import numpy as np

        from winml.modelkit.session import HWMonitor, WinMLSession

        if not HWMonitor.is_available():
            pytest.skip("HWMonitor not available (not Windows)")

        # Skip if no NPU — this test needs real NPU inference
        from winml.modelkit.session.monitor._pdh import PdhPoller

        if not PdhPoller.is_npu_available():
            pytest.skip("No NPU detected via PDH — skipping integration test")

        session = WinMLSession(str(simple_onnx_model), device="npu")

        # Generate random inputs matching the model
        inputs = {
            "X": np.random.rand(1, 64).astype(np.float32),
            "Y": np.random.rand(64, 32).astype(np.float32),
        }

        # Warm up session
        session.run(inputs)

        with HWMonitor(poll_interval_ms=50) as hw:
            for _ in range(500):
                session.run(inputs)
            # Give poller time to capture at least one sample
            time.sleep(0.15)

        # Monitor should have collected metrics without errors
        d = hw.to_dict()
        assert d["monitor"] == "HWMonitor"
        assert d["device_kind"] == "npu"  # NPU was discovered
        assert d["adapter_luid"] is not None

        # Verify JSON serializability
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

        # Memory should be detected (even if utilization is 0 for fast ops)
        assert isinstance(hw.peak_memory_mb, float)
        assert isinstance(hw.mean_utilization_pct, float)
