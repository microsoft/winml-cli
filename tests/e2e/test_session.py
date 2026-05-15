# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for WinMLSession requiring specific hardware EPs.

Extracted from tests/unit/session/test_winml_session.py.
These tests require specific hardware (NPU, GPU) to run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from winml.modelkit.session import WinMLEPRegistry, WinMLSession


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class TestWinMLRegistryEPDiscovery:
    """Test WinMLEPRegistry EP discovery."""

    @pytest.mark.e2e
    def test_winml_registry_ep_discovery(self):
        """Test that WinMLEPRegistry can discover EPs when WinML SDK is present."""
        registry = WinMLEPRegistry.get_instance()

        # Registry should be accessible
        assert registry is not None

        # Skip if WinML SDK is not available on this environment
        if not registry.winml_available:
            pytest.skip("WinML SDK not available")

        eps = registry.get_available_eps()
        # If WinML is available, should have at least one EP
        assert len(eps) > 0, "WinML available but no EPs discovered"


@pytest.mark.e2e
class TestWinMLSessionEPSpecific:
    """EP-specific tests using @pytest.mark.ep() markers.

    These tests verify EP-specific behavior and are automatically skipped
    if the required EP is not available on the system.
    """

    @pytest.mark.parametrize(
        ("ep_name", "device", "provider_name"),
        [
            pytest.param("qnn", "npu", "QNNExecutionProvider", marks=pytest.mark.ep("qnn")),
            pytest.param(
                "openvino",
                "npu",
                "OpenVINOExecutionProvider",
                marks=pytest.mark.ep("openvino"),
            ),
            pytest.param(
                "dml",
                "gpu",
                "DmlExecutionProvider",
                marks=pytest.mark.ep("dml"),
            ),
            # CUDA support disabled — re-enable when needed.
            # pytest.param(
            #     "cuda",
            #     "gpu",
            #     "CUDAExecutionProvider",
            #     marks=pytest.mark.ep("cuda"),
            # ),
            pytest.param(
                "nv_tensorrt_rtx",
                "gpu",
                "NvTensorRTRTXExecutionProvider",
                marks=pytest.mark.ep("nv_tensorrt_rtx"),
            ),
            pytest.param(
                "vitisai",
                "npu",
                "VitisAIExecutionProvider",
                marks=pytest.mark.ep("vitisai"),
            ),
            pytest.param("rocm", "gpu", "ROCMExecutionProvider", marks=pytest.mark.ep("rocm")),
        ],
        # "cuda" omitted — CUDA support disabled.
        ids=["qnn", "openvino", "dml", "nv_tensorrt_rtx", "vitisai", "rocm"],
    )
    def test_ep_inference(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
        ep_name: str,
        device: str,
        provider_name: str,
    ):
        """Test inference with specific EP."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device=device,
        )

        outputs = session.run(sample_input)

        # With policy-based selection, ORT picks the best EP for the device.
        # Verify inference succeeds and a non-CPU EP is used for gpu/npu devices.
        providers = session._session.get_providers()
        if device != "cpu":
            non_cpu = [p for p in providers if p != "CPUExecutionProvider"]
            assert len(non_cpu) > 0, f"Expected non-CPU EP for device={device}, got: {providers}"
        assert "C" in outputs
        assert outputs["C"].shape == (1, 4)
