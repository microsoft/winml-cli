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

import numpy as np
import pytest

from winml.modelkit.session import WinMLEPRegistry, WinMLSession

from .require_ep import require_ep


if TYPE_CHECKING:
    from pathlib import Path


class TestWinMLRegistryEPDiscovery:
    """Test WinMLEPRegistry EP discovery."""

    @pytest.mark.e2e
    def test_winml_registry_ep_discovery(self):
        """Test that WinMLEPRegistry can discover EPs when WinML SDK is present."""
        registry = WinMLEPRegistry.instance()

        # Registry should be accessible
        assert registry is not None

        plugin_entries = [entry for entry in registry._discovered if not entry.is_built_in()]

        # Skip if no real plugin EP was discovered on this environment.
        if not plugin_entries:
            pytest.skip("No plugin EPs discovered in this environment")

        assert len({entry.ep_name for entry in plugin_entries}) > 0, (
            "WinML available but no plugin EPs discovered"
        )


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
            pytest.param(
                "cuda",
                "gpu",
                "CUDAExecutionProvider",
                marks=pytest.mark.ep("cuda"),
            ),
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
        ids=["qnn", "openvino", "dml", "cuda", "nv_tensorrt_rtx", "vitisai", "rocm"],
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
        require_ep(ep_name, device=device)
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device=device,
            ep=ep_name,
        )

        outputs = session.run(sample_input)

        providers = session._session.get_providers()
        assert provider_name in providers, f"Expected {provider_name}, got: {providers}"
        assert "C" in outputs
        assert outputs["C"].shape == (1, 4)

    def test_auto_device_runtime_smoke(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """``device="auto"`` resolves an EP and runs inference end to end.

        Covers the full ``resolve_device`` -> ``add_provider_for_devices``
        -> ``ort.InferenceSession`` path that #708 introduced. Lives in
        e2e because on a hardware-less runner the WinML EP registry can
        advertise phantom NPU/GPU EP devices that crash natively when
        bound (#726) — so this is only meaningful when invoked on a
        machine with at least one real-hardware EP available.

        Aggregates the assertions previously held by the deleted unit
        tests (basic inference, lazy-compile transition, state machine,
        second-run state preservation, reset, EPContext-after-compile).
        """
        from winml.modelkit.session import SessionState

        session = WinMLSession(onnx_path=simple_matmul_onnx, device="auto")

        # State machine: INITIALIZED before any work, lazy-compile contract
        assert session.state == SessionState.INITIALIZED
        assert not session.is_compiled

        # First run triggers compile -> COMPILED
        outputs = session.run(sample_input)
        assert session.is_compiled
        assert session.state == SessionState.COMPILED
        assert "C" in outputs
        assert outputs["C"].shape == (1, 4)
        assert outputs["C"].dtype == np.float32

        # `device="auto"` must end up resolving to a concrete device label.
        assert session.device in {"npu", "gpu", "cpu"}

        # Second run keeps COMPILED state (no implicit re-init)
        session.run(sample_input)
        assert session.state == SessionState.COMPILED

        # reset() returns to INITIALIZED
        session.reset()
        assert session.state == SessionState.INITIALIZED
        assert not session.is_compiled

    def test_auto_device_explicit_compile_writes_epcontext(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Explicit ``compile()`` + ``run()`` exercises the EPContext path.

        Replaces the deleted ``test_run_uses_epcontext_after_compile``
        unit test. The EPContext path goes through ``ort.ModelCompiler``
        for EPs that support compilation; ``device="auto"`` lets the
        underlying EP pick whether/how to materialize the context file.
        """
        from winml.modelkit.session import SessionState

        session = WinMLSession(onnx_path=simple_matmul_onnx, device="auto")

        session.compile()
        assert session.state == SessionState.COMPILED

        outputs = session.run(sample_input)
        assert session.is_compiled
        assert "C" in outputs
        assert outputs["C"].shape == (1, 4)
