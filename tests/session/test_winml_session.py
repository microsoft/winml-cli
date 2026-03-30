# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession tests with simple ONNX model.

Test Scope (as per design):
1. Instantiate WinMLSession with PREFER_NPU policy (no explicit EP names)
2. Verify session providers match WinMLEPRegistry ground truth
3. Test basic inference works

Key Principle:
- Use policy-based device selection (device="npu", "gpu", "cpu", "auto")
- Never use explicit EP names like "QNNExecutionProvider"
- Use WinMLEPRegistry.get_available_eps() as ground truth
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path
import pytest

from winml.modelkit.compiler.configs import EPConfig
from winml.modelkit.session import WinMLSession
from winml.modelkit.session.ep_registry import WinMLEPRegistry
from winml.modelkit.session.session import SessionState


class TestWinMLSessionInstantiation:
    """Test WinMLSession instantiation with policy-based device selection."""

    def test_session_init_with_npu_device(self, simple_matmul_onnx: Path):
        """
        Test that WinMLSession can be initialized with device='npu'.

        This uses policy-based selection (PREFER_NPU), not explicit EP names.
        """
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="npu",
        )

        assert session.device == "npu"
        assert session.state == SessionState.INITIALIZED
        assert not session.is_compiled

    def test_session_init_with_auto_device(self, simple_matmul_onnx: Path):
        """Test that WinMLSession can be initialized with device='auto'."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="auto",
        )

        assert session.device == "auto"
        assert session.state == SessionState.INITIALIZED

    def test_session_init_with_cpu_device(self, simple_matmul_onnx: Path):
        """Test that WinMLSession can be initialized with device='cpu'."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        assert session.device == "cpu"
        assert session.state == SessionState.INITIALIZED

    def test_session_init_file_not_found(self, tmp_path: Path):
        """Test that WinMLSession raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            WinMLSession(
                onnx_path=tmp_path / "nonexistent.onnx",
                device="auto",
            )


class TestWinMLSessionCompilation:
    """Test WinMLSession compilation (EPContext creation)."""

    @pytest.mark.skip(reason="Lazy init design is not implemented in source code.")
    def test_compile_creates_epcontext(self, simple_matmul_onnx: Path):
        """
        Test that compile() creates EPContext file.

        With new lazy init design:
        - compile() creates EPContext file only
        - _init_session() (called by run()) creates InferenceSession
        """
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="npu",
        )

        # Compile creates EPContext file
        session.compile()

        # EPContext file should exist
        ctx_path = simple_matmul_onnx.parent / f"{simple_matmul_onnx.stem}_ctx.onnx"
        assert ctx_path.exists(), f"EPContext not created: {ctx_path}"

        # Session not created yet (lazy init)
        assert not session.is_compiled

    def test_compile_is_idempotent(self, simple_matmul_onnx: Path):
        """Test that calling compile() multiple times is safe (idempotent)."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        # First compile creates EPContext
        session.compile()
        ctx_path = simple_matmul_onnx.parent / f"{simple_matmul_onnx.stem}_cpu_ctx.onnx"
        first_mtime = ctx_path.stat().st_mtime

        # Second compile should skip (fresh EPContext exists)
        session.compile()
        second_mtime = ctx_path.stat().st_mtime

        # File should not be recreated
        assert first_mtime == second_mtime

    def test_run_uses_epcontext_after_compile(self, simple_matmul_onnx: Path):
        """Test that run() uses EPContext if compile() was called first."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="auto",
        )

        # Pre-compile to create EPContext
        session.compile()

        # Run should use EPContext and create session
        sample_input = {"A": np.random.randn(1, 4).astype(np.float32)}
        session.run(sample_input)

        # Now session should be compiled
        assert session.is_compiled
        assert session.state == SessionState.COMPILED


class TestWinMLSessionProviders:
    """Test that session providers match WinML EP registry."""

    def test_providers_are_valid_and_include_fallback(self, simple_matmul_onnx: Path):
        """
        Test that session providers are valid and include CPU fallback.

        With policy-based selection (PREFER_NPU), ORT dynamically selects
        providers at session creation time. The key requirements are:
        1. Session initializes successfully
        2. At least one provider is active
        3. CPUExecutionProvider is available as fallback

        Note: WinML-registered EPs (like QNN) may be selected dynamically
        via set_provider_selection_policy() even if not listed in
        get_available_providers() beforehand.
        """
        # Create session with NPU preference
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="npu",  # PREFER_NPU policy
        )

        # Run to trigger lazy init
        sample_input = {"A": np.random.randn(1, 4).astype(np.float32)}
        session.run(sample_input)

        # Get actual providers used by session
        actual_providers = session._session.get_providers()

        # Must have at least one provider
        assert len(actual_providers) > 0, "Session must have at least one provider"

        # CPUExecutionProvider should always be present as fallback
        assert "CPUExecutionProvider" in actual_providers, (
            f"CPUExecutionProvider not in providers: {actual_providers}"
        )

        # Log which providers are being used (useful for debugging)
        # On NPU-capable systems, should see QNNExecutionProvider or similar
        print(f"Active providers: {actual_providers}")

    def test_cpu_provider_always_available(self, simple_matmul_onnx: Path):
        """Test that CPUExecutionProvider is always available as fallback."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        # Run to trigger lazy init
        sample_input = {"A": np.random.randn(1, 4).astype(np.float32)}
        session.run(sample_input)

        providers = session._session.get_providers()
        assert "CPUExecutionProvider" in providers

    def test_winml_registry_ep_discovery(self):
        """Test that WinMLEPRegistry can discover EPs (may be empty on non-Windows)."""
        registry = WinMLEPRegistry.get_instance()

        # Registry should be accessible
        assert registry is not None

        # winml_available indicates if WinML SDK is present
        # This may be False on non-Windows or without WinML SDK
        if registry.winml_available:
            eps = registry.get_available_eps()
            # If WinML is available, should have at least one EP
            assert len(eps) > 0, "WinML available but no EPs discovered"


class TestWinMLSessionInference:
    """Test WinMLSession inference execution."""

    def test_basic_inference(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test basic inference with MatMul model."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="auto",
        )

        # Run inference (auto-compiles)
        outputs = session.run(sample_input)

        # Check output
        assert "C" in outputs
        assert outputs["C"].shape == (1, 4)
        assert outputs["C"].dtype == np.float32

    def test_inference_auto_compiles(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that run() auto-compiles if not compiled."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="auto",
        )

        assert not session.is_compiled

        # Run should trigger auto-compile
        outputs = session.run(sample_input)

        assert session.is_compiled
        assert "C" in outputs

    def test_inference_with_torch_tensor(
        self,
        simple_matmul_onnx: Path,
    ):
        """Test inference with torch.Tensor input (converted to numpy)."""
        pytest.importorskip("torch")
        import torch

        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="auto",
        )

        # Create torch tensor input
        torch_input = {"A": torch.randn(1, 4)}

        # Run inference (should convert to numpy internally)
        outputs = session.run(torch_input)

        assert "C" in outputs
        assert outputs["C"].shape == (1, 4)

    def test_inference_empty_input_raises(self, simple_matmul_onnx: Path):
        """Test that empty input raises ValueError."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="auto",
        )

        with pytest.raises(ValueError, match="inputs cannot be empty"):
            session.run({})


class TestWinMLSessionStateManagement:
    """Test WinMLSession state machine."""

    def test_state_transitions(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test state transitions: INITIALIZED -> COMPILED -> INFERRING -> COMPILED."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="auto",
        )

        # Initial state
        assert session.state == SessionState.INITIALIZED

        # After run (lazy init triggers session creation)
        session.run(sample_input)
        assert session.state == SessionState.COMPILED

        # Run again (should return to COMPILED)
        session.run(sample_input)
        assert session.state == SessionState.COMPILED

    def test_reset_returns_to_initialized(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that reset() returns session to INITIALIZED state."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="auto",
        )

        # Run to create session
        session.run(sample_input)
        assert session.is_compiled

        session.reset()
        assert session.state == SessionState.INITIALIZED
        assert not session.is_compiled


class TestWinMLSessionMetadata:
    """Test WinMLSession metadata methods."""

    def test_io_config_before_session_init(
        self,
        simple_matmul_onnx: Path,
    ):
        """Test that io_config is available before session initialization."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="auto",
        )

        # io_config should work without session (reads ONNX directly)
        io_cfg = session.io_config

        assert io_cfg["input_names"] == ["A"]
        assert io_cfg["output_names"] == ["C"]
        assert io_cfg["input_shapes"] == [[1, 4]]


@pytest.mark.skip(reason="Re-batching not yet implemented")
class TestWinMLSessionReBatching:
    """Test re-batching for static batch size models."""

    def test_rebatch_splits_large_batch(self, static_batch1_onnx: Path):
        """Test that batch > model_batch triggers re-batching (lines 343-363)."""
        session = WinMLSession(
            onnx_path=static_batch1_onnx,
            device="cpu",
        )

        # Model expects batch=1, we send batch=3
        large_input = {"A": np.random.randn(3, 4).astype(np.float32)}
        outputs = session.run(large_input)

        # Output should have batch=3 (concatenated from 3 runs of batch=1)
        assert "C" in outputs
        assert outputs["C"].shape == (3, 4)

    def test_rebatch_with_batch2_model(self, static_batch2_onnx: Path):
        """Test re-batching with batch=2 model and batch=4 input (exact multiple)."""
        session = WinMLSession(
            onnx_path=static_batch2_onnx,
            device="cpu",
        )

        # Model expects batch=2, we send batch=4 (exact multiple)
        # Should split into: [2, 2] -> 2 runs
        large_input = {"A": np.random.randn(4, 4).astype(np.float32)}
        outputs = session.run(large_input)

        # Output should have batch=4 (concatenated)
        assert "C" in outputs
        assert outputs["C"].shape == (4, 4)

    def test_rebatch_preserves_values(self, static_batch1_onnx: Path):
        """Test that re-batched outputs are numerically correct."""
        session = WinMLSession(
            onnx_path=static_batch1_onnx,
            device="cpu",
        )

        # Create known input
        np.random.seed(123)
        input_data = np.random.randn(3, 4).astype(np.float32)

        # Run with re-batching (batch=3 on batch=1 model)
        outputs = session.run({"A": input_data})

        # Run each row individually and compare
        for i in range(3):
            session.reset()
            single_output = session.run({"A": input_data[i : i + 1]})
            np.testing.assert_allclose(
                outputs["C"][i : i + 1],
                single_output["C"],
                rtol=1e-5,
                err_msg=f"Re-batched output[{i}] doesn't match single inference",
            )

    def test_no_rebatch_when_batch_fits(self, static_batch2_onnx: Path):
        """Test that batch <= model_batch runs directly without splitting."""
        session = WinMLSession(
            onnx_path=static_batch2_onnx,
            device="cpu",
        )

        # Model expects batch=2, we send batch=2 (exact fit)
        exact_input = {"A": np.random.randn(2, 4).astype(np.float32)}
        outputs = session.run(exact_input)

        assert "C" in outputs
        assert outputs["C"].shape == (2, 4)

    def test_batch_smaller_than_model_fails(self, static_batch2_onnx: Path):
        """Test that batch < model_batch fails with static batch model.

        ORT with static batch models requires exact batch size match.
        Sending batch=1 to a batch=2 model raises INVALID_ARGUMENT.
        """
        from winml.modelkit.session.session import InferenceError

        session = WinMLSession(
            onnx_path=static_batch2_onnx,
            device="cpu",
        )

        # Model expects batch=2, we send batch=1 (smaller) - ORT rejects this
        small_input = {"A": np.random.randn(1, 4).astype(np.float32)}

        with pytest.raises(InferenceError, match="INVALID_ARGUMENT"):
            session.run(small_input)


class TestWinMLSessionErrorState:
    """Test error state handling (line 312)."""

    def test_run_in_error_state_raises(self, simple_matmul_onnx: Path):
        """Test that run() raises InferenceError when session is in error state."""
        from winml.modelkit.session.session import InferenceError

        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        # Trigger first run to initialize session
        sample = {"A": np.random.randn(1, 4).astype(np.float32)}
        session.run(sample)

        # Manually set error state
        session._state = SessionState.ERROR
        session._last_error = RuntimeError("Test error")

        # Run should raise InferenceError
        with pytest.raises(InferenceError, match="Session in error state"):
            session.run(sample)

    def test_reset_clears_error_state(self, simple_matmul_onnx: Path):
        """Test that reset() clears error state and allows re-run."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        # Run, then set error state
        sample = {"A": np.random.randn(1, 4).astype(np.float32)}
        session.run(sample)
        session._state = SessionState.ERROR
        session._last_error = RuntimeError("Test error")

        # Reset should clear error
        session.reset()
        assert session.state == SessionState.INITIALIZED
        assert session._last_error is None

        # Should be able to run again
        outputs = session.run(sample)
        assert "C" in outputs


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
                "directml",
                "gpu",
                "DmlExecutionProvider",
                marks=pytest.mark.ep("directml"),
            ),
            pytest.param(
                "cuda",
                "gpu",
                "CUDAExecutionProvider",
                marks=pytest.mark.ep("cuda"),
            ),
            pytest.param(
                "tensorrt",
                "gpu",
                "TensorrtExecutionProvider",
                marks=pytest.mark.ep("tensorrt"),
            ),
            pytest.param(
                "tensorrt_rtx",
                "gpu",
                "NvTensorRTRTXExecutionProvider",
                marks=pytest.mark.ep("tensorrt_rtx"),
            ),
            pytest.param(
                "vitisai",
                "npu",
                "VitisAIExecutionProvider",
                marks=pytest.mark.ep("vitisai"),
            ),
            pytest.param("rocm", "gpu", "ROCMExecutionProvider", marks=pytest.mark.ep("rocm")),
        ],
        ids=["qnn", "openvino", "directml", "cuda", "tensorrt", "tensorrt_rtx", "vitisai", "rocm"],
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

        # Verify expected EP is being used
        providers = session._session.get_providers()
        assert provider_name in providers, (
            f"{ep_name} EP ({provider_name}) not in providers: {providers}"
        )
        assert "C" in outputs
        assert outputs["C"].shape == (1, 4)


class TestWinMLSessionExplicitProviders:
    """Test explicit provider specification (bypassing policy-based selection).

    Uses `providers` parameter to explicitly specify EPs instead of device policy.

    Note: Explicit providers only work with natively registered EPs (those in
    ort.get_available_providers()). WinML-registered EPs (QNN, OpenVINO via WinML)
    must use policy-based selection (device="npu") instead.
    """

    def test_explicit_cpu_provider(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test explicit CPU provider works without EP marker."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
            ep_config=EPConfig(provider="cpu", provider_options={"CPUExecutionProvider": {}}),
        )

        outputs = session.run(sample_input)

        providers = session._session.get_providers()
        assert "CPUExecutionProvider" in providers
        assert "C" in outputs

    def test_explicit_provider_with_options(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test ep_config.provider_options dict is passed correctly."""
        # CPU provider doesn't need options, but we verify the parameter works
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
            ep_config=EPConfig(provider="cpu", provider_options={"CPUExecutionProvider": {}}),
        )

        outputs = session.run(sample_input)
        assert "C" in outputs


class TestWinMLSessionPerfTracking:
    """Test WinMLSession performance tracking with context manager."""

    def test_perf_disabled_by_default(self, simple_matmul_onnx: Path):
        """Test that performance tracking is disabled by default."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        assert session.perf_stats is None

    def test_perf_context_manager_returns_stats(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that perf() context manager returns PerfStats."""
        from winml.modelkit.session import PerfStats

        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        with session.perf() as stats:
            assert stats is not None
            assert isinstance(stats, PerfStats)
            assert stats.count == 0

    def test_perf_records_samples(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that inference runs are recorded within context."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        with session.perf() as stats:
            for _ in range(5):
                session.run(sample_input)

            assert stats.count == 5
            assert len(stats.samples_ms) == 5
            assert all(t > 0 for t in stats.samples_ms)

    def test_perf_stats_computed_correctly(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that computed stats are correct."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        with session.perf() as stats:
            for _ in range(10):
                session.run(sample_input)

        assert stats.count == 10
        assert stats.total_ms > 0
        assert stats.mean_ms > 0
        assert stats.min_ms > 0
        assert stats.max_ms >= stats.min_ms
        assert stats.p50_ms > 0
        assert stats.p90_ms >= stats.p50_ms
        assert stats.p99_ms >= stats.p90_ms

    def test_perf_warmup_excludes_samples(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test warmup parameter excludes first N samples."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        with session.perf(warmup=3) as stats:
            for _ in range(10):
                session.run(sample_input)

        # 10 total, 3 warmup = 7 effective
        assert stats.total_count == 10
        assert stats.count == 7
        assert len(stats.samples_ms) == 7
        assert len(stats.all_samples_ms) == 10

    def test_perf_disabled_after_context(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that perf tracking is disabled after context exits."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        with session.perf() as stats:
            session.run(sample_input)
            assert stats.count == 1

        # After context, perf_stats should be None
        assert session.perf_stats is None

        # Running outside context should not record
        session.run(sample_input)
        # stats object still has data from context
        assert stats.count == 1

    def test_perf_output_not_affected(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that perf tracking doesn't affect inference output."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        # Run without tracking
        output_no_perf = session.run(sample_input)

        # Run with tracking
        with session.perf():
            output_with_perf = session.run(sample_input)

        # Outputs should be identical
        np.testing.assert_array_equal(output_no_perf["C"], output_with_perf["C"])

    def test_perf_stats_accessible_after_context(
        self,
        simple_matmul_onnx: Path,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that stats object remains accessible after context."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            device="cpu",
        )

        with session.perf(warmup=2) as stats:
            for _ in range(5):
                session.run(sample_input)

        # Stats still accessible after context
        assert stats.count == 3  # 5 - 2 warmup
        assert stats.mean_ms > 0
        assert stats.p99_ms > 0
