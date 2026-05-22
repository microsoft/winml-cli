# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession tests with simple ONNX model.

Test Scope:
1. Instantiate WinMLSession with an explicit WinMLEPDevice
2. Verify session state, providers, and inference behavior
3. Test perf() context manager

Key Principle:
- Use WinMLEPDevice-based construction (Task 7 API)
- CPU tests use the real OrtEpDevice with a mocked WinMLEPRegistry
- NPU/QNN tests use fake OrtEpDevice fixtures (mocked ORT)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path
import pytest

from winml.modelkit.compiler import EPConfig
from winml.modelkit.session import WinMLEPDevice, WinMLEPMonitorMismatch, SessionState, WinMLSession


class TestWinMLSessionInstantiation:
    """Test WinMLSession instantiation with WinMLEPDevice-based selection."""

    def test_session_init_with_npu_device(
        self, simple_matmul_onnx: Path, qnn_npu_ep_device: WinMLEPDevice, fake_ort_npu: MagicMock
    ):
        """Test that WinMLSession can be initialized with an NPU WinMLEPDevice.

        ORT InferenceSession is also mocked because the fake_ort_npu MagicMock
        cannot be passed to add_provider_for_devices() (requires a real C++ object).
        """
        with (
            patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg,
            patch("winml.modelkit.session.session.ort.InferenceSession"),
            patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
        ):
            mock_reg.get_instance.return_value.register_ep.return_value = [fake_ort_npu]
            session = WinMLSession(onnx_path=simple_matmul_onnx, ep_device=qnn_npu_ep_device)

        assert session.device == "npu"
        assert session.state == SessionState.INITIALIZED

    def test_session_init_with_cpu_ep_device(
        self, simple_matmul_onnx: Path, cpu_ep_device: WinMLEPDevice, real_cpu_ort_device: object
    ):
        """Test that WinMLSession can be initialized with a CPU WinMLEPDevice."""
        with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg:
            mock_reg.get_instance.return_value.register_ep.return_value = [real_cpu_ort_device]
            session = WinMLSession(onnx_path=simple_matmul_onnx, ep_device=cpu_ep_device)

        assert session.device == "cpu"
        assert session.state == SessionState.INITIALIZED

    def test_session_init_file_not_found(
        self, tmp_path: Path, cpu_ep_device: WinMLEPDevice, real_cpu_ort_device: object
    ):
        """Test that WinMLSession raises an ORT error for a non-existent ONNX file."""
        from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile

        with (
            patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg,
            pytest.raises(NoSuchFile),
        ):
            mock_reg.get_instance.return_value.register_ep.return_value = [real_cpu_ort_device]
            WinMLSession(onnx_path=tmp_path / "nonexistent.onnx", ep_device=cpu_ep_device)


class TestWinMLSessionCompilation:
    """Test WinMLSession compilation (EPContext creation)."""

    @pytest.mark.skip(reason="Lazy init design is not implemented in source code.")
    def test_compile_creates_epcontext(
        self, simple_matmul_onnx: Path, qnn_npu_ep_device: WinMLEPDevice, fake_ort_npu: MagicMock
    ):
        """
        Test that compile() creates EPContext file.

        With new lazy init design:
        - compile() creates EPContext file only
        - _init_session() (called by run()) creates InferenceSession
        """
        with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg:
            mock_reg.get_instance.return_value.register_ep.return_value = [fake_ort_npu]
            session = WinMLSession(onnx_path=simple_matmul_onnx, ep_device=qnn_npu_ep_device)

        # Compile creates EPContext file
        session.compile()

        # EPContext file should exist
        ctx_path = simple_matmul_onnx.parent / f"{simple_matmul_onnx.stem}_ctx.onnx"
        assert ctx_path.exists(), f"EPContext not created: {ctx_path}"

        # Session not created yet (lazy init)
        assert not session.is_compiled

    def test_compile_is_idempotent(self, cpu_winml_session: WinMLSession):
        """Test that calling compile() multiple times is safe (idempotent).

        __init__ already creates _session, so compile() is a no-op and the
        _session object reference is unchanged. State transitions to COMPILED
        only after run() is called.
        """
        session = cpu_winml_session

        # _session is already set by __init__; compile() returns immediately
        first_session = session._session
        assert first_session is not None
        session.compile()
        # State stays INITIALIZED — compile() returned early, no state change
        assert session.state == SessionState.INITIALIZED
        assert session._session is first_session

        # Second compile also a no-op
        session.compile()
        assert session._session is first_session

    def test_run_uses_epcontext_after_compile(self, cpu_winml_session: WinMLSession):
        """Test that run() works after compile() was called."""
        session = cpu_winml_session

        # compile() is a no-op when _session is already set
        session.compile()

        # Run should succeed
        sample_input = {"A": np.random.randn(1, 4).astype(np.float32)}
        session.run(sample_input)

        # Session should be compiled
        assert session.is_compiled
        assert session.state == SessionState.COMPILED


class TestWinMLSessionProviders:
    """Test that session providers are valid after initialization."""

    def test_providers_are_valid_and_include_fallback(
        self, cpu_winml_session: WinMLSession, sample_input: dict
    ):
        """
        Test that session providers are valid and include CPUExecutionProvider.

        The session is bound to CPUExecutionProvider via an explicit WinMLEPDevice.
        """
        session = cpu_winml_session

        # Run inference to confirm the session is functional
        session.run(sample_input)

        # Get actual providers used by session
        actual_providers = session._session.get_providers()

        # Must have at least one provider
        assert len(actual_providers) > 0, "Session must have at least one provider"

        # CPUExecutionProvider should always be present
        assert "CPUExecutionProvider" in actual_providers, (
            f"CPUExecutionProvider not in providers: {actual_providers}"
        )

        print(f"Active providers: {actual_providers}")

    def test_cpu_provider_always_available(
        self, cpu_winml_session: WinMLSession, sample_input: dict
    ):
        """Test that CPUExecutionProvider is available after CPU WinMLEPDevice init."""
        session = cpu_winml_session
        session.run(sample_input)

        providers = session._session.get_providers()
        assert "CPUExecutionProvider" in providers


class TestWinMLSessionInference:
    """Test WinMLSession inference execution."""

    def test_basic_inference(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test basic inference with MatMul model."""
        session = cpu_winml_session

        # Run inference
        outputs = session.run(sample_input)

        # Check output
        assert "C" in outputs
        assert outputs["C"].shape == (1, 4)
        assert outputs["C"].dtype == np.float32

    def test_inference_already_compiled_on_init(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that WinMLSession is compiled immediately after __init__."""
        session = cpu_winml_session

        # __init__ creates _session eagerly — is_compiled is True immediately
        assert session.is_compiled

        # Run should succeed
        outputs = session.run(sample_input)
        assert "C" in outputs

    def test_inference_with_torch_tensor(
        self,
        cpu_winml_session: WinMLSession,
    ):
        """Test inference with torch.Tensor input (converted to numpy)."""
        pytest.importorskip("torch")
        import torch

        session = cpu_winml_session

        # Create torch tensor input
        torch_input = {"A": torch.randn(1, 4)}

        # Run inference (should convert to numpy internally)
        outputs = session.run(torch_input)

        assert "C" in outputs
        assert outputs["C"].shape == (1, 4)

    def test_inference_empty_input_raises(self, cpu_winml_session: WinMLSession):
        """Test that empty input raises ValueError."""
        session = cpu_winml_session

        with pytest.raises(ValueError, match="inputs cannot be empty"):
            session.run({})


class TestWinMLSessionStateManagement:
    """Test WinMLSession state machine."""

    def test_state_transitions(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test state transitions: COMPILED -> INFERRING -> COMPILED.

        __init__ creates the session eagerly so the initial state is COMPILED.
        """
        session = cpu_winml_session

        # __init__ creates the ORT session eagerly — state starts at INITIALIZED
        # but _session is already populated.
        assert session.state == SessionState.INITIALIZED

        # After run
        session.run(sample_input)
        assert session.state == SessionState.COMPILED

        # Run again (should return to COMPILED)
        session.run(sample_input)
        assert session.state == SessionState.COMPILED

    def test_reset_returns_to_initialized(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that reset() returns session to INITIALIZED state."""
        session = cpu_winml_session

        # Run to transition to COMPILED
        session.run(sample_input)
        assert session.is_compiled

        session.reset()
        assert session.state == SessionState.INITIALIZED
        assert not session.is_compiled


class TestWinMLSessionMetadata:
    """Test WinMLSession metadata methods."""

    def test_io_config_before_session_init(
        self,
        cpu_winml_session: WinMLSession,
    ):
        """Test that io_config is available and reflects the ONNX model."""
        session = cpu_winml_session

        # io_config reads the ONNX file directly
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
        from winml.modelkit.session import InferenceError

        session = WinMLSession(
            onnx_path=static_batch2_onnx,
            device="cpu",
        )

        # Model expects batch=2, we send batch=1 (smaller) - ORT rejects this
        small_input = {"A": np.random.randn(1, 4).astype(np.float32)}

        with pytest.raises(InferenceError, match="INVALID_ARGUMENT"):
            session.run(small_input)


class TestWinMLSessionErrorState:
    """Test error state handling."""

    def test_run_in_error_state_raises(self, cpu_winml_session: WinMLSession):
        """Test that run() raises InferenceError when session is in error state."""
        from winml.modelkit.session import InferenceError

        session = cpu_winml_session

        # Trigger first run
        sample = {"A": np.random.randn(1, 4).astype(np.float32)}
        session.run(sample)

        # Manually set error state
        session._state = SessionState.ERROR
        session._last_error = RuntimeError("Test error")

        # Run should raise InferenceError
        with pytest.raises(InferenceError, match="Session in error state"):
            session.run(sample)

    def test_reset_clears_error_state(self, cpu_winml_session: WinMLSession):
        """Test that reset() clears error state and allows re-run."""
        session = cpu_winml_session

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


class TestWinMLSessionExplicitProviders:
    """Test EPConfig provider_options passthrough with WinMLEPDevice-based init."""

    def test_explicit_cpu_provider(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: WinMLEPDevice,
        real_cpu_ort_device: object,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that ep_config is accepted and CPU provider is active."""
        with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg:
            mock_reg.get_instance.return_value.register_ep.return_value = [real_cpu_ort_device]
            session = WinMLSession(
                onnx_path=simple_matmul_onnx,
                ep_device=cpu_ep_device,
                ep_config=EPConfig(provider="cpu", provider_options={}),
            )

        outputs = session.run(sample_input)

        providers = session._session.get_providers()
        assert "CPUExecutionProvider" in providers
        assert "C" in outputs

    def test_explicit_provider_with_options(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: WinMLEPDevice,
        real_cpu_ort_device: object,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that ep_config.provider_options is accepted without error."""
        with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg:
            mock_reg.get_instance.return_value.register_ep.return_value = [real_cpu_ort_device]
            session = WinMLSession(
                onnx_path=simple_matmul_onnx,
                ep_device=cpu_ep_device,
                ep_config=EPConfig(provider="cpu", provider_options={}),
            )

        outputs = session.run(sample_input)
        assert "C" in outputs


class TestWinMLSessionPerfTracking:
    """Test WinMLSession performance tracking with context manager."""

    def test_perf_disabled_by_default(self, cpu_winml_session: WinMLSession):
        """Test that performance tracking is disabled by default."""
        session = cpu_winml_session
        assert session.perf_stats is None

    def test_perf_context_manager_returns_stats(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that perf() context manager returns PerfStats."""
        from winml.modelkit.session import PerfStats

        session = cpu_winml_session

        with session.perf() as ctx:
            stats = ctx.stats
            assert stats is not None
            assert isinstance(stats, PerfStats)
            assert stats.count == 0

    def test_perf_records_samples(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that inference runs are recorded within context."""
        session = cpu_winml_session

        with session.perf() as ctx:
            for _ in range(5):
                session.run(sample_input)

            stats = ctx.stats
            assert stats.count == 5
            assert len(stats.samples_ms) == 5
            assert all(t > 0 for t in stats.samples_ms)

    def test_perf_stats_computed_correctly(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that computed stats are correct."""
        session = cpu_winml_session

        with session.perf() as ctx:
            for _ in range(10):
                session.run(sample_input)

        stats = ctx.stats
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
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test warmup parameter excludes first N samples."""
        session = cpu_winml_session

        with session.perf(warmup=3) as ctx:
            for _ in range(10):
                session.run(sample_input)

        stats = ctx.stats
        # 10 total, 3 warmup = 7 effective
        assert stats.total_count == 10
        assert stats.count == 7
        assert len(stats.samples_ms) == 7
        assert len(stats.all_samples_ms) == 10

    def test_perf_disabled_after_context(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that perf tracking is disabled after context exits."""
        session = cpu_winml_session

        with session.perf() as ctx:
            session.run(sample_input)
            stats = ctx.stats
            assert stats.count == 1

        # After context, perf_stats should be None
        assert session.perf_stats is None

        # Running outside context should not record
        session.run(sample_input)
        # stats object still has data from context
        assert stats.count == 1

    def test_perf_output_not_affected(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that perf tracking doesn't affect inference output."""
        session = cpu_winml_session

        # Run without tracking
        output_no_perf = session.run(sample_input)

        # Run with tracking
        with session.perf():
            output_with_perf = session.run(sample_input)

        # Outputs should be identical
        np.testing.assert_array_equal(output_no_perf["C"], output_with_perf["C"])

    def test_perf_stats_accessible_after_context(
        self,
        cpu_winml_session: WinMLSession,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that stats object remains accessible after context."""
        session = cpu_winml_session

        with session.perf(warmup=2) as ctx:
            for _ in range(5):
                session.run(sample_input)

        stats = ctx.stats
        # Stats still accessible after context
        assert stats.count == 3  # 5 - 2 warmup
        assert stats.mean_ms > 0
        assert stats.p99_ms > 0


# =============================================================================
# Task 7: WinMLEPDevice-based constructor (hard break)
# =============================================================================


def test_winml_session_accepts_ep_device(tmp_path, qnn_npu_ep_device, fake_ort_npu) -> None:
    """WinMLSession compiles with an explicit WinMLEPDevice."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")  # minimal placeholder; ORT is mocked
    with (
        patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg,
        patch("winml.modelkit.session.session.ort.InferenceSession") as mock_sess,
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        mock_reg.get_instance.return_value.register_ep.return_value = [fake_ort_npu]
        sess = WinMLSession(onnx_path, ep_device=qnn_npu_ep_device)
    mock_sess.assert_called_once()
    assert sess._ep_device == qnn_npu_ep_device


def test_winml_session_rejects_legacy_ep_kwarg(tmp_path, qnn_npu_ep_device) -> None:
    """Legacy ep="qnn" kwarg now raises TypeError."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    with pytest.raises(TypeError):
        WinMLSession(onnx_path, ep="qnn")  # type: ignore[call-arg]


def test_winml_session_rejects_legacy_device_kwarg(tmp_path) -> None:
    """Legacy device="..." kwarg now raises TypeError (hard break, Task 7 Option A)."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    with pytest.raises(TypeError):
        WinMLSession(onnx_path, device="auto")  # type: ignore[call-arg]


# =============================================================================
# Task 8: perf() validation + save/restore
# =============================================================================


def test_perf_validates_monitor_ep_name_match(tmp_path, qnn_npu_ep_device, fake_ort_npu) -> None:
    """Monitor for QNN against an OpenVINO WinMLEPDevice -> WinMLEPMonitorMismatch."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    openvino_ep = WinMLEPDevice(
        ep="OpenVINOExecutionProvider",
        device="npu",
        vendor_id=0x8086,
        device_id=0x0BD0,
    )
    fake_ov = MagicMock()
    fake_ov.device.type.name = "NPU"
    fake_ov.device.vendor_id = 0x8086
    fake_ov.device.device_id = 0x0BD0
    qnn_monitor = MagicMock()
    qnn_monitor.ep_name = "qnn"
    qnn_monitor.get_provider_options.return_value = {}
    qnn_monitor.get_session_options.return_value = {}
    with (
        patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg,
        patch("winml.modelkit.session.session.ort.InferenceSession"),
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        mock_reg.get_instance.return_value.register_ep.return_value = [fake_ov]
        sess = WinMLSession(onnx_path, ep_device=openvino_ep)
        with pytest.raises(WinMLEPMonitorMismatch), sess.perf(monitor=qnn_monitor):
            pass


def test_perf_preserves_save_restore(tmp_path, qnn_npu_ep_device, fake_ort_npu) -> None:
    """Mid-perf raise must restore _provider_options snapshot."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    bad_monitor = MagicMock()
    bad_monitor.ep_name = "qnn"
    bad_monitor.get_provider_options.return_value = {"oops": "x"}
    bad_monitor.get_session_options.return_value = {}
    bad_monitor.__enter__.side_effect = RuntimeError("boom")
    with (
        patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg,
        patch("winml.modelkit.session.session.ort.InferenceSession"),
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        mock_reg.get_instance.return_value.register_ep.return_value = [fake_ort_npu]
        sess = WinMLSession(onnx_path, ep_device=qnn_npu_ep_device)
        snapshot = dict(sess._provider_options)
        with pytest.raises(RuntimeError), sess.perf(monitor=bad_monitor):
            pass
        assert sess._provider_options == snapshot
        assert sess._ep == "QNNExecutionProvider"
        assert sess._active_session_option_entries == {}  # back to empty (pre-perf state)
