# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession tests with simple ONNX model.

Test Scope:
1. Instantiate WinMLSession with an explicit EPDeviceTarget
2. Verify session state, providers, and inference behavior
3. Test perf() context manager

Key Principle:
- Use EPDeviceTarget-based construction (Task 7 API)
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
from winml.modelkit.session import (
    EPDeviceTarget,
    WinMLEPDevice,
    WinMLEPMonitorMismatch,
    WinMLSession,
)
from winml.modelkit.session.session import SessionState


def _stub_registry(monkeypatch: pytest.MonkeyPatch, ep_device: object) -> MagicMock:
    """Provide the public registry contract for ergonomic session construction."""
    from winml.modelkit.session.ep_registry import WinMLEPRegistry

    registry = MagicMock()
    registry.auto_device.return_value = ep_device
    registry.available_eps.return_value = frozenset(
        {getattr(getattr(ep_device, "device", None), "ep_name", "CPUExecutionProvider")}
    )
    monkeypatch.setattr(WinMLEPRegistry, "instance", classmethod(lambda _cls: registry))
    return registry


class TestWinMLSessionInstantiation:
    """Test WinMLSession instantiation with EPDeviceTarget-based selection."""

    def test_session_init_with_npu_device(
        self, simple_matmul_onnx: Path, qnn_npu_ep_device: EPDeviceTarget, fake_ort_npu: MagicMock
    ):
        """Test that WinMLSession can be initialized with an NPU WinMLEPDevice.

        ORT InferenceSession is also mocked because the fake_ort_npu MagicMock
        cannot be passed to add_provider_for_devices() (requires a real C++ object).
        """
        with (
            patch("winml.modelkit.session.session.ort.InferenceSession"),
            patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
        ):
            session = WinMLSession(onnx_path=simple_matmul_onnx, ep_device=qnn_npu_ep_device)

        assert session.device == "npu"
        assert session.state == SessionState.INITIALIZED

    def test_session_init_with_cpu_ep_device(
        self, simple_matmul_onnx: Path, cpu_ep_device: EPDeviceTarget
    ):
        """Test that WinMLSession can be initialized with a CPU WinMLEPDevice."""
        session = WinMLSession(onnx_path=simple_matmul_onnx, ep_device=cpu_ep_device)

        assert session.device == "cpu"
        assert session.state == SessionState.INITIALIZED

    def test_session_init_file_not_found(self, tmp_path: Path, cpu_ep_device: EPDeviceTarget):
        """Test that WinMLSession raises an ORT error for a non-existent ONNX file."""
        from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile

        with pytest.raises(NoSuchFile):
            WinMLSession(onnx_path=tmp_path / "nonexistent.onnx", ep_device=cpu_ep_device)

    def test_ep_name_is_none_before_compile(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """ep_name returns None before compile() since no providers are bound yet."""
        _stub_registry(monkeypatch, cpu_ep_device)
        session = WinMLSession(onnx_path=simple_matmul_onnx, device="cpu", ep="cpu")
        assert session.ep_name is None

    def test_ep_name_after_compile(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """ep_name returns the primary provider name once the session is built."""
        _stub_registry(monkeypatch, cpu_ep_device)
        session = WinMLSession(onnx_path=simple_matmul_onnx, device="cpu", ep="cpu")
        session.compile()
        assert isinstance(session.ep_name, str)
        assert session.ep_name.endswith("ExecutionProvider")

    def test_explicit_ep_cpu_binds_cpu_execution_provider(self, simple_matmul_onnx: Path):
        """`--ep cpu` must bind CPUExecutionProvider explicitly.

        Regression: previously the explicit-EP branch carried a
        `self._ep != "cpu"` exception, so `ep="cpu"` fell through to
        PREFER_CPU policy. On systems with OpenVINO (or any other
        CPU-capable EP) registered, ORT then chose OV-on-CPU as the primary
        and silently ignored the user's `--ep cpu` choice. The fix routes
        `ep="cpu"` through `add_provider_for_devices` like any other EP, so
        the resulting session has CPUExecutionProvider as the primary
        provider regardless of what else is registered.
        """
        session = WinMLSession(onnx_path=simple_matmul_onnx, device="cpu", ep="cpu")
        session.compile()
        assert session.ep_name == "CPUExecutionProvider"
        assert session._session.get_providers()[0] == "CPUExecutionProvider"


class TestWinMLSessionCompilation:
    """Test WinMLSession compilation (EPContext creation)."""

    @pytest.mark.skip(reason="Lazy init design is not implemented in source code.")
    def test_compile_creates_epcontext(
        self, simple_matmul_onnx: Path, qnn_npu_ep_device: EPDeviceTarget, fake_ort_npu: MagicMock
    ):
        """
        Test that compile() creates EPContext file.

        With new lazy init design:
        - compile() creates EPContext file only
        - _init_session() (called by run()) creates InferenceSession
        """
        with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg:
            mock_reg.instance.return_value.register_ep.return_value = [fake_ort_npu]
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

    def test_runtime_compile_bypasses_model_compiler(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Lazy runtime sessions construct ORT directly without AOT artifacts."""
        _stub_registry(monkeypatch, cpu_ep_device)
        with (
            patch("winml.modelkit.session.session.ort.InferenceSession") as inference_session,
            patch("winml.modelkit.session.session.ort.ModelCompiler") as model_compiler,
        ):
            session = WinMLSession(onnx_path=simple_matmul_onnx, device="cpu")
            assert session._session is None

            session.compile()

        inference_session.assert_called_once()
        model_compiler.assert_not_called()
        assert not (simple_matmul_onnx.parent / "compile.log").exists()

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

        The session is bound to CPUExecutionProvider via an explicit EPDeviceTarget.
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
        """Test that CPUExecutionProvider is available after CPU EPDeviceTarget init.

        Also pins the `is_compiled` -> run() -> `is_compiled` behavior and the
        output dtype check (assertions ported from deleted device='auto' tests;
        no other test pins these on device='cpu').
        """
        session = cpu_winml_session
        outputs = session.run(sample_input)

        assert session.is_compiled
        providers = session._session.get_providers()
        assert "CPUExecutionProvider" in providers
        assert outputs["C"].dtype == np.float32


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


class TestWinMLSessionPrecisionDetection:
    """Test `_get_precision` estimation across the detection ladder."""

    @staticmethod
    def _save(model, path: Path) -> Path:
        from onnx import save

        save(model, str(path))
        return path

    def test_precision_fp32_from_initializers(
        self, simple_matmul_onnx: Path, cpu_ep_device: WinMLEPDevice
    ):
        """Float initializers (fp32) → 'fp32'."""
        session = WinMLSession(onnx_path=simple_matmul_onnx, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "fp32"

    def test_precision_fp16_from_initializers(self, tmp_path: Path, cpu_ep_device: WinMLEPDevice):
        """Float initializers (fp16) → 'fp16'."""
        import numpy as np
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, 4])
        c = helper.make_tensor_value_info("C", TensorProto.FLOAT16, [1, 4])
        b_vals = np.random.randn(4, 4).astype(np.float16)
        b = helper.make_tensor("B", TensorProto.FLOAT16, [4, 4], b_vals.tobytes(), raw=True)
        node = helper.make_node("MatMul", ["A", "B"], ["C"])
        graph = helper.make_graph([node], "fp16", [a], [c], [b])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        path = self._save(model, tmp_path / "fp16.onnx")

        session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "fp16"

    def test_precision_int8_from_qdq(self, tmp_path: Path, cpu_ep_device: WinMLEPDevice):
        """QDQ pair with int8 zero_point on a weight initializer → 'int8'."""
        import numpy as np
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
        c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])

        w_q = helper.make_tensor(
            "W_q",
            TensorProto.INT8,
            [4, 4],
            np.zeros((4, 4), dtype=np.int8).tobytes(),
            raw=True,
        )
        w_scale = helper.make_tensor("W_scale", TensorProto.FLOAT, [], [0.1])
        w_zp = helper.make_tensor(
            "W_zp", TensorProto.INT8, [], np.array([0], dtype=np.int8).tobytes(), raw=True
        )

        dq = helper.make_node("DequantizeLinear", ["W_q", "W_scale", "W_zp"], ["W"], name="dq")
        matmul = helper.make_node("MatMul", ["A", "W"], ["C"], name="mm")

        graph = helper.make_graph([dq, matmul], "qdq_int8", [a], [c], [w_q, w_scale, w_zp])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        path = self._save(model, tmp_path / "qdq_int8.onnx")

        session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "w8a8"

    def test_precision_w8a16_mixed_qdq(self, tmp_path: Path, cpu_ep_device: WinMLEPDevice):
        """Activation quantized to uint16 + weight to int8 → 'w8a16'."""
        import numpy as np
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
        c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])

        # Activation Q→DQ with uint16 zero_point (dynamic input → activation side)
        a_scale = helper.make_tensor("A_scale", TensorProto.FLOAT, [], [0.05])
        a_zp = helper.make_tensor(
            "A_zp",
            TensorProto.UINT16,
            [],
            np.array([0], dtype=np.uint16).tobytes(),
            raw=True,
        )
        q_act = helper.make_node("QuantizeLinear", ["A", "A_scale", "A_zp"], ["A_q"], name="q_act")
        dq_act = helper.make_node(
            "DequantizeLinear", ["A_q", "A_scale", "A_zp"], ["A_d"], name="dq_act"
        )

        # Weight DQ with int8 zero_point (initializer → weight side)
        w_q = helper.make_tensor(
            "W_q",
            TensorProto.INT8,
            [4, 4],
            np.zeros((4, 4), dtype=np.int8).tobytes(),
            raw=True,
        )
        w_scale = helper.make_tensor("W_scale", TensorProto.FLOAT, [], [0.1])
        w_zp = helper.make_tensor(
            "W_zp", TensorProto.INT8, [], np.array([0], dtype=np.int8).tobytes(), raw=True
        )
        dq_w = helper.make_node("DequantizeLinear", ["W_q", "W_scale", "W_zp"], ["W"], name="dq_w")

        matmul = helper.make_node("MatMul", ["A_d", "W"], ["C"], name="mm")

        graph = helper.make_graph(
            [q_act, dq_act, dq_w, matmul],
            "qdq_w8a16",
            [a],
            [c],
            [a_scale, a_zp, w_q, w_scale, w_zp],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        path = self._save(model, tmp_path / "qdq_w8a16.onnx")

        # Precision detection is a static read of the QDQ graph; the uint16
        # activation zero-point is a valid w8a16 signal but ORT's CPU EP rejects
        # it at compile time, so mock the InferenceSession — we only assert on
        # the statically-derived io_config precision, not on a runnable session.
        with patch("winml.modelkit.session.session.ort.InferenceSession"):
            session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "w8a16"

    def test_precision_int8_ignores_int32_bias_zp(
        self, tmp_path: Path, cpu_ep_device: WinMLEPDevice
    ):
        """INT32 bias DQ on the weight side must not poison the label.

        Mirrors the NPU-quantized ResNet-50 case: every Conv has an
        INT8-weight DQ alongside an INT32-bias DQ. The bias is a quant
        accumulator, not a weight, so it must be excluded from weight-side
        bit-width counting; otherwise the result becomes 'w32a8'.
        """
        import numpy as np
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
        c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])

        # Activation Q→DQ with UINT8 zero_point
        a_scale = helper.make_tensor("A_scale", TensorProto.FLOAT, [], [0.05])
        a_zp = helper.make_tensor(
            "A_zp", TensorProto.UINT8, [], np.array([0], dtype=np.uint8).tobytes(), raw=True
        )
        q_act = helper.make_node("QuantizeLinear", ["A", "A_scale", "A_zp"], ["A_q"], name="q_act")
        dq_act = helper.make_node(
            "DequantizeLinear", ["A_q", "A_scale", "A_zp"], ["A_d"], name="dq_act"
        )

        # Weight DQ with INT8 zero_point (initializer → weight side)
        w_q = helper.make_tensor(
            "W_q",
            TensorProto.INT8,
            [4, 4],
            np.zeros((4, 4), dtype=np.int8).tobytes(),
            raw=True,
        )
        w_scale = helper.make_tensor("W_scale", TensorProto.FLOAT, [], [0.1])
        w_zp = helper.make_tensor(
            "W_zp", TensorProto.INT8, [], np.array([0], dtype=np.int8).tobytes(), raw=True
        )
        dq_w = helper.make_node("DequantizeLinear", ["W_q", "W_scale", "W_zp"], ["W"], name="dq_w")

        # Bias DQ with INT32 zero_point (initializer → would be classified
        # weight-side; this is the node that previously poisoned the label).
        b_q = helper.make_tensor(
            "B_q", TensorProto.INT32, [4], np.zeros(4, dtype=np.int32).tobytes(), raw=True
        )
        b_scale = helper.make_tensor("B_scale", TensorProto.FLOAT, [], [0.005])
        b_zp = helper.make_tensor(
            "B_zp", TensorProto.INT32, [], np.array([0], dtype=np.int32).tobytes(), raw=True
        )
        dq_b = helper.make_node("DequantizeLinear", ["B_q", "B_scale", "B_zp"], ["B"], name="dq_b")

        matmul = helper.make_node("MatMul", ["A_d", "W"], ["MM"], name="mm")
        add = helper.make_node("Add", ["MM", "B"], ["C"], name="add_bias")

        graph = helper.make_graph(
            [q_act, dq_act, dq_w, dq_b, matmul, add],
            "qdq_with_int32_bias",
            [a],
            [c],
            [a_scale, a_zp, w_q, w_scale, w_zp, b_q, b_scale, b_zp],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        path = self._save(model, tmp_path / "qdq_int32_bias.onnx")

        session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "w8a8"

    def test_precision_matmulnbits_w4a16(self, tmp_path: Path, cpu_ep_device: WinMLEPDevice):
        """MatMulNBits with bits=4 + fp16 initializers → 'w4a16'."""
        import numpy as np
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, 32])
        c = helper.make_tensor_value_info("C", TensorProto.FLOAT16, [1, 16])

        # MatMulNBits packed-weight + scales (dummy shapes — schema doesn't validate)
        w_packed = helper.make_tensor(
            "W",
            TensorProto.UINT8,
            [16, 1, 16],
            np.zeros((16, 1, 16), dtype=np.uint8).tobytes(),
            raw=True,
        )
        scales = helper.make_tensor(
            "scales",
            TensorProto.FLOAT16,
            [16],
            np.ones(16, dtype=np.float16).tobytes(),
            raw=True,
        )

        node = helper.make_node(
            "MatMulNBits",
            ["A", "W", "scales"],
            ["C"],
            domain="com.microsoft",
            K=32,
            N=16,
            bits=4,
            block_size=32,
        )

        graph = helper.make_graph([node], "mmnbits_w4", [a], [c], [w_packed, scales])
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )
        model.ir_version = 7
        path = self._save(model, tmp_path / "mmnbits_w4.onnx")

        session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] == "w4a16"

    def test_precision_no_signal_returns_none(self, tmp_path: Path, cpu_ep_device: WinMLEPDevice):
        """No QDQ ops, no MatMulNBits, no float initializers → None."""
        from onnx import TensorProto, helper

        a = helper.make_tensor_value_info("A", TensorProto.INT64, [1, 4])
        c = helper.make_tensor_value_info("C", TensorProto.INT64, [1, 4])

        identity = helper.make_node("Identity", ["A"], ["C"])
        graph = helper.make_graph([identity], "no_signal", [a], [c])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        path = self._save(model, tmp_path / "no_signal.onnx")

        session = WinMLSession(onnx_path=path, ep_device=cpu_ep_device)
        assert session.io_config["precision"] is None


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
    """Test error state handling."""

    def test_run_in_error_state_raises(self, cpu_winml_session: WinMLSession):
        """Test that run() raises InferenceError when session is in error state."""
        from winml.modelkit.session.session import InferenceError

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
    """Test EPConfig provider_options passthrough with EPDeviceTarget-based init."""

    def test_explicit_cpu_provider(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that ep_config is accepted and CPU provider is active."""
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
        cpu_ep_device: EPDeviceTarget,
        sample_input: dict[str, np.ndarray],
    ):
        """Test that ep_config.provider_options is accepted without error."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=cpu_ep_device,
            ep_config=EPConfig(provider="cpu", provider_options={}),
        )

        outputs = session.run(sample_input)

        providers = session._session.get_providers()
        assert "CPUExecutionProvider" in providers
        assert "C" in outputs

    def test_ep_config_provider_options_forwarded(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        sample_input: dict[str, np.ndarray],
    ):
        """Verify ep_config.provider_options is forwarded to add_provider_for_devices."""
        import onnxruntime as ort

        options = {"arbitrary_key": "arbitrary_value"}
        captured: list[dict[str, str]] = []
        real_method = ort.SessionOptions.add_provider_for_devices

        def spy(self_sess, ep_devices, provider_opts):
            captured.append(dict(provider_opts))
            return real_method(self_sess, ep_devices, provider_opts)

        with patch.object(ort.SessionOptions, "add_provider_for_devices", spy):
            session = WinMLSession(
                onnx_path=simple_matmul_onnx,
                ep_device=cpu_ep_device,
                ep_config=EPConfig(provider="cpu", provider_options=options),
            )
            outputs = session.run(sample_input)

        assert options in captured, f"provider_options not forwarded; got calls with: {captured}"
        assert "C" in outputs

    def test_runtime_provider_options_forwarded(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
        sample_input: dict[str, np.ndarray],
    ):
        """Runtime ``provider_options`` kwarg is forwarded to add_provider_for_devices."""
        import onnxruntime as ort

        options = {"runtime_only_key": "runtime_only_value"}
        captured: list[dict[str, str]] = []
        real_method = ort.SessionOptions.add_provider_for_devices

        def spy(self_sess, ep_devices, provider_opts):
            captured.append(dict(provider_opts))
            return real_method(self_sess, ep_devices, provider_opts)

        with patch.object(ort.SessionOptions, "add_provider_for_devices", spy):
            session = WinMLSession(
                onnx_path=simple_matmul_onnx,
                ep_device=cpu_ep_device,
                provider_options=options,
            )
            outputs = session.run(sample_input)

        assert options in captured, f"provider_options not forwarded; got calls with: {captured}"
        assert "C" in outputs

    def test_runtime_provider_options_override_ep_config(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
    ):
        """Runtime ``provider_options`` merge on top of and override ep_config options."""
        session = WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=cpu_ep_device,
            ep_config=EPConfig(
                provider="cpu",
                provider_options={"shared": "from_build", "build_only": "x"},
            ),
            provider_options={"shared": "from_runtime", "runtime_only": "y"},
        )

        # Runtime value wins for the shared key; both source-specific keys survive.
        assert session._provider_options == {
            "shared": "from_runtime",
            "build_only": "x",
            "runtime_only": "y",
        }

    def test_runtime_provider_options_do_not_mutate_ep_config(
        self,
        simple_matmul_onnx: Path,
        cpu_ep_device: EPDeviceTarget,
    ):
        """Session-local option overrides leave the caller's config reusable."""
        ep_config = EPConfig(
            provider="cpu",
            provider_options={"shared": "from_build", "build_only": "x"},
        )

        WinMLSession(
            onnx_path=simple_matmul_onnx,
            ep_device=cpu_ep_device,
            ep_config=ep_config,
            provider_options={"shared": "from_runtime", "runtime_only": "y"},
        )

        assert ep_config.provider_options == {
            "shared": "from_build",
            "build_only": "x",
        }

    def test_explicit_unavailable_target_propagates_structured_error(
        self,
        simple_matmul_onnx: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """An unavailable explicit EP/device request never falls back silently."""
        from winml.modelkit.session import WinMLEPNotDiscovered

        registry = _stub_registry(monkeypatch, None)
        registry.auto_device.side_effect = WinMLEPNotDiscovered("QNN unavailable")

        with pytest.raises(WinMLEPNotDiscovered, match="QNN unavailable"):
            WinMLSession(onnx_path=simple_matmul_onnx, device="gpu", ep="qnn")


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
# Task 7: EPDeviceTarget-based constructor (hard break)
# =============================================================================


def test_winml_session_accepts_ep_device(tmp_path, qnn_npu_ep_device, fake_ort_npu) -> None:
    """WinMLSession compiles with an explicit WinMLEPDevice."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")  # minimal placeholder; ORT is mocked
    with (
        patch("winml.modelkit.session.session.ort.InferenceSession") as mock_sess,
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        sess = WinMLSession(onnx_path, ep_device=qnn_npu_ep_device)
    mock_sess.assert_called_once()
    assert sess._ep_device is qnn_npu_ep_device


def test_winml_session_rejects_legacy_ep_kwarg(tmp_path, qnn_npu_ep_device) -> None:
    """Legacy ep="qnn" kwarg now raises TypeError."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    with pytest.raises(TypeError):
        WinMLSession(onnx_path, ep="qnn")  # type: ignore[call-arg]


def test_winml_session_accepts_device_kwarg_lazily(tmp_path, cpu_ep_device, monkeypatch) -> None:
    """The public device shortcut resolves a registry device without creating ORT."""
    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    _stub_registry(monkeypatch, cpu_ep_device)

    session = WinMLSession(onnx_path, device="cpu")

    assert session.device == "cpu"
    assert session._session is None


# =============================================================================
# Task 8: perf() validation + save/restore
# =============================================================================


def test_perf_validates_monitor_ep_name_match(tmp_path, qnn_npu_ep_device, fake_ort_npu) -> None:
    """Monitor for QNN against an OpenVINO WinMLEPDevice -> WinMLEPMonitorMismatch."""
    from .conftest import make_stub_winml_ep_device

    onnx_path = tmp_path / "noop.onnx"
    onnx_path.write_bytes(b"\x08\x01")
    fake_ov = MagicMock()
    fake_ov.ep_name = "OpenVINOExecutionProvider"
    fake_ov.device.type.name = "NPU"
    fake_ov.device.vendor_id = 0x8086
    fake_ov.device.device_id = 0x0BD0
    openvino_ep_device = make_stub_winml_ep_device(fake_ov, "OpenVINOExecutionProvider")
    qnn_monitor = MagicMock()
    qnn_monitor.ep_name = "qnn"
    qnn_monitor.get_provider_options.return_value = {}
    qnn_monitor.get_session_options.return_value = {}
    with (
        patch("winml.modelkit.session.session.ort.InferenceSession"),
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        sess = WinMLSession(onnx_path, ep_device=openvino_ep_device)
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
        patch("winml.modelkit.session.session.ort.InferenceSession"),
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        sess = WinMLSession(onnx_path, ep_device=qnn_npu_ep_device)
        snapshot = dict(sess._provider_options)
        with pytest.raises(RuntimeError), sess.perf(monitor=bad_monitor):
            pass
        assert sess._provider_options == snapshot
        assert sess._ep == "QNNExecutionProvider"
        assert sess._active_session_option_entries == {}  # back to empty (pre-perf state)
