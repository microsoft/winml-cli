# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for quantization passes and the Quantizer pipeline."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from winml.modelkit.quant import WinMLQuantizationConfig
from winml.modelkit.quant.config import QuantizeResult
from winml.modelkit.quant.passes import BaseQuantPass, FP16Pass, RTNPass, StaticPass


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_result(output_path: Path, *, total_time: float = 0.1) -> QuantizeResult:
    return QuantizeResult(success=True, output_path=output_path, total_time_seconds=total_time)


def _fail_result() -> QuantizeResult:
    return QuantizeResult(success=False, output_path=None, errors=["boom"])


class _StubPass(BaseQuantPass):
    """Configurable stub pass for Quantizer pipeline tests."""

    def __init__(self, config: WinMLQuantizationConfig, *, succeed: bool = True) -> None:
        super().__init__(config)
        self.called_with: list[tuple[Path, Path]] = []
        self._succeed = succeed

    def run(
        self,
        model_path: Path,
        output_path: Path,
        *,
        use_external_data: bool = True,
    ) -> QuantizeResult:
        self.called_with.append((model_path, output_path))
        if self._succeed:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("model")
            return _ok_result(output_path)
        return _fail_result()


# ---------------------------------------------------------------------------
# expand_precision
# ---------------------------------------------------------------------------


class TestExpandPrecision:
    def test_fp16_returns_fp16_pass(self) -> None:
        from winml.modelkit.quant.quantizer import expand_precision

        config = WinMLQuantizationConfig(mode="fp16")
        passes = expand_precision("fp16", config)
        assert len(passes) == 1
        assert isinstance(passes[0], FP16Pass)
        assert passes[0].config is config

    def test_rtn_returns_rtn_pass(self) -> None:
        from winml.modelkit.quant.quantizer import expand_precision

        config = WinMLQuantizationConfig(mode="rtn")
        passes = expand_precision("rtn", config)
        assert len(passes) == 1
        assert isinstance(passes[0], RTNPass)
        assert passes[0].config is config

    def test_static_returns_qdq_pass(self) -> None:
        from winml.modelkit.quant.quantizer import expand_precision

        config = WinMLQuantizationConfig(mode="static")
        passes = expand_precision("static", config)
        assert len(passes) == 1
        assert isinstance(passes[0], StaticPass)

    def test_dynamic_returns_qdq_pass(self) -> None:
        from winml.modelkit.quant.quantizer import expand_precision

        config = WinMLQuantizationConfig(mode="static")
        passes = expand_precision("dynamic", config)
        assert len(passes) == 1
        assert isinstance(passes[0], StaticPass)

    def test_unknown_mode_raises(self) -> None:
        from winml.modelkit.quant.quantizer import expand_precision

        with pytest.raises(ValueError, match="Unknown precision mode"):
            expand_precision("int8_only")

    def test_none_config_uses_default(self) -> None:
        from winml.modelkit.quant.quantizer import expand_precision

        passes = expand_precision("fp16")
        assert len(passes) == 1
        assert isinstance(passes[0].config, WinMLQuantizationConfig)

    def test_no_mode_uses_config_mode(self) -> None:
        """expand_precision(config=cfg) should use cfg.mode when mode is not given."""
        from winml.modelkit.quant.quantizer import expand_precision

        config = WinMLQuantizationConfig(mode="rtn", rtn_bits=4)
        passes = expand_precision(config=config)
        assert len(passes) == 1
        assert isinstance(passes[0], RTNPass)
        assert passes[0].config is config


# ---------------------------------------------------------------------------
# Quantizer — single pass
# ---------------------------------------------------------------------------


class TestQuantizerSinglePass:
    def test_single_pass_calls_run_with_correct_paths(self, tmp_path: Path) -> None:
        from winml.modelkit.quant.quantizer import Quantizer

        config = WinMLQuantizationConfig()
        model = tmp_path / "model.onnx"
        model.write_text("x")
        out = tmp_path / "out.onnx"

        stub = _StubPass(config)
        result = Quantizer([stub]).run(model, out)

        assert result.success
        assert result.output_path == out
        assert stub.called_with == [(model, out)]

    def test_single_pass_missing_model_returns_failure(self, tmp_path: Path) -> None:
        from winml.modelkit.quant.quantizer import Quantizer

        config = WinMLQuantizationConfig()
        stub = _StubPass(config)
        result = Quantizer([stub]).run(tmp_path / "missing.onnx", tmp_path / "out.onnx")

        assert not result.success
        assert "not found" in result.errors[0].lower()
        assert stub.called_with == []

    def test_single_pass_exception_returns_failure(self, tmp_path: Path) -> None:
        from winml.modelkit.quant.quantizer import Quantizer

        config = WinMLQuantizationConfig()
        model = tmp_path / "model.onnx"
        model.write_text("x")

        class _ExplodingPass(BaseQuantPass):
            def run(self, model_path, output_path, *, use_external_data=True):
                raise RuntimeError("kaboom")

        result = Quantizer([_ExplodingPass(config)]).run(model, tmp_path / "out.onnx")
        assert not result.success
        assert any("kaboom" in e for e in result.errors)

    def test_empty_passes_raises(self) -> None:
        from winml.modelkit.quant.quantizer import Quantizer

        with pytest.raises(ValueError, match="at least one pass"):
            Quantizer([])


# ---------------------------------------------------------------------------
# quantize_onnx — kwargs guard
# ---------------------------------------------------------------------------


class TestQuantizeOnnxKwargsGuard:
    def test_unexpected_kwarg_raises_type_error(self, tmp_path: Path) -> None:
        """quantize_onnx must raise TypeError on unrecognised kwargs."""
        from winml.modelkit.quant import quantize_onnx

        model_path = tmp_path / "model.onnx"
        model_path.write_text("x")

        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            quantize_onnx(model_path, use_external_data_format=False)


# ---------------------------------------------------------------------------
# Quantizer — multi-pass chaining
# ---------------------------------------------------------------------------


class TestQuantizerMultiPass:
    def test_multi_pass_chains_input_output(self, tmp_path: Path) -> None:
        from winml.modelkit.quant.quantizer import Quantizer

        config = WinMLQuantizationConfig()
        model = tmp_path / "model.onnx"
        model.write_text("x")
        final_out = tmp_path / "final.onnx"

        p1 = _StubPass(config)
        p2 = _StubPass(config)
        result = Quantizer([p1, p2]).run(model, final_out)

        assert result.success
        assert result.output_path == final_out

        # p1 receives the original model; p2 receives p1's output (a temp file)
        p1_input, p1_output = p1.called_with[0]
        p2_input, p2_output = p2.called_with[0]

        assert p1_input == model
        assert p1_output != final_out  # intermediate temp file
        assert p2_input == p1_output  # chained correctly
        assert p2_output == final_out

    def test_multi_pass_stats_are_merged(self, tmp_path: Path) -> None:
        from winml.modelkit.quant.quantizer import Quantizer

        config = WinMLQuantizationConfig()
        model = tmp_path / "model.onnx"
        model.write_text("x")

        class _TimedPass(BaseQuantPass):
            def __init__(self, cfg, *, nodes: int, time_s: float) -> None:
                super().__init__(cfg)
                self._nodes = nodes
                self._time = time_s

            def run(self, model_path, output_path, *, use_external_data=True):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("x")
                return QuantizeResult(
                    success=True,
                    output_path=output_path,
                    nodes_quantized=self._nodes,
                    total_time_seconds=self._time,
                )

        p1 = _TimedPass(config, nodes=10, time_s=1.0)
        p2 = _TimedPass(config, nodes=5, time_s=2.0)
        result = Quantizer([p1, p2]).run(model, tmp_path / "out.onnx")

        assert result.nodes_quantized == 15
        assert abs(result.total_time_seconds - 3.0) < 1e-9

    def test_multi_pass_aborts_on_failure(self, tmp_path: Path) -> None:
        from winml.modelkit.quant.quantizer import Quantizer

        config = WinMLQuantizationConfig()
        model = tmp_path / "model.onnx"
        model.write_text("x")

        p1 = _StubPass(config, succeed=False)
        p2 = _StubPass(config, succeed=True)

        result = Quantizer([p1, p2]).run(model, tmp_path / "out.onnx")

        assert not result.success
        assert p2.called_with == []  # p2 never called

    def test_multi_pass_warnings_concatenated(self, tmp_path: Path) -> None:
        from winml.modelkit.quant.quantizer import Quantizer

        config = WinMLQuantizationConfig()
        model = tmp_path / "model.onnx"
        model.write_text("x")

        class _WarnPass(BaseQuantPass):
            def __init__(self, cfg, msg: str) -> None:
                super().__init__(cfg)
                self._msg = msg

            def run(self, model_path, output_path, *, use_external_data=True):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("x")
                return QuantizeResult(success=True, output_path=output_path, warnings=[self._msg])

        result = Quantizer(
            [
                _WarnPass(config, "w1"),
                _WarnPass(config, "w2"),
            ]
        ).run(model, tmp_path / "out.onnx")

        assert result.warnings == ["w1", "w2"]


# ---------------------------------------------------------------------------
# FP16Pass — config field wiring
# ---------------------------------------------------------------------------


class TestFP16PassConfig:
    def test_reads_fp16_fields_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FP16Pass should pass fp16_keep_io_types and fp16_op_block_list to convert_to_fp16."""
        config = WinMLQuantizationConfig(
            mode="fp16",
            fp16_keep_io_types=False,
            fp16_op_block_list=["Gather"],
        )
        model_path = tmp_path / "model.onnx"
        model_path.write_text("x")
        output_path = tmp_path / "out.onnx"

        calls: list[dict] = []
        fake_model = SimpleNamespace()

        def fake_convert(model, *, keep_io_types, op_block_list):
            calls.append({"keep_io_types": keep_io_types, "op_block_list": op_block_list})
            return model

        # Patch the source modules that are lazily imported inside run()
        fake_onnx_mod = ModuleType("winml.modelkit.onnx")
        fake_onnx_mod.load_onnx = lambda *a, **k: fake_model  # type: ignore[attr-defined]
        fake_onnx_mod.save_onnx = lambda *a, **k: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "winml.modelkit.onnx", fake_onnx_mod)

        fake_fp16_mod = ModuleType("winml.modelkit.quant.fp16")
        fake_fp16_mod.convert_to_fp16 = fake_convert  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "winml.modelkit.quant.fp16", fake_fp16_mod)

        result = FP16Pass(config).run(model_path, output_path)

        assert result.success
        assert calls == [{"keep_io_types": False, "op_block_list": ["Gather"]}]


# ---------------------------------------------------------------------------
# RTNPass — config field wiring
# ---------------------------------------------------------------------------


def _install_fake_ort_nbits(
    monkeypatch: pytest.MonkeyPatch,
    fake_quantized_model: Any,
    init_kwargs: list[dict],
) -> None:
    """Install a minimal fake MatMulNBitsQuantizer into sys.modules."""

    class FakeMatMulNBitsQuantizer:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.append(kwargs)

        def process(self) -> None:
            pass

        model = SimpleNamespace(model=fake_quantized_model)

    fake_ort_quant = ModuleType("onnxruntime.quantization.matmul_nbits_quantizer")
    fake_ort_quant.MatMulNBitsQuantizer = FakeMatMulNBitsQuantizer  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime.quantization.matmul_nbits_quantizer",
        fake_ort_quant,
    )

    fake_onnx_mod = ModuleType("winml.modelkit.onnx")
    fake_onnx_mod.save_onnx = lambda *a, **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "winml.modelkit.onnx", fake_onnx_mod)


class TestRTNPassConfig:
    def test_reads_rtn_fields_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RTNPass should forward all rtn_* fields to MatMulNBitsQuantizer."""
        config = WinMLQuantizationConfig(
            mode="rtn",
            rtn_bits=8,
            rtn_block_size=64,
            rtn_symmetric=False,
            rtn_accuracy_level=2,
        )
        model_path = tmp_path / "model.onnx"
        model_path.write_text("x")
        output_path = tmp_path / "out.onnx"

        init_kwargs: list[dict] = []
        fake_quantized_model = SimpleNamespace(graph=SimpleNamespace(node=[]))
        _install_fake_ort_nbits(monkeypatch, fake_quantized_model, init_kwargs)

        result = RTNPass(config).run(model_path, output_path)

        assert result.success
        assert init_kwargs[0]["bits"] == 8
        assert init_kwargs[0]["block_size"] == 64
        assert init_kwargs[0]["is_symmetric"] is False
        assert init_kwargs[0]["accuracy_level"] == 2

    def test_accuracy_level_zero_maps_to_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = WinMLQuantizationConfig(mode="rtn", rtn_accuracy_level=0)
        model_path = tmp_path / "model.onnx"
        model_path.write_text("x")

        init_kwargs: list[dict] = []
        fake_quantized_model = SimpleNamespace()
        _install_fake_ort_nbits(monkeypatch, fake_quantized_model, init_kwargs)

        RTNPass(config).run(model_path, tmp_path / "out.onnx")
        assert init_kwargs[0]["accuracy_level"] is None
