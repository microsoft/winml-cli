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
from google.protobuf.message import EncodeError

from winml.modelkit.quant import WinMLQuantizationConfig
from winml.modelkit.quant.config import QuantizeResult
from winml.modelkit.quant.fp16 import convert_to_fp16
from winml.modelkit.quant.passes import BaseQuantPass, DynamicPass, FP16Pass, RTNPass, StaticPass


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

    def test_dynamic_returns_dynamic_pass(self) -> None:
        from winml.modelkit.quant.quantizer import expand_precision

        config = WinMLQuantizationConfig(mode="dynamic")
        passes = expand_precision("dynamic", config)
        assert len(passes) == 1
        assert isinstance(passes[0], DynamicPass)
        assert passes[0].config is config

    def test_unknown_mode_raises(self) -> None:
        from winml.modelkit.quant.quantizer import expand_precision

        with pytest.raises(ValueError, match="Unknown precision"):
            expand_precision("int8_only")

    def test_none_config_uses_default(self) -> None:
        from winml.modelkit.quant.quantizer import expand_precision

        passes = expand_precision("fp16")
        assert len(passes) == 1
        assert isinstance(passes[0].config, WinMLQuantizationConfig)

    def test_no_mode_uses_config_mode(self) -> None:
        """expand_precision(config=cfg) should use cfg.mode when precision is not given."""
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


class TestFP16Conversion:
    def test_retries_without_shape_inference_when_proto_serialization_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Large external-data models can exceed protobuf's in-memory serialize limit."""
        calls: list[dict] = []
        model = SimpleNamespace(graph=SimpleNamespace(initializer=[], node=[]))

        def fake_convert(model_arg, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise EncodeError("Failed to serialize proto")
            return model_arg

        monkeypatch.setattr(
            "onnxruntime.transformers.float16.convert_float_to_float16",
            fake_convert,
        )
        monkeypatch.setattr(
            "onnxruntime.transformers.onnx_model.OnnxModel.graph_topological_sort",
            lambda graph: None,
        )

        assert (
            convert_to_fp16(
                model,
                keep_io_types=True,
                op_block_list=["Softmax"],
            )
            is model
        )
        assert calls == [
            {"keep_io_types": True, "op_block_list": ["Softmax"]},
            {
                "keep_io_types": True,
                "disable_shape_infer": True,
                "op_block_list": ["Softmax"],
            },
        ]


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


# ---------------------------------------------------------------------------
# DynamicPass — config field wiring
# ---------------------------------------------------------------------------


def _install_fake_onnx_for_dynamic(
    monkeypatch: pytest.MonkeyPatch,
    fake_model: Any,
) -> None:
    """Install a fake ``winml.modelkit.onnx`` so DynamicPass needs no real I/O."""
    fake_onnx_mod = ModuleType("winml.modelkit.onnx")
    fake_onnx_mod.load_onnx = lambda *a, **k: fake_model  # type: ignore[attr-defined]
    fake_onnx_mod.save_onnx = lambda *a, **k: None  # type: ignore[attr-defined]
    fake_onnx_mod.capture_metadata = (  # type: ignore[attr-defined]
        lambda m: SimpleNamespace(node_count=len(m.graph.node))
    )
    fake_onnx_mod.restore_metadata = lambda *a, **k: None  # type: ignore[attr-defined]
    fake_onnx_mod.infer_shapes = lambda m: m  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "winml.modelkit.onnx", fake_onnx_mod)


def _patch_quantize_dynamic(
    monkeypatch: pytest.MonkeyPatch,
    init_kwargs: dict,
) -> None:
    """Patch the real ``quantize_dynamic``/``add_pre_process_metadata`` with no-ops."""
    import onnxruntime.quantization as oq
    import onnxruntime.quantization.quant_utils as oqu

    def fake_quantize_dynamic(**kwargs: Any) -> None:
        init_kwargs.update(kwargs)

    monkeypatch.setattr(oq, "quantize_dynamic", fake_quantize_dynamic)
    monkeypatch.setattr(oqu, "add_pre_process_metadata", lambda *a, **k: None)


class TestDynamicPassConfig:
    def test_reads_dynamic_fields_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DynamicPass should forward config fields to quantize_dynamic."""
        from onnxruntime.quantization import QuantType

        config = WinMLQuantizationConfig(
            mode="dynamic",
            weight_type="int8",
            per_channel=True,
            reduce_range=True,
            symmetric=True,
            op_types_to_quantize=["MatMul"],
            nodes_to_exclude=["skip_me"],
        )
        model_path = tmp_path / "model.onnx"
        model_path.write_text("x")
        output_path = tmp_path / "out.onnx"

        init_kwargs: dict = {}
        _patch_quantize_dynamic(monkeypatch, init_kwargs)
        fake_model = SimpleNamespace(
            graph=SimpleNamespace(
                node=[
                    SimpleNamespace(op_type="MatMulInteger"),
                    SimpleNamespace(op_type="DynamicQuantizeLinear"),
                    SimpleNamespace(op_type="Add"),
                ]
            )
        )
        _install_fake_onnx_for_dynamic(monkeypatch, fake_model)

        result = DynamicPass(config).run(model_path, output_path)

        assert result.success
        assert init_kwargs["weight_type"] == QuantType.QInt8
        assert init_kwargs["per_channel"] is True
        assert init_kwargs["reduce_range"] is True
        assert init_kwargs["op_types_to_quantize"] == ["MatMul"]
        assert init_kwargs["nodes_to_exclude"] == ["skip_me"]
        assert init_kwargs["extra_options"]["WeightSymmetric"] is True
        # Only the two dynamic-quant ops are counted, not the Add.
        assert result.nodes_quantized == 2

    def test_weight_symmetric_override_takes_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = WinMLQuantizationConfig(
            mode="dynamic",
            symmetric=False,
            weight_symmetric=True,
        )
        model_path = tmp_path / "model.onnx"
        model_path.write_text("x")

        init_kwargs: dict = {}
        _patch_quantize_dynamic(monkeypatch, init_kwargs)
        _install_fake_onnx_for_dynamic(monkeypatch, SimpleNamespace(graph=SimpleNamespace(node=[])))

        DynamicPass(config).run(model_path, tmp_path / "out.onnx")
        assert init_kwargs["extra_options"]["WeightSymmetric"] is True

    def test_uint8_weight_maps_to_quint8(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from onnxruntime.quantization import QuantType

        config = WinMLQuantizationConfig(mode="dynamic", weight_type="uint8")
        model_path = tmp_path / "model.onnx"
        model_path.write_text("x")

        init_kwargs: dict = {}
        _patch_quantize_dynamic(monkeypatch, init_kwargs)
        _install_fake_onnx_for_dynamic(monkeypatch, SimpleNamespace(graph=SimpleNamespace(node=[])))

        DynamicPass(config).run(model_path, tmp_path / "out.onnx")
        assert init_kwargs["weight_type"] == QuantType.QUInt8

    def test_16bit_weight_falls_back_to_int8_with_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dynamic quantization only supports 8-bit weights."""
        from onnxruntime.quantization import QuantType

        config = WinMLQuantizationConfig(mode="dynamic", weight_type="int16")
        model_path = tmp_path / "model.onnx"
        model_path.write_text("x")

        init_kwargs: dict = {}
        _patch_quantize_dynamic(monkeypatch, init_kwargs)
        _install_fake_onnx_for_dynamic(monkeypatch, SimpleNamespace(graph=SimpleNamespace(node=[])))

        result = DynamicPass(config).run(model_path, tmp_path / "out.onnx")
        assert init_kwargs["weight_type"] == QuantType.QInt8
        assert any("8-bit" in w for w in result.warnings)

    def test_dequantizelinear_counted_but_quantizelinear_not(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DequantizeLinear (weight/embedding restore) is counted as a quantized
        node; a stray static QuantizeLinear is not.

        Embedding models emit DequantizeLinear when a statically-quantized
        embedding feeds a non-integer op (e.g. Gather -> Add), so it must count
        toward ``nodes_quantized``. QuantizeLinear never appears in
        ``quantize_dynamic`` output and must be ignored.
        """
        config = WinMLQuantizationConfig(mode="dynamic")
        model_path = tmp_path / "model.onnx"
        model_path.write_text("x")

        init_kwargs: dict = {}
        _patch_quantize_dynamic(monkeypatch, init_kwargs)
        fake_model = SimpleNamespace(
            graph=SimpleNamespace(
                node=[
                    SimpleNamespace(op_type="DynamicQuantizeLinear"),
                    SimpleNamespace(op_type="MatMulInteger"),
                    SimpleNamespace(op_type="DequantizeLinear"),
                    SimpleNamespace(op_type="QuantizeLinear"),  # must NOT count
                    SimpleNamespace(op_type="Add"),
                ]
            )
        )
        _install_fake_onnx_for_dynamic(monkeypatch, fake_model)

        result = DynamicPass(config).run(model_path, tmp_path / "out.onnx")
        # DynamicQuantizeLinear + MatMulInteger + DequantizeLinear = 3.
        # QuantizeLinear and Add are excluded.
        assert result.nodes_quantized == 3


# ---------------------------------------------------------------------------
# WinMLQuantizationConfig — dynamic serialization
# ---------------------------------------------------------------------------


class TestDynamicConfigSerialization:
    def test_reduce_range_round_trips_in_dynamic_mode(self) -> None:
        config = WinMLQuantizationConfig(mode="dynamic", reduce_range=True)
        data = config.to_dict()
        assert data["reduce_range"] is True
        assert WinMLQuantizationConfig.from_dict(data).reduce_range is True

    def test_reduce_range_omitted_for_non_dynamic_modes(self) -> None:
        config = WinMLQuantizationConfig(mode="static", reduce_range=True)
        assert "reduce_range" not in config.to_dict()
