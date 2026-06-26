# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for quantizer cleanup behavior."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

from winml.modelkit.quant import WinMLQuantizationConfig, quantize_onnx


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pytest


class _FakeCalibrationReader:
    """Minimal calibration reader that satisfies the quantizer protocol."""

    def get_next(self) -> dict[str, np.ndarray] | None:
        return None

    def rewind(self) -> None:
        return None


class _FakeOrtModule(ModuleType):
    quantization: ModuleType


class _FakeOrtQuantizationModule(ModuleType):
    CalibrationMethod: Any
    QuantType: Any
    get_qdq_config: Any
    quantize: Any
    quant_utils: ModuleType


class _FakeOrtQuantUtilsModule(ModuleType):
    add_pre_process_metadata: Any


class _FakeOnnxModule(ModuleType):
    capture_metadata: Any
    load_onnx: Any
    restore_metadata: Any
    save_onnx: Any
    infer_shapes: Any


class _FakeQdqFixModule(ModuleType):
    fix_qdq_dtype_info: Any


class _FakeCompilerModule(ModuleType):
    QDQ_OP_TYPES: set[str]


def _install_fake_ort_quantization(monkeypatch: pytest.MonkeyPatch, *, quantize_impl) -> None:
    """Install a minimal fake ORT quantization module for unit testing."""
    ort_module = _FakeOrtModule("onnxruntime")
    quant_module = _FakeOrtQuantizationModule("onnxruntime.quantization")
    quant_module.CalibrationMethod = SimpleNamespace(
        MinMax="minmax",
        Entropy="entropy",
        Percentile="percentile",
    )
    quant_module.QuantType = SimpleNamespace(
        QUInt8="uint8",
        QInt8="int8",
        QUInt16="uint16",
        QInt16="int16",
    )
    quant_module.get_qdq_config = lambda **_: SimpleNamespace(use_external_data_format=False)
    quant_module.quantize = quantize_impl
    quant_utils_module = _FakeOrtQuantUtilsModule("onnxruntime.quantization.quant_utils")
    quant_utils_module.add_pre_process_metadata = lambda _model: None
    quant_module.quant_utils = quant_utils_module
    ort_module.quantization = quant_module
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quant_module)
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization.quant_utils", quant_utils_module)


def test_quantize_onnx_removes_only_exact_external_data_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup should remove only the exact .data sidecar for the output model."""
    model_path = tmp_path / "model.onnx"
    model_path.write_text("input")
    output_path = tmp_path / "quantized.onnx"
    exact_sidecar = tmp_path / f"{output_path.name}.data"
    extra_suffix_sidecar = tmp_path / f"{output_path.name}.data.bak"
    exact_sidecar.write_text("stale")
    extra_suffix_sidecar.write_text("keep")

    # The quantizer hands the in-memory input model (not the path) to ORT so it
    # can tag it as pre-processed without mutating the user's input file.
    input_model = SimpleNamespace()

    def fake_quantize(*, model_input, model_output: str, quant_config) -> None:
        assert model_input is input_model
        assert model_output == str(output_path.resolve())
        assert quant_config.use_external_data_format is True
        assert not exact_sidecar.exists()
        assert extra_suffix_sidecar.exists()
        output_path.write_text("quantized")

    _install_fake_ort_quantization(monkeypatch, quantize_impl=fake_quantize)

    fake_onnx_module = _FakeOnnxModule("winml.modelkit.onnx")
    quantized_model = SimpleNamespace(
        graph=SimpleNamespace(node=[SimpleNamespace(op_type="QuantizeLinear")])
    )
    load_results = [input_model, quantized_model]
    fake_onnx_module.capture_metadata = lambda _model: SimpleNamespace(node_count=0)
    fake_onnx_module.load_onnx = lambda *_args, **_kwargs: load_results.pop(0)
    fake_onnx_module.restore_metadata = lambda *_args, **_kwargs: None
    fake_onnx_module.save_onnx = lambda *_args, **_kwargs: None
    fake_onnx_module.infer_shapes = lambda model: model
    monkeypatch.setitem(sys.modules, "winml.modelkit.onnx", fake_onnx_module)

    fake_qdq_fix_module = _FakeQdqFixModule("winml.modelkit.quant.qdq_fix")
    fake_qdq_fix_module.fix_qdq_dtype_info = lambda _model: SimpleNamespace(warnings=[])
    monkeypatch.setitem(sys.modules, "winml.modelkit.quant.qdq_fix", fake_qdq_fix_module)

    fake_compiler_module = _FakeCompilerModule("winml.modelkit.compiler")
    fake_compiler_module.QDQ_OP_TYPES = {"QuantizeLinear", "DequantizeLinear"}
    monkeypatch.setitem(sys.modules, "winml.modelkit.compiler", fake_compiler_module)

    result = quantize_onnx(
        model_path,
        output_path=output_path,
        config=WinMLQuantizationConfig(calibration_data=_FakeCalibrationReader()),
    )

    assert result.success is True
    assert extra_suffix_sidecar.exists()


def test_quantize_onnx_applies_model_type_finalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered model_type finalizer is resolved + applied before dispatch.

    The model-type-specific quant policy used to be dispatched at each call site
    (CLI build, library build). It now lives behind a single seam in
    quantize_onnx, keyed on ``config.model_type``: the finalizer is resolved from
    the calibration registry and its returned config is what the mode handler
    receives.
    """
    import winml.modelkit.quant.calibration as calibration_mod
    import winml.modelkit.quant.quantizer as quantizer_mod

    model_path = tmp_path / "model.onnx"
    model_path.write_text("input")
    output_path = tmp_path / "quantized.onnx"

    finalized_config = WinMLQuantizationConfig(
        model_type="dummy_type",
        calibration_data=_FakeCalibrationReader(),
    )

    finalize_calls: list[dict[str, Any]] = []

    class _StubFinalizer:
        def finalize(self, config, *, onnx_path, model_id):  # type: ignore[no-untyped-def]
            finalize_calls.append({"config": config, "onnx_path": onnx_path, "model_id": model_id})
            return finalized_config

    monkeypatch.setattr(calibration_mod, "get_quant_finalizer", lambda model_type: _StubFinalizer())

    handler_calls: list[WinMLQuantizationConfig] = []

    def _fake_qdq(*, config, **_kwargs):  # type: ignore[no-untyped-def]
        handler_calls.append(config)
        return SimpleNamespace(success=True, output_path=output_path, errors=[])

    monkeypatch.setattr(quantizer_mod, "_quantize_qdq", _fake_qdq)

    result = quantize_onnx(
        model_path,
        output_path=output_path,
        config=WinMLQuantizationConfig(
            model_type="dummy_type",
            model_id="some/model-id",
        ),
    )

    assert result.success is True
    # Finalizer was resolved + invoked with the exported graph + model id.
    assert len(finalize_calls) == 1
    assert finalize_calls[0]["onnx_path"] == model_path
    assert finalize_calls[0]["model_id"] == "some/model-id"
    # The handler ran against the finalized config, not the original.
    assert handler_calls == [finalized_config]


def test_quantize_onnx_skips_finalizer_when_calibration_data_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-supplied calibration reader bypasses the model_type finalizer."""
    import winml.modelkit.quant.calibration as calibration_mod
    import winml.modelkit.quant.quantizer as quantizer_mod

    model_path = tmp_path / "model.onnx"
    model_path.write_text("input")
    output_path = tmp_path / "quantized.onnx"

    def _boom(_model_type):  # type: ignore[no-untyped-def]
        raise AssertionError("finalizer must not be resolved when calibration_data is set")

    monkeypatch.setattr(calibration_mod, "get_quant_finalizer", _boom)

    def _fake_qdq(*, config, **_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(success=True, output_path=output_path, errors=[])

    monkeypatch.setattr(quantizer_mod, "_quantize_qdq", _fake_qdq)

    result = quantize_onnx(
        model_path,
        output_path=output_path,
        config=WinMLQuantizationConfig(
            model_type="dummy_type",
            calibration_data=_FakeCalibrationReader(),
        ),
    )

    assert result.success is True
