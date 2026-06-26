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


def _write_minimal_onnx_model(path: Path) -> None:
    """Write a tiny but valid ONNX model (with an ai.onnx opset) to *path*.

    The quantizer's input guard parses the file and requires a default opset
    import, so tests that exercise the quantize flow need a real model on disk.
    """
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save_model(model, str(path))


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
    _write_minimal_onnx_model(model_path)
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


def _write_opsetless_onnx_model(path: Path) -> None:
    """Write a parseable ONNX model that declares no default (ai.onnx) opset."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "g", [x], [y])
    # Only a custom-domain opset — no "" / "ai.onnx" import, which is the
    # signature ORT's get_opset_version() rejects.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.example", 1)])
    onnx.save_model(model, str(path))


def test_quantize_empty_input_model_surfaces_clear_error(tmp_path: Path) -> None:
    """A zero-byte input model yields a clear disk-space/corruption error.

    Regression for #259: a truncated optimize output must not surface as ORT's
    opaque "Failed to find proper ai.onnx domain".
    """
    model_path = tmp_path / "optimized.onnx"
    model_path.write_bytes(b"")  # truncated/zero-byte write left by disk-full

    result = quantize_onnx(model_path, output_path=tmp_path / "quantized.onnx")

    assert result.success is False
    assert result.output_path is None
    joined = " ".join(result.errors).lower()
    assert "disk space" in joined
    assert "failed to find proper ai.onnx domain" not in joined


def test_quantize_opsetless_input_model_surfaces_clear_error(tmp_path: Path) -> None:
    """A model with no ai.onnx opset import yields a clear, specific error."""
    model_path = tmp_path / "optimized.onnx"
    _write_opsetless_onnx_model(model_path)

    result = quantize_onnx(model_path, output_path=tmp_path / "quantized.onnx")

    assert result.success is False
    assert result.output_path is None
    joined = " ".join(result.errors)
    assert "no ai.onnx opset import" in joined
    assert "Failed to find proper ai.onnx domain" not in joined


def test_quantize_unparseable_input_model_surfaces_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An input model that fails to parse yields a clear error, not a traceback."""
    import onnx

    model_path = tmp_path / "optimized.onnx"
    _write_minimal_onnx_model(model_path)

    def _raise_parse_error(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("protobuf parse error")

    monkeypatch.setattr(onnx, "load", _raise_parse_error)

    result = quantize_onnx(model_path, output_path=tmp_path / "quantized.onnx")

    assert result.success is False
    assert result.output_path is None
    joined = " ".join(result.errors)
    assert "could not be parsed" in joined
    assert "disk space" in joined.lower()
