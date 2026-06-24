# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the quantization debug engine.

Covers the WinML wrapper around ORT's ``qdq_loss_debug``: activation SQNR,
weight SQNR, and graph-output cumulative SQNR. ORT's measurement functions are
faked so the tests run without an inference session.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING

from winml.modelkit.debug import debug_quantization
from winml.modelkit.debug.debugger import _graph_output_names


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _build_tiny_model(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    matmul = helper.make_node("MatMul", ["X", "W"], ["Y"], name="matmul0")
    relu = helper.make_node("Relu", ["Y"], ["Z"], name="relu0")
    graph = helper.make_graph(
        [matmul, relu],
        "tiny",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
        outputs=[helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
        initializer=[helper.make_tensor("W", TensorProto.FLOAT, [2, 2], [1, 0, 0, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def test_graph_output_names(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny.onnx"
    _build_tiny_model(model_path)

    assert _graph_output_names(model_path) == ["Z"]


class _FakeReader:
    def get_next(self) -> None:
        return None

    def rewind(self) -> None:
        return None


def _install_fake_qdq_loss_debug(
    monkeypatch: pytest.MonkeyPatch,
    act_err: dict[str, dict[str, float]],
    weight_err: dict[str, float],
) -> None:
    mod = ModuleType("onnxruntime.quantization.qdq_loss_debug")
    mod.modify_model_output_intermediate_tensors = (  # type: ignore[attr-defined]
        lambda _in, out, **_kw: __import__("pathlib").Path(out).write_text("x")
    )
    mod.create_activation_matching = lambda *_a, **_k: {}  # type: ignore[attr-defined]
    mod.compute_activation_error = lambda _m: act_err  # type: ignore[attr-defined]
    mod.create_weight_matching = lambda *_a, **_k: {}  # type: ignore[attr-defined]
    mod.compute_weight_error = lambda _m, **_k: weight_err  # type: ignore[attr-defined]
    mod.collect_activations = lambda *_a, **_k: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization.qdq_loss_debug", mod)

    # Avoid real dataset construction and inference.
    import winml.modelkit.datasets as datasets_mod

    monkeypatch.setattr(
        datasets_mod, "DatasetCalibrationReader", lambda **_kw: _FakeReader()
    )


def test_debug_quantization_returns_activations_weights_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    float_model = tmp_path / "tiny.onnx"
    _build_tiny_model(float_model)
    quant_model = tmp_path / "tiny_quant.onnx"
    quant_model.write_text("ignored")

    act_err = {
        "Y": {"qdq_err": 80.0, "xmodel_err": 40.0},
        "Z": {"qdq_err": 20.0, "xmodel_err": 10.0},
    }
    weight_err = {"W": 12.5}
    _install_fake_qdq_loss_debug(monkeypatch, act_err, weight_err)

    result = debug_quantization(float_model, quant_model)

    activations = {a["tensor_name"]: a for a in result["activations"]}
    assert activations["Y"]["local_sqnr_db"] == 80.0
    assert activations["Y"]["cumulative_sqnr_db"] == 40.0
    assert activations["Z"]["cumulative_sqnr_db"] == 10.0

    assert result["weights"] == [{"weight_name": "W", "weight_sqnr_db": 12.5}]

    # The single graph output Z carries its cumulative SQNR.
    assert result["model_outputs"] == [
        {"output_name": "Z", "cumulative_sqnr_db": 10.0}
    ]

    assert result["summary"]["local"] == {
        "count": 2,
        "mean": 50.0,
        "std": 30.0,
        "min": 20.0,
        "max": 80.0,
    }
    assert result["summary"]["cumulative"]["count"] == 2
    assert result["summary"]["weight"] == {
        "count": 1,
        "mean": 12.5,
        "std": 0.0,
        "min": 12.5,
        "max": 12.5,
    }


def test_debug_quantization_handles_missing_cumulative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    float_model = tmp_path / "tiny.onnx"
    _build_tiny_model(float_model)
    quant_model = tmp_path / "tiny_quant.onnx"
    quant_model.write_text("ignored")

    # No xmodel_err -> cumulative stays None (no float reference).
    act_err = {"Z": {"qdq_err": 15.0}}
    _install_fake_qdq_loss_debug(monkeypatch, act_err, {})

    result = debug_quantization(float_model, quant_model)

    (z,) = result["activations"]
    assert z["cumulative_sqnr_db"] is None
    assert z["local_sqnr_db"] == 15.0
    assert result["weights"] == []
    # Output Z has no measured cumulative SQNR.
    assert result["model_outputs"] == [
        {"output_name": "Z", "cumulative_sqnr_db": None}
    ]

    # No cumulative or weight values remain, so those summaries are empty.
    assert result["summary"]["local"]["count"] == 1
    assert result["summary"]["cumulative"] == {
        "count": 0,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
    }
    assert result["summary"]["weight"]["count"] == 0
