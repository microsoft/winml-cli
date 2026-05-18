# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared test helpers for session-related tests."""

from __future__ import annotations

from pathlib import Path


def get_minimal_onnx_model_path() -> Path:
    """Return path to a tiny Identity ONNX model used by session tests."""
    import onnx
    from onnx import TensorProto, helper

    fixture_dir = Path(__file__).parent / "_fixtures"
    fixture_dir.mkdir(exist_ok=True)
    fixture = fixture_dir / "identity.onnx"
    if not fixture.exists():
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Identity", ["input"], ["output"])
        graph = helper.make_graph([node], "identity", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.save(model, fixture)
    return fixture
