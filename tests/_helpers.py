# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared test helpers.

Combines:
- ``get_minimal_onnx_model_path``: a tiny Identity ONNX model used by
  session-related tests.
- ``run_inspect``: a thin wrapper around ``CliRunner.invoke`` for the
  ``inspect`` command used by both ``tests/cli/test_inspect_cli.py`` and
  ``tests/e2e/test_inspect_e2e.py`` so the invocation envelope
  (``obj={}``, ``mix_stderr`` defaults, etc.) lives in a single place.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner, Result

from winml.modelkit.commands.inspect import inspect


def get_minimal_onnx_model_path() -> Path:
    """Return path to a tiny Identity ONNX model used by session tests."""
    from onnx import TensorProto, helper, save

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
        save(model, fixture)
    return fixture


def run_inspect(*args: str) -> Result:
    """Invoke the ``inspect`` Click command with *args and return the Result."""
    return CliRunner().invoke(inspect, list(args), obj={})
