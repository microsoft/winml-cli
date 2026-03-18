"""Shared fixtures for E2E tests.

These fixtures generate real ONNX files on-the-fly and provide
model-task combination parameters for parametrized tests.

E2E tests are auto-skipped unless explicitly selected with:
    uv run pytest -m e2e
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


if TYPE_CHECKING:
    from pathlib import Path


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip E2E tests unless '-m e2e' is explicitly passed."""
    marker_expr = config.getoption("-m", default="")
    if "e2e" in str(marker_expr):
        return  # User explicitly requested E2E tests
    skip_e2e = pytest.mark.skip(reason="E2E tests require -m e2e (skipped by default)")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


@pytest.fixture(scope="session")
def onnx_fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a session-scoped temp directory with generated ONNX files."""
    d = tmp_path_factory.mktemp("onnx_fixtures")
    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5])
    w_init = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [10, 5],
        np.random.randn(10, 5).astype(np.float32).tobytes(),
        raw=True,
    )
    node = helper.make_node("MatMul", ["input", "weight"], ["output"])
    graph = helper.make_graph([node], "test_graph", [x_info], [y_info], [w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx_path = d / "test_model.onnx"
    onnx.save(model, str(onnx_path))
    return d


@pytest.fixture(scope="session")
def onnx_model_path(onnx_fixture_dir: Path) -> Path:
    """Path to a valid minimal ONNX model for testing."""
    return onnx_fixture_dir / "test_model.onnx"


@pytest.fixture
def build_config_path(tmp_path: Path) -> Path:
    """Create a minimal WinMLBuildConfig JSON file for build/perf tests."""
    config = {
        "loader": {"task": "image-classification"},
        "export": {"opset_version": 17, "batch_size": 1},
        "optim": {},
        "quant": None,
        "compile": None,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config))
    return p
