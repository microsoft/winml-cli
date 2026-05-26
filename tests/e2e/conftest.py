# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared fixtures and helpers for E2E tests.

Provides:
  - Hub model parametrization helpers (``HUB_PAIRS``, ``pytest_id``)
  - Cache-aware model resolution (``find_cache_dir``, ``resolve_model_arg``)
  - Common sample text inputs (``SAMPLE_TEXT``, ``TEXT_BY_FIELD``)
  - Shared fixtures (``test_image``, ``runner``)
  - Auto-skip for ``-m e2e``

E2E tests are auto-skipped unless explicitly selected with:
    uv run pytest -m e2e
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image


if TYPE_CHECKING:
    from click.testing import CliRunner


# ---------------------------------------------------------------------------
# Hub model parametrization (single source of truth)
# ---------------------------------------------------------------------------

_HUB_JSON = (
    Path(__file__).resolve().parents[2] / "src" / "winml" / "modelkit" / "data" / "hub_models.json"
)
HUB_DATA: dict = json.loads(_HUB_JSON.read_text(encoding="utf-8"))


def _unique_pairs() -> list[dict[str, str]]:
    """Deduplicate ``(model_id, task)`` — keep first occurrence."""
    seen: set[tuple[str, str]] = set()
    pairs: list[dict[str, str]] = []
    for entry in HUB_DATA["models"]:
        key = (entry["model_id"], entry["task"])
        if key not in seen:
            seen.add(key)
            pairs.append({"model_id": entry["model_id"], "task": entry["task"]})
    return pairs


HUB_PAIRS: list[dict[str, str]] = _unique_pairs()


def hub_test_id(pair: dict[str, str]) -> str:
    """Readable pytest ID, e.g. ``finbert-text_classification``."""
    short = pair["model_id"].rsplit("/", 1)[-1]
    task = pair["task"].replace("-", "_")
    return f"{short}-{task}"


# ---------------------------------------------------------------------------
# Cache-aware model resolution
# ---------------------------------------------------------------------------


def find_cache_dir(model_id: str, task: str | None = None) -> Path | None:
    """Find the winml build-cache directory for a model+task, or None.

    Looks for ``~/.cache/winml/artifacts/{slug}/`` containing a
    ``*_model.onnx`` file whose manifest task matches *task*.
    """
    from winml.modelkit.cache import get_cache_dir, model_id_to_slug
    from winml.modelkit.inference.engine import _find_build_artifacts

    slug = model_id_to_slug(model_id)
    cache_dir = get_cache_dir() / "artifacts" / slug
    if not cache_dir.is_dir():
        return None
    try:
        _find_build_artifacts(cache_dir, task=task)
        return cache_dir
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def resolve_model_arg(model_id: str, task: str | None = None) -> str:
    """Return the cache directory (fast) or HF model ID (slow rebuild)."""
    cache_dir = find_cache_dir(model_id, task=task)
    if cache_dir is not None:
        return str(cache_dir)
    return model_id


# ---------------------------------------------------------------------------
# Common sample inputs
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."

TEXT_BY_FIELD: dict[str, str] = {
    "question": "What is the capital of France?",
    "context": (
        "Paris is the capital of France. "
        "It is known for the Eiffel Tower and its rich cultural heritage."
    ),
    "text_1": "A man is eating food.",
    "text_2": "A man is eating a piece of bread.",
}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_image(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Generate a 224x224 random RGB JPEG (reused across the module)."""
    d = tmp_path_factory.mktemp("run_e2e_assets")
    img_path = d / "test_image.jpg"
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(img_path), format="JPEG")
    return str(img_path)


@pytest.fixture
def runner() -> CliRunner:
    from click.testing import CliRunner

    return CliRunner()


# ---------------------------------------------------------------------------
# Auto-skip E2E
# ---------------------------------------------------------------------------


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
def simple_matmul_onnx(tmp_path: Path) -> Path:
    """Create simple MatMul ONNX model (A @ B = C) for EP inference tests."""
    a_input = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    c_output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])
    np.random.seed(42)
    b_values = np.random.randn(4, 4).astype(np.float32)
    b_tensor = helper.make_tensor("B", TensorProto.FLOAT, [4, 4], b_values.flatten().tolist())
    matmul_node = helper.make_node("MatMul", ["A", "B"], ["C"], name="matmul")
    graph = helper.make_graph([matmul_node], "test_matmul", [a_input], [c_output], [b_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    path = tmp_path / "test_matmul.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def qdq_matmul_onnx(tmp_path: Path) -> Path:
    """Synthetic QDQ INT8 MatMul (A @ B = C) for EPs that require quantized
    graphs (e.g. VitisAI on AMD NPU, which crashes on FP32 graphs lacking a
    compilable subgraph).

    Same A/C signature as ``simple_matmul_onnx`` so ``sample_input`` works
    unchanged. Scales and zero points are synthetic — no real calibration
    is performed; just a symmetric INT8 scheme with ``scale = max(|x|)/127``
    and ``zero_point = 0``.
    """
    from onnx import numpy_helper

    np.random.seed(42)
    b_fp32 = np.random.randn(4, 4).astype(np.float32)
    b_scale = float(np.max(np.abs(b_fp32)) / 127.0)
    b_int8 = np.clip(np.round(b_fp32 / b_scale), -128, 127).astype(np.int8)

    a_scale = np.float32(1.0 / 127.0)
    y_scale = np.float32(1.0 / 127.0)
    zp = np.int8(0)

    initializers = [
        numpy_helper.from_array(b_int8, name="B_int8"),
        numpy_helper.from_array(np.array(b_scale, np.float32), name="B_scale"),
        numpy_helper.from_array(np.array(zp, np.int8), name="B_zp"),
        numpy_helper.from_array(np.array(a_scale, np.float32), name="A_scale"),
        numpy_helper.from_array(np.array(zp, np.int8), name="A_zp"),
        numpy_helper.from_array(np.array(y_scale, np.float32), name="Y_scale"),
        numpy_helper.from_array(np.array(zp, np.int8), name="Y_zp"),
    ]

    a_input = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    c_output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])

    nodes = [
        helper.make_node("QuantizeLinear", ["A", "A_scale", "A_zp"], ["A_q"], name="Q_A"),
        helper.make_node(
            "DequantizeLinear", ["A_q", "A_scale", "A_zp"], ["A_dq"], name="DQ_A"
        ),
        helper.make_node(
            "DequantizeLinear", ["B_int8", "B_scale", "B_zp"], ["B_dq"], name="DQ_B"
        ),
        helper.make_node("MatMul", ["A_dq", "B_dq"], ["M_out"], name="matmul"),
        helper.make_node(
            "QuantizeLinear", ["M_out", "Y_scale", "Y_zp"], ["M_q"], name="Q_M"
        ),
        helper.make_node(
            "DequantizeLinear", ["M_q", "Y_scale", "Y_zp"], ["C"], name="DQ_M"
        ),
    ]
    graph = helper.make_graph(nodes, "qdq_matmul", [a_input], [c_output], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = tmp_path / "test_matmul_qdq.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture
def sample_input() -> dict[str, np.ndarray]:
    """Create sample input for MatMul model."""
    np.random.seed(123)
    return {"A": np.random.randn(1, 4).astype(np.float32)}


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
