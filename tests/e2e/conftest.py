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

# Default per-test timeout (seconds) for E2E tests when --timeout is not passed
# on the command line. Higher than the global 300 s ini default because cold
# E2E runs build the model end-to-end (export -> optimize -> quantize ->
# compile). Precedence (highest first): a per-test ``@pytest.mark.timeout``
# marker, then ``--timeout`` on the CLI, then this default. An explicit
# ``--timeout`` therefore always wins over this fallback.
E2E_DEFAULT_TIMEOUT = 900


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip E2E tests unless '-m e2e' is passed; else apply the e2e timeout default.

    Additionally, tests marked ``e2e_run`` or ``e2e_serve`` are always skipped
    unless their marker name appears in the ``-m`` expression (the commands are
    not yet enabled — see https://github.com/microsoft/winml-cli/issues/892).
    """
    marker_expr = str(config.getoption("-m", default=""))
    if "e2e" in marker_expr:
        # E2E tests are running. Inject the default timeout only when neither a
        # per-test marker nor --timeout is given, so an explicit --timeout on the
        # CLI always takes effect (pytest-timeout precedence: marker > CLI > ini).
        if config.getoption("timeout", default=None) is None:
            for item in items:
                if "e2e" in item.keywords and item.get_closest_marker("timeout") is None:
                    item.add_marker(pytest.mark.timeout(E2E_DEFAULT_TIMEOUT))

        # Skip e2e_run / e2e_serve unless explicitly opted-in via -m
        _disabled_commands = {
            "e2e_run": "winml run is not yet enabled (see #892)",
            "e2e_serve": "winml serve is not yet enabled (see #892)",
        }
        for marker_name, reason in _disabled_commands.items():
            if marker_name not in marker_expr:
                skip_marker = pytest.mark.skip(reason=reason)
                for item in items:
                    if marker_name in item.keywords:
                        item.add_marker(skip_marker)

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
