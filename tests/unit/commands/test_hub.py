# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the winml hub CLI command (no network calls, catalog mocked)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.hub import (
    _filter_models,
    _fmt_model_id,
    _overall_verdict,
    hub,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CATALOG = {
    "version": "1.0",
    "models": [
        {
            "model_id": "google-bert/bert-base-uncased",
            "model_type": "bert",
            "task": "fill-mask",
            "perf": None,
            "accuracy": None,
        },
        {
            "model_id": "dslim/bert-base-NER",
            "model_type": "bert",
            "task": "token-classification",
            "perf": {
                "QNN": {
                    "avg_ms": 13.71,
                    "p50_ms": 13.75,
                    "p90_ms": 13.84,
                    "p95_ms": 13.84,
                    "p99_ms": 13.84,
                    "min_ms": 13.59,
                    "max_ms": 13.84,
                    "throughput_qps": 72.93,
                },
                "OV": {
                    "avg_ms": 25.28,
                    "p50_ms": 24.84,
                    "p90_ms": 35.33,
                    "p95_ms": 35.33,
                    "p99_ms": 35.33,
                    "min_ms": 20.6,
                    "max_ms": 35.33,
                    "throughput_qps": 39.56,
                },
            },
            "accuracy": {
                "QNN": {"verdict": "PASS", "drop_pct": 0.0},
                "OV": {"verdict": "PASS", "drop_pct": 0.0},
            },
        },
        {
            "model_id": "facebook/detr-resnet-50",
            "model_type": "detr",
            "task": "object-detection",
            "perf": None,
            "accuracy": {
                "QNN": {"verdict": "REGRESSION", "drop_pct": -36.84},
                "OV": {"verdict": "REGRESSION", "drop_pct": -32.67},
            },
        },
        {
            "model_id": "openai/clip-vit-base-patch32",
            "model_type": "clip",
            "task": "zero-shot-image-classification",
            "perf": None,
            "accuracy": None,
        },
    ],
}


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def patched_catalog():
    """Patch _load_catalog to return MINIMAL_CATALOG."""
    with patch("winml.modelkit.commands.hub._load_catalog", return_value=MINIMAL_CATALOG):
        yield


# ---------------------------------------------------------------------------
# _filter_models unit tests
# ---------------------------------------------------------------------------


def test_filter_no_filters_returns_all():
    result = _filter_models(MINIMAL_CATALOG["models"], model_type=None, task=None)
    assert len(result) == 4


def test_filter_by_model_type():
    result = _filter_models(MINIMAL_CATALOG["models"], model_type="bert", task=None)
    assert len(result) == 2
    assert all(m["model_type"] == "bert" for m in result)


def test_filter_by_task():
    result = _filter_models(MINIMAL_CATALOG["models"], model_type=None, task="fill-mask")
    assert len(result) == 1
    assert result[0]["model_id"] == "google-bert/bert-base-uncased"


def test_filter_model_type_case_insensitive():
    result = _filter_models(MINIMAL_CATALOG["models"], model_type="BERT", task=None)
    assert len(result) == 2


def test_filter_no_match_returns_empty():
    result = _filter_models(MINIMAL_CATALOG["models"], model_type="llama", task=None)
    assert result == []


# ---------------------------------------------------------------------------
# Formatting helpers unit tests
# ---------------------------------------------------------------------------


def test_fmt_model_id_with_org():
    t = _fmt_model_id("openai/clip-vit-base-patch32")
    assert "openai/" in t.plain
    assert "clip-vit-base-patch32" in t.plain


def test_fmt_model_id_no_org():
    t = _fmt_model_id("bert-base-uncased")
    assert t.plain == "bert-base-uncased"


# ---------------------------------------------------------------------------
# _overall_verdict unit tests
# ---------------------------------------------------------------------------


def test_overall_verdict_priority():
    assert _overall_verdict({"QNN": {"verdict": "PASS"}, "OV": {"verdict": "PASS"}}) == "PASS"
    assert _overall_verdict({"QNN": {"verdict": "AT_RISK"}, "OV": {"verdict": "PASS"}}) == "AT_RISK"
    assert (
        _overall_verdict({"QNN": {"verdict": "REGRESSION"}, "OV": {"verdict": "AT_RISK"}})
        == "REGRESSION"
    )


# ---------------------------------------------------------------------------
# CLI integration tests via CliRunner
# ---------------------------------------------------------------------------


def test_hub_default_shows_table(runner, patched_catalog):
    result = runner.invoke(hub, ["--output", "/dev/null"])
    assert result.exit_code == 0
    assert "ModelKit Catalog" in result.output
    assert "4 validated model(s)" in result.output
    assert "bert" in result.output
    assert "detr" in result.output


def test_hub_table_shows_hint(runner, patched_catalog):
    result = runner.invoke(hub, ["--output", "/dev/null"])
    assert result.exit_code == 0
    assert "winml hub --model" in result.output


def test_hub_saves_json_file(runner, patched_catalog, tmp_path):
    out = tmp_path / "catalog.json"
    result = runner.invoke(hub, ["--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == 4
    first = data[0]
    assert "model_id" in first
    assert "model_type" in first
    assert "task" in first
    assert "perf" in first
    assert "accuracy" in first


def test_hub_shows_accuracy_pass(runner, patched_catalog, tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(hub, ["--model-type", "bert", "--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    verdicts = {
        ep: info["verdict"] for m in data if m.get("accuracy") for ep, info in m["accuracy"].items()
    }
    assert "PASS" in verdicts.values()


def test_hub_shows_accuracy_regression(runner, patched_catalog, tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(hub, ["--model-type", "detr", "--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert len(data) == 1
    verdicts = [info["verdict"] for info in data[0]["accuracy"].values()]
    assert "REGRESSION" in verdicts


def test_hub_filter_model_type(runner, patched_catalog, tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(hub, ["--model-type", "bert", "--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert all(m["model_type"] == "bert" for m in data)
    assert len(data) == 2


def test_hub_filter_task(runner, patched_catalog, tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(hub, ["--task", "fill-mask", "--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["model_id"] == "google-bert/bert-base-uncased"


def test_hub_no_match_shows_message(runner, patched_catalog):
    result = runner.invoke(hub, ["--model-type", "llama", "--output", "/dev/null"])
    assert result.exit_code == 0
    assert "No models match" in result.output


def test_hub_json_accuracy_structure(runner, patched_catalog, tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(
        hub, ["--model-type", "bert", "--task", "token-classification", "--output", str(out)]
    )
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert len(data) == 1
    acc = data[0]["accuracy"]
    assert acc["QNN"]["verdict"] == "PASS"
    assert acc["QNN"]["drop_pct"] == 0.0


def test_hub_saves_path_shown_in_output(runner, patched_catalog, tmp_path):
    out = tmp_path / "my_catalog.json"
    result = runner.invoke(hub, ["--output", str(out)])
    assert result.exit_code == 0
    assert "Results saved to:" in result.output
    assert "my_catalog.json" in result.output


def test_hub_model_detail_shows_perf(runner, patched_catalog, tmp_path):
    result = runner.invoke(hub, ["--model", "dslim/bert-base-NER", "--output", "/dev/null"])
    assert result.exit_code == 0
    assert "Latency (ms)" in result.output
    assert "QNN" in result.output
    assert "OV" in result.output


def test_hub_model_detail_shows_accuracy(runner, patched_catalog, tmp_path):
    result = runner.invoke(hub, ["--model", "dslim/bert-base-NER", "--output", "/dev/null"])
    assert result.exit_code == 0
    assert "Accuracy" in result.output
    assert "PASS" in result.output


def test_hub_model_detail_regression(runner, patched_catalog):
    result = runner.invoke(hub, ["--model", "facebook/detr-resnet-50", "--output", "/dev/null"])
    assert result.exit_code == 0
    assert "REGRESSION" in result.output
    assert "-36.84%" in result.output


def test_hub_model_detail_no_data(runner, patched_catalog):
    result = runner.invoke(
        hub, ["--model", "openai/clip-vit-base-patch32", "--output", "/dev/null"]
    )
    assert result.exit_code == 0
    assert "No benchmark data" in result.output


def test_hub_model_saves_json(runner, patched_catalog, tmp_path):
    out = tmp_path / "model.json"
    result = runner.invoke(hub, ["--model", "dslim/bert-base-NER", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["model_id"] == "dslim/bert-base-NER"
    assert "perf" in data
    assert "accuracy" in data


def test_hub_model_partial_match(runner, patched_catalog):
    # "base-NER" matches exactly one model in MINIMAL_CATALOG
    result = runner.invoke(hub, ["--model", "base-NER", "--output", "/dev/null"])
    assert result.exit_code == 0
    assert "token-classification" in result.output


def test_hub_model_not_found(runner, patched_catalog):
    result = runner.invoke(hub, ["--model", "nonexistent-model-xyz"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_hub_model_ambiguous(runner, patched_catalog):
    # "bert" matches multiple model IDs
    result = runner.invoke(hub, ["--model", "bert"])
    assert result.exit_code != 0
    assert "Ambiguous" in result.output


def test_hub_catalog_load_error(runner):
    with patch(
        "winml.modelkit.commands.hub._load_catalog",
        side_effect=FileNotFoundError("missing"),
    ):
        result = runner.invoke(hub, [])
    assert result.exit_code != 0
    assert "Failed to load model catalog" in result.output


def test_hub_real_catalog_loads(runner, tmp_path):
    """Smoke test: real hub_models.json is loadable and returns expected fields."""
    out = tmp_path / "catalog.json"
    result = runner.invoke(hub, ["--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert len(data) > 0
    for entry in data:
        assert "model_id" in entry
        assert "model_type" in entry
        assert "task" in entry
        assert "perf" in entry
        assert "accuracy" in entry
