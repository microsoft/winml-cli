# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the winml catalog CLI command (no network calls, catalog mocked)."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from winml.modelkit.commands.catalog import (
    _build_list_renderable,
    _filter_by_device,
    _filter_by_ep,
    _filter_models,
    _fmt_model_id,
    _fmt_size,
    _make_ep_col_fn_for_device,
    _make_ep_col_fn_for_ep,
    _type_color,
    catalog,
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
            "size_mb": 104.4,
            "task": "fill-mask",
            "supported_eps": {
                "cpu": ["CPU"],
                "dml": ["GPU"],
                "qnn": ["GPU", "NPU"],
                "openvino": ["CPU", "GPU", "NPU"],
            },
        },
        {
            "model_id": "dslim/bert-base-NER",
            "model_type": "bert",
            "size_mb": 104.4,
            "task": "token-classification",
            "supported_eps": {
                "cpu": ["CPU"],
                "dml": ["GPU"],
                "qnn": ["GPU", "NPU"],
                "openvino": ["CPU", "GPU", "NPU"],
            },
        },
        {
            # OV only (no QNN, no VitisAI)
            "model_id": "facebook/detr-resnet-50",
            "model_type": "detr",
            "size_mb": 159.6,
            "task": "object-detection",
            "supported_eps": {
                "cpu": ["CPU"],
                "dml": ["GPU"],
                "openvino": ["CPU", "GPU", "NPU"],
            },
        },
        {
            # VitisAI only (for optional EPs)
            "model_id": "openai/clip-vit-base-patch32",
            "model_type": "clip",
            "size_mb": 354.2,
            "task": "zero-shot-image-classification",
            "supported_eps": {
                "cpu": ["CPU"],
                "dml": ["GPU"],
                "vitisai": ["NPU"],
            },
        },
    ],
}


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def patched_catalog():
    """Patch _load_catalog to return MINIMAL_CATALOG."""
    with patch("winml.modelkit.commands.catalog._load_catalog", return_value=MINIMAL_CATALOG):
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


def test_fmt_model_id_org_prefix_dimmed():
    """Full model ID should be rendered in uniform cyan bold."""
    t = _fmt_model_id("openai/clip-vit-base-patch32")
    spans = list(t._spans)
    # Single span covers the entire model ID in cyan bold
    assert len(spans) == 1
    span = spans[0]
    assert span.start == 0
    assert span.end == len("openai/clip-vit-base-patch32")
    assert "cyan" in str(span.style)
    assert "bold" in str(span.style)


def test_fmt_model_id_no_org():
    t = _fmt_model_id("bert-base-uncased")
    assert t.plain == "bert-base-uncased"


def test_fmt_model_id_overflow_is_ascii_safe():
    """Text overflow must use 'fold', not 'ellipsis' (regression: #233).

    'ellipsis' emits U+2026 (…) which is unrepresentable in cp1252 terminals.
    'fold' wraps long text onto the next line without emitting non-ASCII chars.
    """
    t = _fmt_model_id("org/model-name")
    assert t.overflow == "fold", (
        f"Expected overflow='fold', got {t.overflow!r}. "
        "Using 'ellipsis' would emit U+2026 (…) on cp1252 terminals."
    )


# ---------------------------------------------------------------------------
# _fmt_size unit tests
# ---------------------------------------------------------------------------


def test_fmt_size_mb():
    assert _fmt_size(104.4) == "104MB"


def test_fmt_size_gb():
    assert _fmt_size(1536.0) == "1.5GB"


def test_fmt_size_none():
    assert _fmt_size(None) == "\u2014"


def test_fmt_size_boundary():
    assert _fmt_size(1024.0) == "1.0GB"


# ---------------------------------------------------------------------------
# _type_color unit tests
# ---------------------------------------------------------------------------


def test_type_color_returns_palette_member():
    from winml.modelkit.commands.catalog import _TYPE_PALETTE

    assert _type_color("bert") in _TYPE_PALETTE


def test_type_color_is_deterministic():
    assert _type_color("vit") == _type_color("vit")


def test_type_color_differs_across_types():
    # Not all types should map to the same color (palette has 6 slots)
    colors = {_type_color(t) for t in ["bert", "vit", "swin", "clip", "detr", "roberta"]}
    assert len(colors) > 1


# ---------------------------------------------------------------------------
# CLI integration tests via CliRunner
# ---------------------------------------------------------------------------


def test_catalog_default_shows_table(runner, patched_catalog, tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(catalog, ["--output", str(out)])
    assert result.exit_code == 0
    assert "WinML CLI Catalog" in result.output
    assert "4 validated model(s)" in result.output
    assert "bert" in result.output
    assert "detr" in result.output


def test_catalog_table_shows_size_column():
    buf = StringIO()
    wide = Console(file=buf, width=120, highlight=False)
    wide.print(_build_list_renderable(MINIMAL_CATALOG["models"]))
    rendered = buf.getvalue()
    assert "Size" in rendered
    assert "104MB" in rendered


def test_catalog_saves_json_file(runner, patched_catalog, tmp_path):
    out = tmp_path / "catalog.json"
    result = runner.invoke(catalog, ["--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == 4
    first = data[0]
    assert "model_id" in first
    assert "model_type" in first
    assert "task" in first
    assert "supported_eps" in first
    assert "size_mb" in first


def test_catalog_existing_output_blocked_without_overwrite(runner, patched_catalog, tmp_path):
    """An existing --output is not clobbered unless --overwrite is passed."""
    out = tmp_path / "catalog.json"
    out.write_text("ORIGINAL")
    result = runner.invoke(catalog, ["--output", str(out)])
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert "--overwrite" in result.output
    # The original file is left untouched.
    assert out.read_text() == "ORIGINAL"


def test_catalog_existing_output_replaced_with_overwrite(runner, patched_catalog, tmp_path):
    """--overwrite allows replacing an existing output file."""
    out = tmp_path / "catalog.json"
    out.write_text("ORIGINAL")
    result = runner.invoke(catalog, ["--output", str(out), "--overwrite"])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == 4


def test_catalog_filter_model_type(runner, patched_catalog, tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(catalog, ["--model-type", "bert", "--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert all(m["model_type"] == "bert" for m in data)
    assert len(data) == 2


def test_catalog_filter_task(runner, patched_catalog, tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(catalog, ["--task", "fill-mask", "--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["model_id"] == "google-bert/bert-base-uncased"


def test_catalog_no_match_shows_message(runner, patched_catalog, tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(catalog, ["--model-type", "llama", "--output", str(out)])
    assert result.exit_code == 0
    assert "No models match" in result.output


def test_catalog_saves_path_shown_in_output(runner, patched_catalog, tmp_path):
    out = tmp_path / "my_catalog.json"
    result = runner.invoke(catalog, ["--output", str(out)])
    assert result.exit_code == 0
    assert "Results saved to:" in result.output
    assert "my_catalog.json" in result.output


def test_catalog_load_error(runner):
    with patch(
        "winml.modelkit.commands.catalog._load_catalog",
        side_effect=FileNotFoundError("missing"),
    ):
        result = runner.invoke(catalog, [])
    assert result.exit_code != 0
    assert "Failed to load model catalog" in result.output


def test_catalog_real_catalog_loads(runner, tmp_path):
    """Smoke test: real hub_models.json is loadable and returns expected fields."""
    out = tmp_path / "catalog.json"
    result = runner.invoke(catalog, ["--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert len(data) > 0
    for entry in data:
        assert "model_id" in entry
        assert "model_type" in entry
        assert "task" in entry
        assert "supported_eps" in entry
        assert "size_mb" in entry


# ---------------------------------------------------------------------------
# _filter_by_ep unit tests
# ---------------------------------------------------------------------------


def test_filter_by_ep_none_returns_all():
    result = _filter_by_ep(MINIMAL_CATALOG["models"], None)
    assert len(result) == 4


def test_filter_by_ep_qnn_alias():
    # bert models have QNN; detr (OV only) and clip (VitisAI only) do not
    result = _filter_by_ep(MINIMAL_CATALOG["models"], "qnn")
    assert len(result) == 2
    assert all("qnn" in m["supported_eps"] for m in result)


def test_filter_by_ep_openvino_alias():
    # bert (x2) and detr have OV; clip does not
    result = _filter_by_ep(MINIMAL_CATALOG["models"], "openvino")
    assert len(result) == 3
    assert all("openvino" in m["supported_eps"] for m in result)


def test_filter_by_ep_vitisai_alias():
    # Only clip has VitisAI EP
    result = _filter_by_ep(MINIMAL_CATALOG["models"], "vitisai")
    assert len(result) == 1
    assert result[0]["model_id"] == "openai/clip-vit-base-patch32"


def test_filter_by_ep_dml_returns_all():
    # DML is always-on for all catalog models
    result = _filter_by_ep(MINIMAL_CATALOG["models"], "dml")
    assert len(result) == 4


def test_filter_by_ep_cpu_returns_all():
    # CPU (MLAS) is always-on for all catalog models
    result = _filter_by_ep(MINIMAL_CATALOG["models"], "cpu")
    assert len(result) == 4


def test_filter_by_ep_full_name():
    result = _filter_by_ep(MINIMAL_CATALOG["models"], "QNNExecutionProvider")
    assert len(result) == 2


def test_filter_by_ep_unknown_ep_returns_empty():
    result = _filter_by_ep(MINIMAL_CATALOG["models"], "nv_tensorrt_rtx")
    assert result == []


# ---------------------------------------------------------------------------
# _filter_by_device unit tests
# ---------------------------------------------------------------------------


def test_filter_by_device_none_returns_all():
    result = _filter_by_device(MINIMAL_CATALOG["models"], None)
    assert len(result) == 4


def test_filter_by_device_cpu_returns_all():
    # MLAS always-on → CPU always supported → all models
    result = _filter_by_device(MINIMAL_CATALOG["models"], "CPU")
    assert len(result) == 4


def test_filter_by_device_gpu_returns_all():
    # DML always-on → GPU always supported → all models
    result = _filter_by_device(MINIMAL_CATALOG["models"], "GPU")
    assert len(result) == 4


def test_filter_by_device_npu():
    # NPU needs QNN EP, OV EP, or VitisAI EP
    # bert: QNN+OV, bert: QNN+OV, detr: OV, clip: VitisAI → all 4 match
    result = _filter_by_device(MINIMAL_CATALOG["models"], "NPU")
    assert len(result) == 4


def test_filter_by_device_npu_partial():
    """Models with no optional EP are excluded from NPU results."""
    models_no_eps = [{"model_id": "x", "model_type": "t", "task": "t", "supported_eps": []}]
    result = _filter_by_device(models_no_eps, "NPU")
    assert result == []


def test_filter_by_device_case_insensitive():
    result_upper = _filter_by_device(MINIMAL_CATALOG["models"], "NPU")
    result_lower = _filter_by_device(MINIMAL_CATALOG["models"], "npu")
    assert len(result_upper) == len(result_lower)


# ---------------------------------------------------------------------------
# _make_ep_col_fn_for_ep unit tests
# ---------------------------------------------------------------------------


def test_make_ep_col_fn_for_ep_header():
    header, _ = _make_ep_col_fn_for_ep("QNNExecutionProvider")
    assert header == "Devices"


def test_make_ep_col_fn_for_ep_qnn_devices():
    # Devices are read from the model's supported_eps tuples
    _, fn = _make_ep_col_fn_for_ep("QNNExecutionProvider")
    bert = MINIMAL_CATALOG["models"][0]  # has QNN on GPU and NPU
    assert fn(bert) == "GPU / NPU"


def test_make_ep_col_fn_for_ep_ov_devices():
    _, fn = _make_ep_col_fn_for_ep("OpenVINOExecutionProvider")
    bert = MINIMAL_CATALOG["models"][0]  # has OV on CPU, GPU, NPU
    assert fn(bert) == "CPU / GPU / NPU"


def test_make_ep_col_fn_for_ep_vitisai_devices():
    _, fn = _make_ep_col_fn_for_ep("VitisAIExecutionProvider")
    clip = MINIMAL_CATALOG["models"][3]  # has VitisAI on NPU only
    assert fn(clip) == "NPU"


def test_make_ep_col_fn_for_ep_dml_devices():
    _, fn = _make_ep_col_fn_for_ep("DmlExecutionProvider")
    bert = MINIMAL_CATALOG["models"][0]  # all models have DML → GPU
    assert fn(bert) == "GPU"


def test_make_ep_col_fn_for_ep_cpu_devices():
    _, fn = _make_ep_col_fn_for_ep("CPUExecutionProvider")
    bert = MINIMAL_CATALOG["models"][0]  # all models have CPU EP → CPU
    assert fn(bert) == "CPU"


# ---------------------------------------------------------------------------
# _make_ep_col_fn_for_device unit tests
# ---------------------------------------------------------------------------


def test_make_ep_col_fn_for_device_header():
    header, _ = _make_ep_col_fn_for_device("NPU")
    assert header == "EPs"


def test_make_ep_col_fn_for_device_npu_all_eps():
    """Model with QNN+OpenVINO shows both labels in supported_eps key order."""
    _, fn = _make_ep_col_fn_for_device("NPU")
    bert = MINIMAL_CATALOG["models"][0]  # google-bert: qnn before openvino in JSON
    assert fn(bert) == "QNN / OPENVINO"


def test_make_ep_col_fn_for_device_npu_openvino_only():
    _, fn = _make_ep_col_fn_for_device("NPU")
    detr = MINIMAL_CATALOG["models"][2]  # openvino only
    assert fn(detr) == "OPENVINO"


def test_make_ep_col_fn_for_device_npu_vitisai_only():
    _, fn = _make_ep_col_fn_for_device("NPU")
    clip = MINIMAL_CATALOG["models"][3]  # vitisai only
    assert fn(clip) == "VITISAI"


def test_make_ep_col_fn_for_device_cpu_always_present():
    """cpu alias is always-on → every model's CPU cell includes it."""
    _, fn = _make_ep_col_fn_for_device("CPU")
    for m in MINIMAL_CATALOG["models"]:
        assert "CPU" in fn(m)


def test_make_ep_col_fn_for_device_cpu_openvino_added_when_present():
    _, fn = _make_ep_col_fn_for_device("CPU")
    bert = MINIMAL_CATALOG["models"][0]  # has cpu + openvino (both support CPU)
    assert fn(bert) == "CPU / OPENVINO"


def test_make_ep_col_fn_for_device_gpu_shows_all_gpu_eps():
    _, fn = _make_ep_col_fn_for_device("GPU")
    bert = MINIMAL_CATALOG["models"][0]  # dml + qnn + openvino all support GPU
    assert fn(bert) == "DML / QNN / OPENVINO"


def test_make_ep_col_fn_for_device_gpu_dml_always_present():
    """DML is in every model's supported_eps → always appears in GPU column."""
    _, fn = _make_ep_col_fn_for_device("GPU")
    clip = MINIMAL_CATALOG["models"][3]  # vitisai only for optional EPs
    assert fn(clip) == "DML"


def test_make_ep_col_fn_for_device_cpu_without_ov():
    """Model with no ov EP shows only 'CPU' for CPU column."""
    _, fn = _make_ep_col_fn_for_device("CPU")
    clip = MINIMAL_CATALOG["models"][3]  # vitisai only for optional EPs
    assert fn(clip) == "CPU"
    assert "OV" not in fn(clip)


def test_make_ep_col_fn_for_device_no_eps_returns_dash():
    _, fn = _make_ep_col_fn_for_device("NPU")
    assert fn({"supported_eps": {}}) == "\u2014"


def test_make_ep_col_fn_for_device_gpu_no_eps_returns_dash():
    _, fn = _make_ep_col_fn_for_device("GPU")
    assert fn({"supported_eps": {}}) == "\u2014"


# ---------------------------------------------------------------------------
# EP / Device column visibility tests
# ---------------------------------------------------------------------------


def test_ep_col_hidden_by_default(runner, patched_catalog):
    result = runner.invoke(catalog, [])
    assert result.exit_code == 0
    assert "Devices" not in result.output
    assert "EPs" not in result.output


def test_ep_col_shown_when_header_given():
    _, fn = _make_ep_col_fn_for_ep("QNNExecutionProvider")
    buf = StringIO()
    wide = Console(file=buf, width=160, highlight=False)
    wide.print(
        _build_list_renderable(MINIMAL_CATALOG["models"], ep_col_header="Devices", ep_col_fn=fn)
    )
    assert "Devices" in buf.getvalue()


def test_ep_col_hidden_when_no_header():
    buf = StringIO()
    wide = Console(file=buf, width=160, highlight=False)
    wide.print(_build_list_renderable(MINIMAL_CATALOG["models"]))
    assert "Devices" not in buf.getvalue()
    assert "EPs" not in buf.getvalue()


def test_cli_ep_filter_shows_devices_col(runner, patched_catalog):
    result = runner.invoke(catalog, ["--ep", "qnn"])
    assert result.exit_code == 0
    assert "Devices" in result.output


def test_cli_device_filter_shows_eps_col(runner, patched_catalog):
    result = runner.invoke(catalog, ["--device", "NPU"])
    assert result.exit_code == 0
    assert "EPs" in result.output


def test_cli_ep_and_device_hides_extra_col(runner, patched_catalog):
    result = runner.invoke(catalog, ["--ep", "qnn", "--device", "NPU"])
    assert result.exit_code == 0
    assert "Devices" not in result.output
    assert "EPs" not in result.output


def test_ep_filter_shows_correct_devices():
    """--ep qnn → Devices column shows 'GPU / NPU' for every row."""
    buf = StringIO()
    wide = Console(file=buf, width=160, highlight=False)
    _, fn = _make_ep_col_fn_for_ep("QNNExecutionProvider")
    wide.print(
        _build_list_renderable(MINIMAL_CATALOG["models"][:2], ep_col_header="Devices", ep_col_fn=fn)
    )
    rendered = buf.getvalue()
    assert rendered.count("GPU / NPU") >= 2


def test_device_filter_shows_correct_eps():
    """--device NPU → EPs column shows per-model EP strings."""
    buf = StringIO()
    wide = Console(file=buf, width=160, highlight=False)
    _, fn = _make_ep_col_fn_for_device("NPU")
    wide.print(_build_list_renderable(MINIMAL_CATALOG["models"], ep_col_header="EPs", ep_col_fn=fn))
    rendered = buf.getvalue()
    assert "QNN / OPENVINO" in rendered  # bert models (JSON key order: qnn before openvino)
    assert "VITISAI" in rendered  # clip model


def test_cli_ep_filter_narrows_results(runner, patched_catalog, tmp_path):
    out = tmp_path / "out.json"
    result = runner.invoke(catalog, ["--ep", "vitisai", "--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["model_id"] == "openai/clip-vit-base-patch32"


def test_cli_device_npu_narrows_results(runner, patched_catalog, tmp_path):
    # All 4 fixture models have at least one NPU EP → all returned
    out = tmp_path / "out.json"
    result = runner.invoke(catalog, ["--device", "NPU", "--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert len(data) == 4


# ---------------------------------------------------------------------------
# --format json tests
# ---------------------------------------------------------------------------


class TestCatalogFormatJson:
    """Tests for --format json output."""

    def test_format_json_outputs_valid_json(self, runner, patched_catalog):
        result = runner.invoke(catalog, ["--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert isinstance(data, list)
        assert len(data) == 4

    def test_format_json_with_task_filter(self, runner, patched_catalog):
        result = runner.invoke(catalog, ["--format", "json", "--task", "fill-mask"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert len(data) == 1
        assert data[0]["model_id"] == "google-bert/bert-base-uncased"

    def test_format_json_with_ep_filter(self, runner, patched_catalog):
        result = runner.invoke(catalog, ["--format", "json", "--ep", "qnn"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        # Only models with QNN EP: bert-base-uncased, bert-base-NER
        assert len(data) == 2

    def test_format_json_with_device_filter(self, runner, patched_catalog):
        result = runner.invoke(catalog, ["--format", "json", "--device", "NPU"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert len(data) == 4

    def test_format_json_contains_expected_keys(self, runner, patched_catalog):
        result = runner.invoke(catalog, ["--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        for model in data:
            assert "model_id" in model
            assert "model_type" in model
            assert "supported_eps" in model
