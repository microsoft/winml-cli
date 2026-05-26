# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the inspect CLI command.

Offline tests (no network) use --model-type or --model-class flags and
validate JSON structure invariants without downloading any model weights.

Network tests use real HuggingFace model IDs (-m flag) and validate
auto-detected task, model_type, and full JSON structure.

CLI surface and --list-tasks tests live in tests/cli/test_inspect_cli.py.

Markers:
    e2e: Full end-to-end test (required to run any test in this file)
    network: Requires network access to HuggingFace Hub
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from tests._helpers import run_inspect as _run
from winml.modelkit.commands.inspect import inspect


pytestmark = [pytest.mark.e2e]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_TOP_KEYS = {
    "model_id",
    "model_type",
    "architectures",
    "task",
    "task_source",
    "overall_support",
    "support_notes",
    "loader",
    "exporter",
    "winml",
    "cache",
    "hierarchy",
    "processor",
    "io_config",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_json(*args: str) -> dict:
    """Invoke inspect with *args + '-f json' and return parsed JSON.

    Reads ``result.stdout`` (not ``result.output``) so the banner and
    spinner emitted on stderr do not corrupt the JSON payload — Click 8.4's
    ``result.output`` aggregates both streams.
    """
    result = _run(*args, "-f", "json")
    assert result.exit_code == 0, f"inspect exited {result.exit_code}:\n{result.output}"
    return json.loads(result.stdout)


def _run_network(model: str, task: str | None = None) -> dict:
    """Invoke inspect with a real model ID and return parsed JSON output.

    Reads ``result.stdout`` (not ``result.output``) so the banner and
    spinner emitted on stderr do not corrupt the JSON payload — Click 8.4's
    ``result.output`` aggregates both streams.
    """
    args: list[str] = ["-m", model, "-f", "json"]
    if task:
        args.extend(["-t", task])
    result = CliRunner().invoke(inspect, args, obj={}, catch_exceptions=False)
    assert result.exit_code == 0, f"inspect failed (exit {result.exit_code}):\n{result.output}"
    return json.loads(result.stdout)


def _assert_common_structure(data: dict, model_id: str, expected_task: str) -> None:
    """Assert the standard JSON structure returned by inspect."""
    assert EXPECTED_TOP_KEYS.issubset(data.keys()), (
        f"Missing keys: {EXPECTED_TOP_KEYS - data.keys()}"
    )
    assert data["model_id"] == model_id
    assert data["task"] == expected_task

    loader = data["loader"]
    assert "hf_model_class" in loader
    assert "support_level" in loader

    exporter = data["exporter"]
    assert "onnx_config_class" in exporter
    assert "support_level" in exporter
    assert isinstance(exporter.get("input_tensors"), list)
    assert isinstance(exporter.get("output_tensors"), list)

    winml = data["winml"]
    assert "winml_class" in winml
    assert "support_level" in winml


# ===========================================================================
# Offline inspection — no network required
# ===========================================================================


class TestInspectModelTypeOnly:
    """Use --model-type / --model-class without downloading any weights."""

    def test_bert_default_task_json(self):
        """--model-type bert resolves to a bert model_type with some task."""
        data = _run_json("--model-type", "bert")
        assert data["model_type"] == "bert"
        assert isinstance(data["task"], str) and data["task"]

    def test_bert_feature_extraction_json(self):
        """--model-type bert -t feature-extraction resolves correctly."""
        data = _run_json("--model-type", "bert", "-t", "feature-extraction")
        assert data["model_type"] == "bert"
        assert data["task"] == "feature-extraction"

    def test_resnet_default_task_json(self):
        """--model-type resnet resolves to a resnet model_type."""
        data = _run_json("--model-type", "resnet")
        assert data["model_type"] == "resnet"
        assert isinstance(data["task"], str) and data["task"]

    def test_model_class_bert_for_masked_lm(self):
        """--model-class BertForMaskedLM resolves to bert / fill-mask."""
        data = _run_json("--model-class", "BertForMaskedLM")
        assert data["model_type"] == "bert"
        assert data["task"] == "fill-mask"

    def test_verbose_flag_accepted(self):
        """--verbose must be accepted without error."""
        data = _run_json("--model-type", "bert", "--verbose")
        assert data["model_type"] == "bert"

    def test_short_verbose_flag_accepted(self):
        """-v short flag must be accepted without error."""
        data = _run_json("--model-type", "bert", "-v")
        assert data["model_type"] == "bert"

    def test_json_output_contains_all_top_level_keys(self):
        """JSON output must include every key in EXPECTED_TOP_KEYS."""
        data = _run_json("--model-type", "bert")
        missing = EXPECTED_TOP_KEYS - data.keys()
        assert not missing, f"Missing top-level keys: {missing}"

    def test_loader_section_structure(self):
        """loader section must have hf_model_class and support_level."""
        data = _run_json("--model-type", "bert")
        loader = data["loader"]
        assert "hf_model_class" in loader
        assert "support_level" in loader

    def test_exporter_section_structure(self):
        """exporter section must have onnx_config_class, support_level, tensors."""
        data = _run_json("--model-type", "bert")
        exporter = data["exporter"]
        assert "onnx_config_class" in exporter
        assert "support_level" in exporter
        assert isinstance(exporter.get("input_tensors"), list)
        assert isinstance(exporter.get("output_tensors"), list)

    def test_winml_section_structure(self):
        """winml section must have winml_class and support_level."""
        data = _run_json("--model-type", "bert")
        winml = data["winml"]
        assert "winml_class" in winml
        assert "support_level" in winml

    def test_table_format_exits_zero(self):
        """Default table format must exit 0 (Rich output is not captured, but exit code is)."""
        result = _run("--model-type", "bert")
        assert result.exit_code == 0

    def test_unknown_model_type_exits_nonzero(self):
        """An unrecognised model type must produce a non-zero exit code."""
        result = _run("--model-type", "totally_nonexistent_model_xyz_123")
        assert result.exit_code != 0

    def test_hierarchy_flag_accepted_without_model(self):
        """--hierarchy flag must be accepted even without a model download."""
        # Without -m, hierarchy_info will be None (skipped), but command should succeed
        data = _run_json("--model-type", "bert", "--hierarchy")
        assert data["model_type"] == "bert"
        assert data["hierarchy"] is None  # hierarchy requires -m


# ===========================================================================
# Network tests — require HuggingFace Hub access
# ===========================================================================


@pytest.mark.network
class TestInspectBert:
    """Inspect bert-base-uncased with auto-detect and explicit tasks."""

    MODEL = "bert-base-uncased"

    def test_auto_detect_fill_mask(self):
        """Auto-detect should resolve BERT to fill-mask via TasksManager."""
        data = _run_network(self.MODEL)
        _assert_common_structure(data, self.MODEL, "fill-mask")
        assert data["model_type"] == "bert"
        assert data["task_source"] == "TasksManager"

    def test_feature_extraction(self):
        """feature-extraction task override must work."""
        data = _run_network(self.MODEL, task="feature-extraction")
        _assert_common_structure(data, self.MODEL, "feature-extraction")
        assert data["model_type"] == "bert"

    def test_next_sentence_prediction(self):
        """next-sentence-prediction task override: clean success or clean error.

        We assert it either succeeds with valid JSON or fails with a
        clean ClickException (non-zero exit code), but never crashes
        with an unhandled traceback.
        """
        result = CliRunner().invoke(
            inspect,
            ["-m", self.MODEL, "-f", "json", "-t", "next-sentence-prediction"],
            obj={},
        )
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert "model_id" in data
        else:
            assert "Traceback (most recent call last)" not in result.output


@pytest.mark.network
class TestInspectVision:
    """Inspect vision models via auto-detect (image-classification)."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "microsoft/resnet-50",
            "facebook/convnext-tiny-224",
            "google/vit-base-patch16-224",
        ],
        ids=["resnet", "convnext", "vit"],
    )
    def test_auto_detect_image_classification(self, model_id: str):
        """Auto-detect should resolve vision models to image-classification."""
        data = _run_network(model_id)
        _assert_common_structure(data, model_id, "image-classification")
        assert data["model_type"] in {"resnet", "convnext", "vit"}


@pytest.mark.network
class TestInspectCLIP:
    """Inspect CLIP with multi-modal tasks."""

    MODEL = "openai/clip-vit-base-patch32"

    def test_auto_detect_feature_extraction(self):
        """Auto-detect should resolve CLIP to feature-extraction."""
        data = _run_network(self.MODEL)
        assert data["model_type"] == "clip"
        assert data["task"] in {"feature-extraction", "zero-shot-image-classification"}

    def test_image_feature_extraction(self):
        """image-feature-extraction task override must work."""
        data = _run_network(self.MODEL, task="image-feature-extraction")
        _assert_common_structure(data, self.MODEL, "image-feature-extraction")
        assert data["model_type"] == "clip"


@pytest.mark.network
class TestInspectDETR:
    """Inspect DETR with object-detection."""

    MODEL = "facebook/detr-resnet-50"

    def test_auto_detect_object_detection(self):
        """Auto-detect should resolve DETR to object-detection."""
        data = _run_network(self.MODEL)
        assert data["model_id"] == self.MODEL
        assert data["model_type"] == "detr"
        assert data["task"] == "object-detection"
