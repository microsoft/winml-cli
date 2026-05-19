# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the inspect CLI command.

Offline tests (no network) use --model-type or --model-class flags and
cover CLI surface (help, no-args errors, format validation), --list-tasks,
and JSON structure invariants.

Network tests use real HuggingFace model IDs (-m flag) and validate
auto-detected task, model_type, and full JSON structure.

Markers:
    e2e: Full end-to-end test (required to run any test in this file)
    network: Requires network access to HuggingFace Hub
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

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


def _run(*args: str) -> object:
    """Invoke inspect with *args and return the CliRunner Result."""
    return CliRunner().invoke(inspect, list(args), obj={})


def _run_json(*args: str) -> dict:
    """Invoke inspect with *args + '-f json' and return parsed JSON.

    Asserts exit_code == 0 and that the output is valid JSON.
    """
    result = _run(*args, "-f", "json")
    assert result.exit_code == 0, f"inspect exited {result.exit_code}:\n{result.output}"
    return json.loads(result.output)


def _run_network(model: str, task: str | None = None) -> dict:
    """Invoke inspect with a real model ID and return parsed JSON output.

    Raises AssertionError when the command exits non-zero or the
    output is not valid JSON.
    """
    args: list[str] = ["-m", model, "-f", "json"]
    if task:
        args.extend(["-t", task])
    result = CliRunner().invoke(inspect, args, obj={}, catch_exceptions=False)
    assert result.exit_code == 0, f"inspect failed (exit {result.exit_code}):\n{result.output}"
    return json.loads(result.output)


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
# CLI surface — no network required
# ===========================================================================


class TestInspectCliSurface:
    """CLI surface: help text, no-args errors, format validation."""

    def test_no_args_exits_usage_error(self):
        """Invoked with no arguments inspect must exit 2 with a UsageError."""
        result = _run()
        assert result.exit_code == 2
        assert "At least one of" in result.output

    def test_help_exits_zero(self):
        """--help must exit 0."""
        result = _run("--help")
        assert result.exit_code == 0

    def test_help_documents_model_flag(self):
        """-m / --model appears in help text."""
        result = _run("--help")
        assert "-m" in result.output
        assert "--model" in result.output

    def test_help_documents_format_flag(self):
        """-f / --format appears in help text."""
        result = _run("--help")
        assert "-f" in result.output
        assert "--format" in result.output

    def test_help_documents_model_type_flag(self):
        """--model-type appears in help text."""
        result = _run("--help")
        assert "--model-type" in result.output

    def test_help_documents_model_class_flag(self):
        """--model-class appears in help text."""
        result = _run("--help")
        assert "--model-class" in result.output

    def test_help_documents_list_tasks_flag(self):
        """--list-tasks appears in help text."""
        result = _run("--help")
        assert "--list-tasks" in result.output

    def test_help_documents_verbose_flag(self):
        """-v / --verbose appears in help text."""
        result = _run("--help")
        assert "-v" in result.output
        assert "--verbose" in result.output

    def test_help_documents_hierarchy_flag(self):
        """-H / --hierarchy appears in help text."""
        result = _run("--help")
        assert "-H" in result.output
        assert "--hierarchy" in result.output

    def test_invalid_format_exits_nonzero(self):
        """An unrecognised --format value must exit non-zero."""
        result = _run("--model-type", "bert", "--format", "xml")
        assert result.exit_code != 0

    def test_invalid_format_names_bad_choice(self):
        """Error output mentions the bad format value or 'choice'."""
        result = _run("--model-type", "bert", "--format", "xml")
        output_lower = result.output.lower()
        assert "xml" in output_lower or "choice" in output_lower or "invalid" in output_lower


# ===========================================================================
# --list-tasks — no network required
# ===========================================================================


class TestInspectListTasks:
    """--list-tasks must exit 0 and print one task per line."""

    def test_list_tasks_exits_zero(self):
        """--list-tasks should not require a model and must exit 0."""
        result = _run("--list-tasks")
        assert result.exit_code == 0, f"--list-tasks exited {result.exit_code}:\n{result.output}"

    def test_list_tasks_output_is_nonempty(self):
        """--list-tasks must print at least one task."""
        result = _run("--list-tasks")
        assert result.exit_code == 0
        lines = [line.strip() for line in result.output.splitlines() if line.strip()]
        assert len(lines) > 0, "Expected at least one task line"

    def test_list_tasks_all_lines_are_strings(self):
        """Every line printed by --list-tasks must be a non-empty string."""
        result = _run("--list-tasks")
        assert result.exit_code == 0
        for line in result.output.splitlines():
            if line.strip():
                assert isinstance(line.strip(), str)
                assert len(line.strip()) > 0

    def test_list_tasks_includes_known_tasks(self):
        """Output must include ModelKit-registered tasks."""
        result = _run("--list-tasks")
        assert result.exit_code == 0
        tasks = {line.strip() for line in result.output.splitlines() if line.strip()}
        assert "feature-extraction" in tasks
        assert "mask-generation" in tasks

    def test_list_tasks_no_model_flag_needed(self):
        """--list-tasks works without -m, --model-type, or --model-class."""
        result = _run("--list-tasks")
        # exit 0 means no UsageError was raised about missing model
        assert result.exit_code == 0

    def test_list_tasks_overrides_missing_model_requirement(self):
        """--list-tasks must not require any model argument."""
        # Confirm that without --list-tasks we'd get exit 2
        no_model_result = _run()
        assert no_model_result.exit_code == 2
        # With --list-tasks we must get exit 0
        result = _run("--list-tasks")
        assert result.exit_code == 0


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
