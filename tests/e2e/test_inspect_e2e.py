"""E2E tests for the inspect CLI command.

These tests exercise the full inspect pipeline with REAL models
downloaded from HuggingFace Hub. They validate JSON output structure
and content for various model-task combinations.

Note: The inspect command's validate_task() has a limited task vocabulary
(feature-extraction, image-feature-extraction, image-segmentation,
mask-generation, next-sentence-prediction). Tasks outside this set are
rejected when passed via --task override. Auto-detect (no --task) uses
TasksManager directly and supports a broader set of tasks.

Markers:
    e2e: Full end-to-end test with real models
    network: Requires network access to HuggingFace Hub
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.inspect import inspect


pytestmark = [pytest.mark.e2e, pytest.mark.network]

# ---------------------------------------------------------------------------
# Helpers
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


def _run_inspect(model: str, task: str | None = None) -> dict:
    """Invoke the inspect command and return parsed JSON output.

    Raises AssertionError when the command exits non-zero or the
    output is not valid JSON.
    """
    runner = CliRunner()
    args = ["-m", model, "-f", "json"]
    if task:
        args.extend(["-t", task])
    result = runner.invoke(inspect, args, obj={}, catch_exceptions=False)
    assert result.exit_code == 0, (
        f"inspect failed (exit {result.exit_code}):\n{result.output}"
    )
    return json.loads(result.output)


def _assert_common_structure(data: dict, model_id: str, expected_task: str) -> None:
    """Assert the standard JSON structure returned by inspect."""
    # All top-level keys present
    assert EXPECTED_TOP_KEYS.issubset(data.keys()), (
        f"Missing keys: {EXPECTED_TOP_KEYS - data.keys()}"
    )

    assert data["model_id"] == model_id
    assert data["task"] == expected_task

    # Loader section
    loader = data["loader"]
    assert "hf_model_class" in loader
    assert "support_level" in loader

    # Exporter section
    exporter = data["exporter"]
    assert "onnx_config_class" in exporter
    assert "support_level" in exporter
    assert isinstance(exporter.get("input_tensors"), list)
    assert isinstance(exporter.get("output_tensors"), list)

    # WinML section
    winml = data["winml"]
    assert "winml_class" in winml
    assert "support_level" in winml


# ===========================================================================
# BERT
# ===========================================================================

class TestInspectBert:
    """Inspect bert-base-uncased with auto-detect and explicit tasks."""

    MODEL = "bert-base-uncased"

    def test_auto_detect_fill_mask(self):
        """Auto-detect should resolve BERT to fill-mask via TasksManager."""
        data = _run_inspect(self.MODEL)
        _assert_common_structure(data, self.MODEL, "fill-mask")
        assert data["model_type"] == "bert"
        assert data["task_source"] == "TasksManager"

    def test_feature_extraction(self):
        """feature-extraction is in the known task list; explicit override works."""
        data = _run_inspect(self.MODEL, task="feature-extraction")
        _assert_common_structure(data, self.MODEL, "feature-extraction")
        assert data["model_type"] == "bert"

    def test_explicit_unknown_task_rejected(self):
        """Tasks not in validate_task vocabulary are cleanly rejected."""
        runner = CliRunner()
        args = ["-m", self.MODEL, "-f", "json", "-t", "text-classification"]
        result = runner.invoke(inspect, args, obj={})
        assert result.exit_code != 0
        assert "Unknown task" in result.output

    def test_next_sentence_prediction(self):
        """next-sentence-prediction is in the known task list.

        We assert it either succeeds with valid JSON or fails with a
        clean ClickException (non-zero exit code), but never crashes
        with an unhandled traceback.
        """
        runner = CliRunner()
        args = ["-m", self.MODEL, "-f", "json", "-t", "next-sentence-prediction"]
        result = runner.invoke(inspect, args, obj={})
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert "model_id" in data
        else:
            # Should be a clean error, not a raw traceback
            assert "Traceback (most recent call last)" not in result.output


# ===========================================================================
# Vision models
# ===========================================================================

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
        data = _run_inspect(model_id)
        _assert_common_structure(data, model_id, "image-classification")
        assert data["model_type"] in {"resnet", "convnext", "vit"}


# ===========================================================================
# CLIP
# ===========================================================================

class TestInspectCLIP:
    """Inspect CLIP with multi-modal tasks."""

    MODEL = "openai/clip-vit-base-patch32"

    def test_auto_detect_feature_extraction(self):
        """Auto-detect should resolve CLIP to feature-extraction."""
        data = _run_inspect(self.MODEL)
        assert data["model_type"] == "clip"
        assert data["task"] in {"feature-extraction", "zero-shot-image-classification"}

    def test_image_feature_extraction(self):
        """image-feature-extraction is in the known task list."""
        data = _run_inspect(self.MODEL, task="image-feature-extraction")
        _assert_common_structure(data, self.MODEL, "image-feature-extraction")
        assert data["model_type"] == "clip"


# ===========================================================================
# DETR
# ===========================================================================

class TestInspectDETR:
    """Inspect DETR with object-detection."""

    MODEL = "facebook/detr-resnet-50"

    def test_auto_detect_object_detection(self):
        """Auto-detect should resolve DETR to object-detection."""
        data = _run_inspect(self.MODEL)
        assert data["model_id"] == self.MODEL
        assert data["model_type"] == "detr"
        assert data["task"] == "object-detection"

    def test_explicit_object_detection_rejected(self):
        """object-detection is NOT in validate_task vocabulary.

        Explicit override should be cleanly rejected, while
        auto-detect (tested above) succeeds.
        """
        runner = CliRunner()
        args = ["-m", self.MODEL, "-f", "json", "-t", "object-detection"]
        result = runner.invoke(inspect, args, obj={})
        assert result.exit_code != 0
        assert "Unknown task" in result.output
