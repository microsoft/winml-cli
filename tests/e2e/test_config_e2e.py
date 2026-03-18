# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the config CLI command.

These tests exercise the full config generation pipeline with REAL models
downloaded from HuggingFace Hub. They validate JSON output structure
for various model-task combinations.

The config command does NOT use @click.pass_context, so no obj={} is needed.

Note: Device resolution (resolve_device) requires hardware detection that
may fail in test environments. We mock it to return ("cpu", ["cpu"]).

Note: The config command writes JSON to stdout via print() and Rich status
messages to stderr via Console(stderr=True). CliRunner captures both in
result.output. We extract JSON by finding the first '{' or '[' character.

Markers:
    e2e: Full end-to-end test with real models
    network: Requires network access to HuggingFace Hub
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.config import config


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [pytest.mark.e2e, pytest.mark.network]


@pytest.fixture(autouse=True)
def _mock_resolve_device():
    """Mock hardware detection to avoid failures in CI/test environments."""
    with patch(
        "winml.modelkit.sysinfo.resolve_device",
        return_value=("cpu", ["cpu"]),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(output: str) -> dict | list:
    """Extract JSON object/array from mixed CLI output.

    The config command mixes Rich status messages (stderr) with JSON
    (stdout) in CliRunner output. Find the first '{' or '[' that
    starts a valid JSON payload.
    """
    # Try to find the start of JSON object
    for start_char in ("{", "["):
        idx = output.find(start_char)
        if idx >= 0:
            candidate = output[idx:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    msg = f"No valid JSON found in output:\n{output[:500]}"
    raise ValueError(msg)


def _run_config(*args: str) -> dict:
    """Invoke the config command and return parsed JSON output."""
    runner = CliRunner()
    result = runner.invoke(config, list(args), catch_exceptions=False)
    assert result.exit_code == 0, (
        f"config failed (exit {result.exit_code}):\n{result.output}"
    )
    return _extract_json(result.output)


def _assert_hf_config_structure(data: dict) -> None:
    """Assert the standard structure for HF model config output."""
    assert "loader" in data
    assert "export" in data
    assert "optim" in data

    # Loader must have task
    loader = data["loader"]
    assert "task" in loader
    assert loader["task"] is not None

    # Export must have opset_version and io specs
    export = data["export"]
    assert "opset_version" in export


def _assert_onnx_config_structure(data: dict) -> None:
    """Assert the structure for ONNX input config output."""
    assert data.get("export") is None  # Marks ONNX build path
    assert "optim" in data


# ===========================================================================
# BERT
# ===========================================================================

class TestConfigBert:
    """Config generation for bert-base-uncased."""

    MODEL = "bert-base-uncased"

    @pytest.mark.parametrize(
        "task",
        [
            "fill-mask",
            "text-classification",
            "token-classification",
        ],
        ids=["fill-mask", "text-cls", "token-cls"],
    )
    def test_with_explicit_task(self, task: str):
        """Config should generate valid output for known BERT tasks."""
        data = _run_config("-m", self.MODEL, "-t", task)
        _assert_hf_config_structure(data)
        assert data["loader"]["task"] == task

    def test_auto_detect(self):
        """Without --task the pipeline should auto-detect a task."""
        data = _run_config("-m", self.MODEL)
        _assert_hf_config_structure(data)
        assert data["loader"]["task"] is not None

    def test_device_cpu_precision_fp32(self):
        """Explicit device=cpu + precision=fp32 should work."""
        data = _run_config("-m", self.MODEL, "-t", "fill-mask", "-d", "cpu", "-p", "fp32")
        _assert_hf_config_structure(data)
        # With fp32 there should be no quantization
        assert data.get("quant") is None

    def test_output_to_file(self, tmp_path: Path):
        """Config output should be writable to a file via -o."""
        outfile = tmp_path / "config.json"
        runner = CliRunner()
        args = ["-m", self.MODEL, "-t", "fill-mask", "-o", str(outfile)]
        result = runner.invoke(config, args, catch_exceptions=False)
        assert result.exit_code == 0, f"config failed: {result.output}"
        assert outfile.exists()
        data = json.loads(outfile.read_text())
        _assert_hf_config_structure(data)

    def test_scenario_c_model_type_only(self):
        """--model-type bert without -m should use default HF config."""
        data = _run_config("--model-type", "bert")
        _assert_hf_config_structure(data)
        assert data["loader"]["task"] is not None


# ===========================================================================
# Vision models
# ===========================================================================

class TestConfigVision:
    """Config generation for vision models."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "microsoft/resnet-50",
            "facebook/convnext-tiny-224",
            "google/vit-base-patch16-224",
        ],
        ids=["resnet", "convnext", "vit"],
    )
    def test_auto_detect(self, model_id: str):
        """Vision models should auto-detect image-classification."""
        data = _run_config("-m", model_id)
        _assert_hf_config_structure(data)
        assert data["loader"]["task"] == "image-classification"


# ===========================================================================
# CLIP
# ===========================================================================

class TestConfigCLIP:
    """Config generation for CLIP."""

    MODEL = "openai/clip-vit-base-patch32"

    def test_feature_extraction(self):
        data = _run_config("-m", self.MODEL, "-t", "feature-extraction")
        _assert_hf_config_structure(data)
        assert data["loader"]["task"] == "feature-extraction"

    def test_zero_shot_image_classification(self):
        data = _run_config("-m", self.MODEL, "-t", "zero-shot-image-classification")
        _assert_hf_config_structure(data)
        assert data["loader"]["task"] == "zero-shot-image-classification"


# ===========================================================================
# DETR
# ===========================================================================

class TestConfigDETR:
    """Config generation for DETR."""

    MODEL = "facebook/detr-resnet-50"

    def test_auto_detect(self):
        data = _run_config("-m", self.MODEL)
        _assert_hf_config_structure(data)
        assert data["loader"]["task"] == "object-detection"


# ===========================================================================
# ONNX input
# ===========================================================================

class TestConfigONNX:
    """Config generation for pre-exported ONNX files."""

    def test_onnx_model_path(self, onnx_model_path: Path):
        """Passing a .onnx file should produce export=None config."""
        data = _run_config("-m", str(onnx_model_path))
        _assert_onnx_config_structure(data)
