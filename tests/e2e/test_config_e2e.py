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

    def test_onnx_with_no_compile(self, onnx_model_path: Path):
        """--no-compile on the ONNX path should yield compile=None."""
        data = _run_config("-m", str(onnx_model_path), "--no-compile")
        _assert_onnx_config_structure(data)
        assert data.get("compile") is None

    def test_onnx_with_no_quant(self, onnx_model_path: Path):
        """--no-quant on the ONNX path should yield quant=None."""
        data = _run_config("-m", str(onnx_model_path), "--no-quant")
        _assert_onnx_config_structure(data)
        assert data.get("quant") is None

    def test_onnx_output_to_file(self, onnx_model_path: Path, tmp_path: Path):
        """ONNX-path config should serialize to disk via -o."""
        outfile = tmp_path / "onnx_config.json"
        runner = CliRunner()
        result = runner.invoke(
            config,
            ["-m", str(onnx_model_path), "-o", str(outfile)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"config failed: {result.output}"
        assert outfile.exists()
        _assert_onnx_config_structure(json.loads(outfile.read_text()))


# ===========================================================================
# BAD PATH — argument validation and CLI-level error handling
#
# These exercises do not need to reach the network: invalid inputs must be
# rejected by Click or by the config command's own validation, producing a
# non-zero exit code and the documented user-facing error (never a raw
# stack trace).
# ===========================================================================


def _invoke_config(*args: str) -> object:
    """Invoke the config command; do NOT raise on non-zero exit."""
    runner = CliRunner()
    return runner.invoke(config, list(args))


class TestConfigBadPath:
    """Bad-path coverage: invalid args, missing inputs, mutually exclusive flags."""

    def test_no_args_is_error(self) -> None:
        """Invoking with no args must fail with a usage error, not a traceback."""
        result = _invoke_config()
        assert result.exit_code != 0
        # No raw Python traceback should leak to the user.
        assert "Traceback (most recent call last)" not in result.output

    def test_missing_entry_point_message(self) -> None:
        """The error message should point the user at the required flags."""
        result = _invoke_config()
        assert result.exit_code != 0
        combined = (result.output or "") + (str(result.exception) if result.exception else "")
        # Either Click's missing-option message OR our UsageError hint.
        assert (
            "--model" in combined
            or "--model-type" in combined
            or "--model-class" in combined
        )

    @pytest.mark.parametrize("bad_device", ["tpu", "fpga", "xpu", "DSP"])
    def test_invalid_device_rejected(self, bad_device: str) -> None:
        """Click's Choice validation must reject unknown --device values."""
        result = _invoke_config("-m", "test", "--device", bad_device)
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    @pytest.mark.parametrize("bad_precision", ["bf16", "fp64", "int4", "w3a5"])
    def test_invalid_precision_rejected(self, bad_precision: str) -> None:
        """Unknown precision strings must produce a UsageError, not a traceback."""
        # --model-type bert avoids a network round-trip while still exercising
        # the precision validation path inside generate_hf_build_config.
        result = _invoke_config(
            "--model-type", "bert", "--task", "fill-mask",
            "--precision", bad_precision,
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    @pytest.mark.parametrize("bad_ep", ["tflite", "coreml", "not-a-real-ep"])
    def test_invalid_ep_rejected(self, bad_ep: str) -> None:
        """Unknown --ep values must produce a UsageError, not a traceback."""
        result = _invoke_config(
            "--model-type", "bert", "--task", "fill-mask",
            "--ep", bad_ep,
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_nonexistent_config_file_rejected(self, tmp_path: Path) -> None:
        """-c pointing at a missing file must be rejected by Click."""
        missing = tmp_path / "does_not_exist.json"
        result = _invoke_config("-m", "test", "-c", str(missing))
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_empty_config_file_rejected(self, tmp_path: Path) -> None:
        """An empty -c file must produce a UsageError."""
        empty = tmp_path / "empty.json"
        empty.write_text("")
        result = _invoke_config("-m", "test", "-c", str(empty))
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_invalid_json_config_file_rejected(self, tmp_path: Path) -> None:
        """Malformed JSON in -c must produce a UsageError."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        result = _invoke_config("-m", "test", "-c", str(bad))
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_non_object_json_config_file_rejected(self, tmp_path: Path) -> None:
        """A JSON array in -c must be rejected (must be an object)."""
        arr = tmp_path / "array.json"
        arr.write_text("[1, 2, 3]")
        result = _invoke_config("-m", "test", "-c", str(arr))
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_empty_shape_config_rejected(self, tmp_path: Path) -> None:
        """An empty --shape-config file must produce a UsageError."""
        empty = tmp_path / "shapes.json"
        empty.write_text("")
        result = _invoke_config("-m", "test", "--shape-config", str(empty))
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_invalid_json_shape_config_rejected(self, tmp_path: Path) -> None:
        """Malformed --shape-config JSON must produce a UsageError."""
        bad = tmp_path / "shapes.json"
        bad.write_text("{height: 224")  # missing quotes
        result = _invoke_config("-m", "test", "--shape-config", str(bad))
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_non_object_shape_config_rejected(self, tmp_path: Path) -> None:
        """A JSON list in --shape-config must be rejected (must be an object)."""
        bad = tmp_path / "shapes.json"
        bad.write_text("[224, 224]")
        result = _invoke_config("-m", "test", "--shape-config", str(bad))
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_module_with_onnx_file_rejected(self, onnx_model_path: Path) -> None:
        """--module is mutually exclusive with .onnx input."""
        result = _invoke_config(
            "-m", str(onnx_model_path),
            "--module", "ResNetConvLayer",
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output
        combined = (result.output or "") + (str(result.exception) if result.exception else "")
        assert "module" in combined.lower()


# ===========================================================================
# FLAG VARIATIONS — every behavior-bearing flag, present vs absent
#
# Uses bert-base-uncased + fill-mask as a stable, well-supported baseline
# so the exercise is about flag plumbing, not model coverage.
# ===========================================================================


class TestConfigFlagVariations:
    """Each enum value / behavior-bearing flag is touched at least once."""

    MODEL = "bert-base-uncased"
    TASK = "fill-mask"

    # --- --device ---------------------------------------------------------
    @pytest.mark.parametrize("device", ["auto", "cpu", "gpu", "npu"])
    def test_every_device_choice(self, device: str) -> None:
        """Every --device choice should produce a valid config."""
        # NPU + auto precision = w8a16; auto-everything = no-op. Pair the
        # device with a precision known to be compatible across devices.
        precision = "fp32" if device == "cpu" else "auto"
        data = _run_config(
            "-m", self.MODEL, "-t", self.TASK,
            "-d", device, "-p", precision,
        )
        _assert_hf_config_structure(data)

    # --- --precision ------------------------------------------------------
    @pytest.mark.parametrize("precision", ["auto", "fp32", "fp16", "int8", "int16"])
    def test_every_named_precision(self, precision: str) -> None:
        """Every named --precision choice should produce a valid config."""
        # Pair each precision with a compatible device to bypass NPU's
        # narrow precision matrix (which would reject fp32/int8 by design).
        device = "cpu" if precision in ("fp32", "fp16") else "npu"
        data = _run_config(
            "-m", self.MODEL, "-t", self.TASK,
            "-p", precision, "-d", device,
        )
        _assert_hf_config_structure(data)

    @pytest.mark.parametrize("mixed", ["w8a8", "w8a16"])
    def test_mixed_precision(self, mixed: str) -> None:
        """Mixed precision w{x}a{y} should be accepted on NPU."""
        data = _run_config(
            "-m", self.MODEL, "-t", self.TASK,
            "-p", mixed, "-d", "npu",
        )
        _assert_hf_config_structure(data)

    # --- --ep -------------------------------------------------------------
    @pytest.mark.parametrize(
        "ep",
        ["qnn", "dml", "openvino", "vitisai", "migraphx", "nv_tensorrt_rtx", "cpu"],
    )
    def test_every_ep_choice(self, ep: str) -> None:
        """Every documented --ep alias should be accepted."""
        # Use auto precision so device-specific constraints don't bite.
        data = _run_config(
            "-m", self.MODEL, "-t", self.TASK,
            "--ep", ep, "-p", "auto",
        )
        _assert_hf_config_structure(data)

    # --- --no-quant / --no-compile / --compile ---------------------------
    def test_no_quant_present(self) -> None:
        """--no-quant must zero out the quant section."""
        data = _run_config("-m", self.MODEL, "-t", self.TASK, "--no-quant")
        assert data.get("quant") is None

    def test_no_quant_absent(self) -> None:
        """Without --no-quant a quantized device should keep quant settings."""
        data = _run_config(
            "-m", self.MODEL, "-t", self.TASK,
            "-d", "npu", "-p", "int8",
        )
        assert data.get("quant") is not None

    def test_no_compile_default(self) -> None:
        """Default behavior excludes compile (--no-compile is the default)."""
        data = _run_config("-m", self.MODEL, "-t", self.TASK)
        assert data.get("compile") is None

    def test_compile_enabled(self) -> None:
        """--compile (negated default) should produce a compile section."""
        data = _run_config(
            "-m", self.MODEL, "-t", self.TASK,
            "--compile", "-d", "npu",
        )
        # When --compile is requested the section must not be null.
        assert data.get("compile") is not None

    # --- --shape-config ---------------------------------------------------
    def test_shape_config_present(self, tmp_path: Path) -> None:
        """--shape-config should be accepted and applied."""
        shapes = tmp_path / "shapes.json"
        shapes.write_text(json.dumps({"sequence_length": 32}))
        data = _run_config(
            "-m", self.MODEL, "-t", self.TASK,
            "--shape-config", str(shapes),
        )
        _assert_hf_config_structure(data)

    # --- --library --------------------------------------------------------
    def test_library_default(self) -> None:
        """Default --library transformers should work without explicit flag."""
        data = _run_config("-m", self.MODEL, "-t", self.TASK)
        _assert_hf_config_structure(data)

    def test_library_explicit(self) -> None:
        """Passing --library transformers explicitly should be accepted."""
        data = _run_config(
            "-m", self.MODEL, "-t", self.TASK,
            "--library", "transformers",
        )
        _assert_hf_config_structure(data)

    # --- --verbose --------------------------------------------------------
    def test_verbose_flag(self) -> None:
        """--verbose / -v should not affect JSON output but must not crash."""
        data = _run_config("-m", self.MODEL, "-t", self.TASK, "-v")
        _assert_hf_config_structure(data)

    # --- --model-type / --model-class ------------------------------------
    def test_model_type_only(self) -> None:
        """--model-type alone (no -m) should auto-pick a supported task."""
        data = _run_config("--model-type", "bert")
        _assert_hf_config_structure(data)

    def test_model_type_with_task(self) -> None:
        """--model-type + --task should be honored."""
        data = _run_config("--model-type", "bert", "--task", "fill-mask")
        _assert_hf_config_structure(data)
        assert data["loader"]["task"] == "fill-mask"

    # --- -c / --config override ------------------------------------------
    def test_config_file_override(self, tmp_path: Path) -> None:
        """-c override file should be loaded and merged."""
        override = tmp_path / "override.json"
        override.write_text(json.dumps({"export": {"opset_version": 18}}))
        data = _run_config(
            "-m", self.MODEL, "-t", self.TASK,
            "-c", str(override),
        )
        _assert_hf_config_structure(data)
        assert data["export"]["opset_version"] == 18

    # --- --trust-remote-code ---------------------------------------------
    def test_trust_remote_code_flag(self) -> None:
        """--trust-remote-code should be accepted on a normal HF model."""
        data = _run_config(
            "-m", self.MODEL, "-t", self.TASK,
            "--trust-remote-code",
        )
        _assert_hf_config_structure(data)

    # --- --module ---------------------------------------------------------
    def test_module_flag_returns_list(self) -> None:
        """--module mode should emit a JSON list of per-submodule configs."""
        data = _run_config(
            "-m", "microsoft/resnet-50",
            "--module", "ResNetConvLayer",
        )
        assert isinstance(data, list), f"Expected JSON list for --module, got {type(data)}"
        assert len(data) > 0
        for cfg in data:
            assert "loader" in cfg
            assert "export" in cfg
