# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for config CLI command -- mock-based, no network, no actual config generation.

Tests the CLI wrapper around generate_hf_build_config() / generate_onnx_build_config() APIs.
NO actual model loading or network calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch


if TYPE_CHECKING:
    from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock resolve_device to avoid hardware detection in all config CLI tests.

    The config command may call resolve_device() for device/precision resolution.
    We mock it at the source module since it's a lazy import.
    """
    with patch(
        "winml.modelkit.sysinfo.resolve_device",
        return_value=("npu", ["npu", "gpu", "cpu"]),
    ):
        yield


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_generate_config():
    """Mock generate_hf_build_config to avoid actual config generation.

    Returns a MagicMock whose to_dict() yields a valid JSON-serializable dict.
    The mock target is the lazy import inside modelkit.commands.config.
    """
    mock_cfg = MagicMock()
    mock_cfg.loader.task = "image-classification"
    mock_cfg.to_dict.return_value = {
        "loader": {
            "task": "image-classification",
            "model_class": "ResNetForImageClassification",
        },
        "export": {"opset_version": 17},
        "optim": {},
        "quant": None,
        "compile": None,
    }
    with patch(
        "winml.modelkit.config.generate_hf_build_config",
        return_value=mock_cfg,
    ) as mock:
        yield mock


# =============================================================================
# CLI INTERFACE TESTS
# =============================================================================


class TestConfigCliInterface:
    """Test CLI flag parsing and help text."""

    def test_help_shows_all_options(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["--help"])
        assert result.exit_code == 0

        # All documented options must appear in help text
        expected_options = [
            "--model", "-m",
            "--task", "-t",
            "--model-class",
            "--model-type",
            "--module",
            "--config", "-c",
            "--shape-config",
            "--device", "-d",
            "--precision", "-p",
            "--output", "-o",
            "--library",
            "--verbose", "-v",
            "--no-quant",
            "--no-compile",
            "--trust-remote-code",
        ]
        for opt in expected_options:
            assert opt in result.output, f"Expected '{opt}' in help output"

    def test_no_entry_point_error(self, runner: CliRunner) -> None:
        """Invoking with no args should fail (need -m/--model-type/--model-class)."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, [])
        assert result.exit_code != 0

    def test_invalid_device_rejected(self, runner: CliRunner) -> None:
        """--device tpu should be rejected by click.Choice validation."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--device", "tpu"])
        assert result.exit_code != 0

    def test_invalid_precision_rejected(self, runner: CliRunner) -> None:
        """--precision bf16 should be rejected by click.Choice validation."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--precision", "bf16"])
        assert result.exit_code != 0

    @pytest.mark.parametrize("device", ["auto", "npu", "gpu", "cpu"])
    def test_valid_device_choices(
        self,
        runner: CliRunner,
        device: str,
        mock_generate_config: MagicMock,
    ) -> None:
        """All valid device choices should be accepted without error."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--device", device])
        assert result.exit_code == 0, (
            f"Device '{device}' should be accepted, got exit_code={result.exit_code}: "
            f"{result.output}"
        )

    @pytest.mark.parametrize("precision", ["auto", "fp32", "fp16", "int8", "int16"])
    def test_valid_precision_choices(
        self,
        runner: CliRunner,
        precision: str,
        mock_generate_config: MagicMock,
    ) -> None:
        """All valid precision choices should be accepted without error."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--precision", precision])
        assert result.exit_code == 0, (
            f"Precision '{precision}' should be accepted, "
            f"got exit_code={result.exit_code}: {result.output}"
        )

    def test_output_to_file(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_generate_config: MagicMock,
    ) -> None:
        """Outputting to a file via -o should not crash."""
        from winml.modelkit.commands.config import config

        output_file = tmp_path / "out.json"
        result = runner.invoke(config, ["-m", "test", "-o", str(output_file)])
        assert result.exit_code == 0, (
            f"Output to file should succeed: {result.output}"
        )

    def test_model_type_without_model(
        self,
        runner: CliRunner,
        mock_generate_config: MagicMock,
    ) -> None:
        """--model-type bert --task fill-mask should be a valid entry point (no -m needed)."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(
            config, ["--model-type", "bert", "--task", "fill-mask"]
        )
        assert result.exit_code == 0, (
            f"model-type without model should succeed: {result.output}"
        )

    def test_config_file_override(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_generate_config: MagicMock,
    ) -> None:
        """A config override file via -c should be accepted."""
        from winml.modelkit.commands.config import config

        override_file = tmp_path / "override.json"
        override_file.write_text('{"loader": {"task": "text-classification"}}')

        result = runner.invoke(config, ["-m", "test", "-c", str(override_file)])
        assert result.exit_code == 0, (
            f"Config file override should succeed: {result.output}"
        )

    def test_shape_config_file(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_generate_config: MagicMock,
    ) -> None:
        """A shape config file via --shape-config should be accepted."""
        from winml.modelkit.commands.config import config

        shapes_file = tmp_path / "shapes.json"
        shapes_file.write_text('{"height": 224, "width": 224}')

        result = runner.invoke(config, ["-m", "test", "--shape-config", str(shapes_file)])
        assert result.exit_code == 0, (
            f"Shape config file should succeed: {result.output}"
        )


    def test_no_quant_sets_quant_none(
        self,
        runner: CliRunner,
        mock_generate_config: MagicMock,
    ) -> None:
        """--no-quant should set quant=None on the generated config."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--no-quant"])
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert mock_generate_config.return_value.quant is None

    def test_no_compile_sets_compile_none(
        self,
        runner: CliRunner,
        mock_generate_config: MagicMock,
    ) -> None:
        """--no-compile should set compile=None on the generated config."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--no-compile"])
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert mock_generate_config.return_value.compile is None

    def test_trust_remote_code_passed_to_api(
        self,
        runner: CliRunner,
        mock_generate_config: MagicMock,
    ) -> None:
        """--trust-remote-code should be passed to generate_hf_build_config."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--trust-remote-code"])
        assert result.exit_code == 0, f"Failed: {result.output}"
        mock_generate_config.assert_called_once()
        call_kwargs = mock_generate_config.call_args.kwargs
        assert call_kwargs.get("trust_remote_code") is True


# =============================================================================
# ONNX PATH OVERRIDE TESTS
# =============================================================================


def _extract_json(output: str) -> dict:
    """Extract JSON object from mixed CLI output (Rich stderr + JSON stdout).

    CliRunner in Click 8.x mixes stderr and stdout. The JSON block starts
    at the first '{' and ends at the last '}'.
    """
    import json

    start = output.index("{")
    end = output.rindex("}") + 1
    return json.loads(output[start:end])


class TestConfigOnnxOverrides:
    """Test --no-quant and --no-compile work on the ONNX path."""

    def test_onnx_no_quant(self, runner: CliRunner, tmp_path: Path) -> None:
        """--no-quant should set quant=None even for ONNX configs."""
        from winml.modelkit.commands.config import config

        # Create a fake .onnx file so _is_onnx_file returns True
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            result = runner.invoke(config, ["-m", str(onnx_file), "--no-quant"])
        assert result.exit_code == 0, f"Failed: {result.output}"

        data = _extract_json(result.output)
        assert data.get("quant") is None

    def test_onnx_no_compile(self, runner: CliRunner, tmp_path: Path) -> None:
        """--no-compile should set compile=None even for ONNX configs."""
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            result = runner.invoke(config, ["-m", str(onnx_file), "--no-compile"])
        assert result.exit_code == 0, f"Failed: {result.output}"

        data = _extract_json(result.output)
        assert data.get("compile") is None


# =============================================================================
# ONNX QDQ AUTO-DETECTION TESTS
# =============================================================================


class TestConfigOnnxQdqDetection:
    """Test config command auto-detects QDQ ONNX and sets quant=None."""

    def test_qdq_onnx_sets_quant_none(self, runner: CliRunner, tmp_path: Path) -> None:
        """Config for a QDQ ONNX file should have quant=null in output."""
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "quantized.onnx"
        onnx_file.write_bytes(b"fake-qdq-onnx")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=True),
        ):
            result = runner.invoke(config, ["-m", str(onnx_file)])

        assert result.exit_code == 0, f"Failed: {result.output}"
        data = _extract_json(result.output)
        assert data.get("quant") is None, (
            f"Expected quant=null for QDQ model, got: {data.get('quant')}"
        )

    def test_qdq_onnx_output_confirms_no_quant(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Config for a QDQ ONNX should produce export=null and quant=null."""
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "quantized.onnx"
        onnx_file.write_bytes(b"fake-qdq-onnx")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=True),
        ):
            result = runner.invoke(config, ["-m", str(onnx_file)])

        assert result.exit_code == 0, f"Failed: {result.output}"
        data = _extract_json(result.output)
        assert data.get("export") is None, "QDQ ONNX build should have export=null"
        assert data.get("quant") is None, "QDQ ONNX build should have quant=null"

    def test_qdq_overrides_device_precision(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """QDQ detection should keep quant=null even with -d npu -p int8."""
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "quantized.onnx"
        onnx_file.write_bytes(b"fake-qdq-onnx")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=True),
        ):
            result = runner.invoke(
                config, ["-m", str(onnx_file), "-d", "npu", "-p", "int8"]
            )

        assert result.exit_code == 0, f"Failed: {result.output}"
        data = _extract_json(result.output)
        assert data.get("quant") is None, (
            "QDQ detection should take precedence over -d npu -p int8"
        )

    def test_non_qdq_onnx_has_default_quant(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Config for non-QDQ ONNX should have default quant settings (not null)."""
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "normal.onnx"
        onnx_file.write_bytes(b"fake-onnx")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            result = runner.invoke(config, ["-m", str(onnx_file)])

        assert result.exit_code == 0, f"Failed: {result.output}"
        data = _extract_json(result.output)
        # Default ONNX config should have quant as a dict (not null)
        assert data.get("quant") is not None, (
            f"Non-QDQ model should have default quant settings, got: {data.get('quant')}"
        )
