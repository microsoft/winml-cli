# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for export command using export_onnx() as single implementation path.

This test module validates that export command:
1. Uses export_onnx() instead of HTPExporter directly
2. Properly handles ExportConfig construction
3. Correctly passes through configuration options
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_export_onnx():
    """Mock export_onnx function to avoid actual model loading."""
    with patch("winml.modelkit.export.pytorch.export_pytorch") as mock:
        mock.return_value = Path("test_output.onnx")
        yield mock


@pytest.fixture
def mock_load_hf_model():
    """Mock load_hf_model to return a minimal mock model."""
    with patch("winml.modelkit.loader.load_hf_model") as mock:
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock.return_value = (mock_model, None, "image-classification")
        yield mock


class TestExportCLIInterface:
    """Test export command CLI interface."""

    def test_export_help_shows_all_options(self, runner: CliRunner) -> None:
        """Test export --help shows all expected options."""
        from winml.modelkit.commands.export import export

        result = runner.invoke(export, ["--help"])
        assert result.exit_code == 0

        # Required options
        assert "--model" in result.output
        assert "-m" in result.output
        assert "--output" in result.output
        assert "-o" in result.output

        # Optional flags
        assert "--with-report" in result.output
        assert "--clean-onnx" in result.output
        assert "--dynamo" in result.output
        assert "--torch-module" in result.output
        assert "--input-specs" in result.output
        assert "--export-config" in result.output

    def test_export_requires_model(self, runner: CliRunner) -> None:
        """Test export fails without --model argument."""
        from winml.modelkit.commands.export import export

        result = runner.invoke(export, ["--output", "test.onnx"])
        assert result.exit_code != 0
        assert "model" in result.output.lower() or "required" in result.output.lower()

    def test_export_requires_output(self, runner: CliRunner) -> None:
        """Test export fails without --output argument."""
        from winml.modelkit.commands.export import export

        result = runner.invoke(export, ["--model", "test-model"])
        assert result.exit_code != 0
        assert "output" in result.output.lower() or "required" in result.output.lower()


class TestExportUsesExportOnnx:
    """Test export uses export_onnx() as single implementation path."""

    def test_export_calls_export_onnx(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test export delegates to export_onnx() correctly."""
        from winml.modelkit.commands.export import export

        output_path = tmp_path / "model.onnx"
        runner.invoke(
            export,
            [
                "--model",
                "test-model",
                "--output",
                str(output_path),
            ],
            obj={"debug": False},
        )

        # Should call export_onnx
        assert mock_export_onnx.called, "export_onnx should be called"

        # Verify call arguments
        call_kwargs = mock_export_onnx.call_args.kwargs
        assert "model" in call_kwargs
        assert "output_path" in call_kwargs
        assert "export_config" in call_kwargs
        assert "model_id" in call_kwargs
        assert call_kwargs["model_id"] == "test-model"

    def test_export_passes_verbose_flag(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test -v global flag propagates verbose to export_onnx."""
        from winml.modelkit.cli import main

        output_path = tmp_path / "model.onnx"
        runner.invoke(
            main,
            ["-v", "export", "--model", "test-model", "--output", str(output_path)],
        )

        call_kwargs = mock_export_onnx.call_args.kwargs
        assert call_kwargs["verbose"]

    def test_export_passes_with_report_flag(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test --with-report flag is passed to export_onnx."""
        from winml.modelkit.commands.export import export

        output_path = tmp_path / "model.onnx"
        runner.invoke(
            export,
            ["--model", "test-model", "--output", str(output_path), "--with-report"],
            obj={"debug": False},
        )

        call_kwargs = mock_export_onnx.call_args.kwargs
        assert call_kwargs["enable_reporting"] is True

    def test_export_passes_detected_task(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test detected task is passed to export_onnx."""
        from winml.modelkit.commands.export import export

        # Mock returns "image-classification" as detected task
        mock_load_hf_model.return_value = (MagicMock(), None, "text-classification")

        output_path = tmp_path / "model.onnx"
        runner.invoke(
            export,
            ["--model", "test-model", "--output", str(output_path)],
            obj={"debug": False},
        )

        call_kwargs = mock_export_onnx.call_args.kwargs
        assert call_kwargs["task"] == "text-classification"


class TestExportExportConfig:
    """Test ExportConfig construction in export command."""

    def test_export_creates_export_config(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test export creates ExportConfig dataclass."""
        from winml.modelkit.commands.export import export
        from winml.modelkit.export import WinMLExportConfig

        output_path = tmp_path / "model.onnx"
        runner.invoke(
            export,
            ["--model", "test-model", "--output", str(output_path)],
            obj={"debug": False},
        )

        call_kwargs = mock_export_onnx.call_args.kwargs
        assert "export_config" in call_kwargs
        assert isinstance(call_kwargs["export_config"], WinMLExportConfig)

    def test_export_clean_onnx_disables_hierarchy_tags(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test --clean-onnx sets enable_hierarchy_tags=False."""
        from winml.modelkit.commands.export import export

        output_path = tmp_path / "model.onnx"
        runner.invoke(
            export,
            ["--model", "test-model", "--output", str(output_path), "--clean-onnx"],
            obj={"debug": False},
        )

        call_kwargs = mock_export_onnx.call_args.kwargs
        config = call_kwargs["export_config"]
        assert config.enable_hierarchy_tags is False

    def test_export_default_enables_hierarchy_tags(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test default behavior enables hierarchy tags."""
        from winml.modelkit.commands.export import export

        output_path = tmp_path / "model.onnx"
        runner.invoke(
            export,
            ["--model", "test-model", "--output", str(output_path)],
            obj={"debug": False},
        )

        call_kwargs = mock_export_onnx.call_args.kwargs
        config = call_kwargs["export_config"]
        assert config.enable_hierarchy_tags is True


class TestExportConfigFiles:
    """Test export handling of JSON configuration files."""

    def test_export_loads_export_config_json(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test --export-config loads JSON configuration."""
        from winml.modelkit.commands.export import export

        # Create config file
        config_file = tmp_path / "config.json"
        config_data = {"opset_version": 15, "do_constant_folding": False}
        config_file.write_text(json.dumps(config_data))

        output_path = tmp_path / "model.onnx"
        runner.invoke(
            export,
            [
                "--model",
                "test-model",
                "--output",
                str(output_path),
                "--export-config",
                str(config_file),
            ],
            obj={"debug": False},
        )

        call_kwargs = mock_export_onnx.call_args.kwargs
        config = call_kwargs["export_config"]
        assert config.opset_version == 15
        assert config.do_constant_folding is False

    def test_export_loads_input_specs_json(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test --input-specs loads JSON input specifications."""
        from winml.modelkit.commands.export import export

        # Create input specs file
        specs_file = tmp_path / "inputs.json"
        specs_data = {
            "pixel_values": {"dtype": "float32", "shape": [1, 3, 224, 224]},
            "input_ids": {"dtype": "int64", "shape": [1, 128]},
        }
        specs_file.write_text(json.dumps(specs_data))

        output_path = tmp_path / "model.onnx"
        runner.invoke(
            export,
            [
                "--model",
                "test-model",
                "--output",
                str(output_path),
                "--input-specs",
                str(specs_file),
            ],
            obj={"debug": False},
        )

        call_kwargs = mock_export_onnx.call_args.kwargs
        config = call_kwargs["export_config"]
        assert config.input_tensors is not None
        assert len(config.input_tensors) == 2

    def test_export_invalid_export_config_raises(
        self,
        runner: CliRunner,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test invalid --export-config raises ClickException."""
        from winml.modelkit.commands.export import export

        # Create invalid JSON file
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json }")

        output_path = tmp_path / "model.onnx"
        result = runner.invoke(
            export,
            [
                "--model",
                "test-model",
                "--output",
                str(output_path),
                "--export-config",
                str(config_file),
            ],
            obj={"debug": False},
        )

        assert result.exit_code != 0
        assert "Failed to load export config" in result.output


class TestExportWarnings:
    """Test export warning messages for unsupported options."""

    def test_export_warns_on_torch_module(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test --torch-module shows warning (not yet supported)."""
        from winml.modelkit.commands.export import export

        output_path = tmp_path / "model.onnx"
        result = runner.invoke(
            export,
            [
                "--model",
                "test-model",
                "--output",
                str(output_path),
                "--torch-module",
                "LayerNorm,Embedding",
            ],
            obj={"debug": False},
        )

        assert "not yet supported" in result.output or "Warning" in result.output

    def test_export_warns_on_dynamo(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test --dynamo shows warning (not yet supported)."""
        from winml.modelkit.commands.export import export

        output_path = tmp_path / "model.onnx"
        result = runner.invoke(
            export,
            [
                "--model",
                "test-model",
                "--output",
                str(output_path),
                "--dynamo",
            ],
            obj={"debug": False},
        )

        assert "not yet supported" in result.output or "Warning" in result.output


class TestExportErrorHandling:
    """Test export error handling."""

    def test_export_handles_model_load_error(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test export handles model loading errors gracefully."""
        from winml.modelkit.commands.export import export

        with patch("winml.modelkit.loader.load_hf_model") as mock:
            mock.side_effect = Exception("Model not found")

            output_path = tmp_path / "model.onnx"
            result = runner.invoke(
                export,
                ["--model", "nonexistent-model", "--output", str(output_path)],
                obj={"debug": False},
            )

            assert result.exit_code != 0
            assert "Export failed" in result.output or "error" in result.output.lower()

    def test_export_handles_export_onnx_error(
        self,
        runner: CliRunner,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test export handles export_onnx() errors gracefully."""
        from winml.modelkit.commands.export import export

        with patch("winml.modelkit.export.pytorch.export_pytorch") as mock:
            mock.side_effect = RuntimeError("ONNX export failed")

            output_path = tmp_path / "model.onnx"
            result = runner.invoke(
                export,
                ["--model", "test-model", "--output", str(output_path)],
                obj={"debug": False},
            )

            assert result.exit_code != 0
            assert "Export failed" in result.output


class TestExportAutoResolveInputTensors:
    """Test export auto-resolves input_tensors via resolve_export_config."""

    def test_export_uses_resolve_export_config(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test export uses resolve_export_config for input_tensors when no --input-specs."""
        from winml.modelkit.commands.export import export
        from winml.modelkit.export import InputTensorSpec, WinMLExportConfig
        from winml.modelkit.loader import WinMLLoaderConfig

        # Create mock return values for resolve_export_config
        mock_export_cfg = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(name="pixel_values", dtype="float32", shape=(1, 3, 224, 224)),
            ],
        )
        mock_loader_cfg = WinMLLoaderConfig(
            task="image-classification",
            model_type="resnet",
        )

        with (
            patch("winml.modelkit.loader.load_hf_model") as mock_load,
            patch(
                "winml.modelkit.export.config.resolve_export_config",
                return_value=(mock_export_cfg, mock_loader_cfg),
            ),
        ):
            mock_model = MagicMock()
            mock_load.return_value = (mock_model, None, "image-classification")

            output_path = tmp_path / "model.onnx"
            runner.invoke(
                export,
                ["--model", "test-model", "--output", str(output_path)],
                obj={"debug": False},
            )

            # Verify export_onnx was called with input_tensors in config
            call_kwargs = mock_export_onnx.call_args.kwargs
            config = call_kwargs["export_config"]
            assert config.input_tensors is not None


class TestExportOutputDirectory:
    """Test export output directory handling."""

    def test_export_creates_output_directory(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test export creates output directory if it doesn't exist."""
        from winml.modelkit.commands.export import export

        # Use nested path that doesn't exist
        output_path = tmp_path / "nested" / "dir" / "model.onnx"
        assert not output_path.parent.exists()

        runner.invoke(
            export,
            ["--model", "test-model", "--output", str(output_path)],
            obj={"debug": False},
        )

        # Directory should be created
        assert output_path.parent.exists()


class TestExportDebugMode:
    """Test export debug mode inheritance."""

    def test_export_inherits_debug_mode(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test export inherits debug/verbose from parent context.

        In the new CLI spec, main sets ctx.obj['verbose'] = 2 when --debug
        is used.  Subcommands read ctx.obj['verbose'] directly.
        """
        from winml.modelkit.commands.export import export

        output_path = tmp_path / "model.onnx"

        # Simulate what main does when --debug is passed
        runner.invoke(
            export,
            ["--model", "test-model", "--output", str(output_path)],
            obj={"debug": True, "verbose": 2},
        )

        # verbose should be truthy (int >= 1) due to debug mode
        call_kwargs = mock_export_onnx.call_args.kwargs
        assert call_kwargs["verbose"]
