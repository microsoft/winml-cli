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
import shlex
from inspect import getdoc
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
    with patch("winml.modelkit.export.export_pytorch") as mock:
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
        assert "--verbose" in result.output
        assert "-v" in result.output
        assert "--with-report" in result.output
        assert "--no-with-report" in result.output
        assert "--hierarchy" in result.output
        assert "--no-hierarchy" in result.output
        assert "--dynamo" in result.output
        assert "--no-dynamo" in result.output
        assert "--torch-module" in result.output
        assert "--input-specs" in result.output
        assert "--export-config" in result.output

    def test_export_help_examples_run(self, runner: CliRunner, tmp_path: Path) -> None:
        """Every command example in export help should execute without crashing."""
        from winml.modelkit.commands.export import export
        from winml.modelkit.export import WinMLExportConfig
        from winml.modelkit.loader import WinMLLoaderConfig

        doc = getdoc(export) or ""
        examples = [
            line.strip() for line in doc.splitlines() if line.strip().startswith("winml export")
        ]
        assert examples, "Expected at least one winml export example in help docstring"

        specs_file = tmp_path / "inputs.json"
        specs_file.write_text(json.dumps({"input_ids": {"dtype": "int64", "shape": [1, 8]}}))
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"opset_version": 17}))

        mock_export_cfg = WinMLExportConfig()
        mock_loader_cfg = WinMLLoaderConfig(task="feature-extraction", model_type="bert")

        with (
            patch("winml.modelkit.loader.load_hf_model") as mock_load,
            patch("winml.modelkit.export.export_pytorch", return_value=tmp_path / "ok.onnx"),
            patch(
                "winml.modelkit.export.resolve_export_config",
                return_value=(mock_export_cfg, mock_loader_cfg),
            ),
        ):
            mock_model = MagicMock()
            mock_load.return_value = (mock_model, None, "feature-extraction")
            saw_input_specs_example = False
            saw_export_config_example = False

            for example in examples:
                try:
                    tokens = shlex.split(example)
                except ValueError as e:
                    pytest.fail(f"Unable to parse example command: {example!r} ({e})")
                if len(tokens) < 2 or tokens[0] != "winml" or tokens[1] != "export":
                    pytest.fail(f"Malformed export example command: {example!r}")
                args = tokens[2:]  # drop "winml export"
                args = [str(specs_file) if arg == "inputs.json" else arg for arg in args]
                args = [str(config_file) if arg == "config.json" else arg for arg in args]
                args = [str(tmp_path / arg) if arg.endswith(".onnx") else arg for arg in args]
                saw_input_specs_example |= str(specs_file) in args
                saw_export_config_example |= str(config_file) in args

                result = runner.invoke(export, args, obj={"debug": False})
                assert result.exit_code == 0, f"Example failed: {example}\n{result.output}"

        assert saw_input_specs_example
        assert saw_export_config_example

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
        """Test --verbose flag is passed to export_onnx."""
        from winml.modelkit.commands.export import export

        output_path = tmp_path / "model.onnx"
        runner.invoke(
            export,
            ["--model", "test-model", "--output", str(output_path), "--verbose"],
            obj={"debug": False},
        )

        call_kwargs = mock_export_onnx.call_args.kwargs
        assert call_kwargs["verbose"] is True

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
        """Test --clean-onnx sets enable_hierarchy_tags=False and emits deprecation warning."""
        from winml.modelkit.commands.export import export

        output_path = tmp_path / "model.onnx"
        result = runner.invoke(
            export,
            ["--model", "test-model", "--output", str(output_path), "--clean-onnx"],
            obj={"debug": False},
        )

        call_kwargs = mock_export_onnx.call_args.kwargs
        config = call_kwargs["export_config"]
        assert config.enable_hierarchy_tags is False
        assert "--clean-onnx is deprecated" in result.output

    def test_export_no_hierarchy_disables_hierarchy_tags(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test --no-hierarchy sets enable_hierarchy_tags=False."""
        from winml.modelkit.commands.export import export

        output_path = tmp_path / "model.onnx"
        runner.invoke(
            export,
            ["--model", "test-model", "--output", str(output_path), "--no-hierarchy"],
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

        with patch("winml.modelkit.export.export_pytorch") as mock:
            mock.side_effect = RuntimeError("ONNX export failed")

            output_path = tmp_path / "model.onnx"
            result = runner.invoke(
                export,
                ["--model", "test-model", "--output", str(output_path)],
                obj={"debug": False},
            )

            assert result.exit_code != 0
            assert "Export failed" in result.output

    def test_export_logs_error_without_traceback_when_not_debug(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Non-debug mode should log a plain error instead of exception traceback."""
        from winml.modelkit.commands.export import export

        with (
            patch("winml.modelkit.loader.load_hf_model", side_effect=ValueError("boom")),
            patch("winml.modelkit.commands.export.logger") as mock_logger,
        ):
            output_path = tmp_path / "model.onnx"
            result = runner.invoke(
                export,
                ["--model", "test-model", "--output", str(output_path)],
                obj={"debug": False},
            )

        assert result.exit_code != 0
        mock_logger.error.assert_called_once()
        mock_logger.exception.assert_not_called()

    def test_export_logs_traceback_when_debug_enabled(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Debug mode should keep traceback logging for diagnosis."""
        from winml.modelkit.commands.export import export

        with (
            patch("winml.modelkit.loader.load_hf_model", side_effect=ValueError("boom")),
            patch("winml.modelkit.commands.export.logger") as mock_logger,
        ):
            output_path = tmp_path / "model.onnx"
            result = runner.invoke(
                export,
                ["--model", "test-model", "--output", str(output_path)],
                obj={"debug": True},
            )

        assert result.exit_code != 0
        mock_logger.exception.assert_called_once_with("Export failed")


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
                "winml.modelkit.export.resolve_export_config",
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


class TestExportTaskValidation:
    """Test export surfaces (model, task) incompatibility cleanly.

    Mirrors `winml config`'s behavior: a `ValueError` raised by Optimum's
    `TasksManager` for an unsupported (model_type, task) pair must surface as
    a clean `click.UsageError` instead of falling through to a misleading
    "Unrecognized configuration class" traceback from `load_hf_model`.

    The narrow exception type (`ONNXConfigNotFoundError`, a `ValueError`
    subclass) used for models not registered in Optimum at all must continue
    to be swallowed so registry-only models (e.g., BLIP-style) are unaffected.
    """

    def test_export_raises_usage_error_for_incompatible_task(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Incompatible (model, task) -> clean UsageError, no misleading traceback.

        Mocks the same ValueError that Optimum's TasksManager raises (and that
        `winml config` already surfaces) to confirm the export command now
        propagates it instead of swallowing it.
        """
        from winml.modelkit.commands.export import export

        optimum_message = (
            "resnet doesn't support task text-classification for the onnx backend. "
            "Supported tasks are: feature-extraction, image-classification."
        )

        with (
            patch("winml.modelkit.loader.load_hf_model") as mock_load,
            patch(
                "winml.modelkit.export.resolve_export_config",
                side_effect=ValueError(optimum_message),
            ),
            patch("winml.modelkit.export.export_pytorch") as mock_export,
        ):
            output_path = tmp_path / "model.onnx"
            result = runner.invoke(
                export,
                [
                    "--model",
                    "microsoft/resnet-50",
                    "--task",
                    "text-classification",
                    "--output",
                    str(output_path),
                ],
                obj={"debug": False},
            )

            assert result.exit_code != 0, (
                f"Expected non-zero exit, got {result.exit_code}\n{result.output}"
            )
            # Exact Optimum message is preserved verbatim, matching `winml config`.
            assert optimum_message in result.output
            # Should NOT be wrapped with the generic "Export failed:" prefix
            # (that's the swallowed-then-rethrown path we're avoiding).
            assert "Export failed" not in result.output
            # Should fail fast — never reach model loading or actual export.
            mock_load.assert_not_called()
            mock_export.assert_not_called()

    def test_export_continues_when_onnx_config_not_found(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """ONNXConfigNotFoundError (model not in Optimum) -> command continues.

        Guards against the regression where naively catching `ValueError`
        would also catch `ONNXConfigNotFoundError` (a `ValueError` subclass)
        and break models that rely on MODEL_BUILD_CONFIGS or manual specs.
        """
        from winml.modelkit.commands.export import export
        from winml.modelkit.export import ONNXConfigNotFoundError

        with (
            patch("winml.modelkit.loader.load_hf_model") as mock_load,
            patch(
                "winml.modelkit.export.resolve_export_config",
                side_effect=ONNXConfigNotFoundError(
                    "No OnnxConfig registered for model_type='some-custom-model'..."
                ),
            ),
        ):
            mock_model = MagicMock()
            mock_load.return_value = (mock_model, None, "feature-extraction")

            output_path = tmp_path / "model.onnx"
            result = runner.invoke(
                export,
                ["--model", "some-custom-model", "--output", str(output_path)],
                obj={"debug": False},
            )

            # ONNXConfigNotFoundError must NOT surface as a UsageError.
            # Command must proceed past auto-resolution into load_hf_model / export.
            mock_load.assert_called_once()
            mock_export_onnx.assert_called_once()
            assert result.exit_code == 0, (
                f"Expected exit 0, got {result.exit_code}\n{result.output}"
            )


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
        """Test export inherits debug mode from parent context."""
        from winml.modelkit.commands.export import export

        output_path = tmp_path / "model.onnx"

        # Pass debug=True in context
        runner.invoke(
            export,
            ["--model", "test-model", "--output", str(output_path)],
            obj={"debug": True},
        )

        # verbose should be True due to debug mode
        call_kwargs = mock_export_onnx.call_args.kwargs
        assert call_kwargs["verbose"] is True


class TestExportComposite:
    """Test export fans a composite model out into <output-stem>_<name>.onnx files."""

    def test_composite_exports_one_onnx_per_component(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A composite writes each sub-model to a stem-suffixed path with its own task."""
        from winml.modelkit.commands.export import export
        from winml.modelkit.export import WinMLExportConfig
        from winml.modelkit.loader import WinMLLoaderConfig

        components = {
            "decoder_prefill": "feature-extraction",
            "decoder_gen": "text2text-generation",
        }
        output_path = tmp_path / "qwen3.onnx"

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=components,
            ),
            patch("winml.modelkit.loader.load_hf_model") as mock_load,
            patch("winml.modelkit.export.resolve_export_config") as mock_resolve_cfg,
        ):
            # Echo the per-component task back as the detected task so it flows
            # through to export_onnx and can be verified per sub-model.
            mock_load.side_effect = lambda _model, task=None: (MagicMock(), None, task)
            mock_resolve_cfg.return_value = (
                WinMLExportConfig(),
                WinMLLoaderConfig(task="text-generation"),
            )
            result = runner.invoke(
                export,
                ["--model", "Qwen/Qwen3-0.6B", "--output", str(output_path)],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        # One export per component, each to <output-stem>_<name>.onnx (flat layout).
        assert mock_export_onnx.call_count == len(components)
        exported_paths = {
            Path(call.kwargs["output_path"]) for call in mock_export_onnx.call_args_list
        }
        assert exported_paths == {
            output_path.with_stem(f"{output_path.stem}_{name}") for name in components
        }

        # Each sub-model's own task must be propagated (not the outer task/None) to
        # resolve_export_config, load_hf_model, and export_onnx.
        expected = {
            output_path.with_stem(f"{output_path.stem}_{name}"): task
            for name, task in components.items()
        }
        # resolve_export_config + load_hf_model receive each component's task.
        assert {c.kwargs["task"] for c in mock_resolve_cfg.call_args_list} == set(
            components.values()
        )
        assert {c.kwargs["task"] for c in mock_load.call_args_list} == set(components.values())
        # export_onnx receives the matching (path, task) pair for each sub-model.
        actual = {
            Path(call.kwargs["output_path"]): call.kwargs["task"]
            for call in mock_export_onnx.call_args_list
        }
        assert actual == expected

    def test_composite_rejects_input_specs(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--input-specs is ambiguous for a composite and must be a usage error."""
        from winml.modelkit.commands.export import export

        specs_file = tmp_path / "inputs.json"
        specs_file.write_text(json.dumps({"input_ids": {"dtype": "int64", "shape": [1, 8]}}))

        with patch(
            "winml.modelkit.loader.resolution.resolve_composite_components",
            return_value={"decoder_prefill": "feature-extraction"},
        ):
            result = runner.invoke(
                export,
                [
                    "--model",
                    "Qwen/Qwen3-0.6B",
                    "--output",
                    str(tmp_path / "qwen3"),
                    "--input-specs",
                    str(specs_file),
                ],
                obj={"debug": False},
            )

        assert result.exit_code != 0
        assert "composite" in result.output.lower()
        mock_export_onnx.assert_not_called()

    def test_composite_resolution_valueerror_surfaces_as_usage_error(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A ValueError during composite resolution is surfaced, not swallowed."""
        from winml.modelkit.commands.export import export

        with patch(
            "winml.modelkit.loader.resolution.resolve_composite_components",
            side_effect=ValueError("qwen3 has multiple composite exports; pass --task explicitly"),
        ):
            result = runner.invoke(
                export,
                ["--model", "Qwen/Qwen3-0.6B", "--output", str(tmp_path / "qwen3.onnx")],
                obj={"debug": False},
            )

        # Must not silently fall back to a single-model export.
        assert result.exit_code != 0
        assert "multiple composite exports" in result.output
        # Surfaced as a usage error, not swallowed then re-wrapped by the generic
        # "Export failed: ..." handler around the export body.
        assert "Export failed" not in result.output
        mock_export_onnx.assert_not_called()

    def test_composite_resolution_unexpected_error_surfaces(
        self,
        runner: CliRunner,
        mock_export_onnx: MagicMock,
        tmp_path: Path,
    ) -> None:
        """An unexpected error during composite detection is surfaced, not masked."""
        from winml.modelkit.commands.export import export

        with patch(
            "winml.modelkit.loader.resolution.resolve_composite_components",
            side_effect=KeyError("boom"),
        ):
            result = runner.invoke(
                export,
                ["--model", "Qwen/Qwen3-0.6B", "--output", str(tmp_path / "qwen3.onnx")],
                obj={"debug": False},
            )

        # Not downgraded to a silent single-model export.
        assert result.exit_code != 0
        assert "unexpectedly" in result.output.lower()
        mock_export_onnx.assert_not_called()

    def test_composite_partial_export_warns_and_keeps_files(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """If a later sub-model fails, completed outputs are kept and the user is warned."""
        from winml.modelkit.commands.export import export
        from winml.modelkit.export import WinMLExportConfig
        from winml.modelkit.loader import WinMLLoaderConfig

        components = {
            "decoder_prefill": "feature-extraction",
            "decoder_gen": "text2text-generation",
        }
        output_path = tmp_path / "qwen3.onnx"
        first_out = output_path.with_stem(f"{output_path.stem}_decoder_prefill")
        second_out = output_path.with_stem(f"{output_path.stem}_decoder_gen")

        def fake_export_onnx(**kwargs):
            out = Path(kwargs["output_path"])
            if out == second_out:
                raise RuntimeError("second sub-model blew up")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"onnx")

        default_export_cfg = WinMLExportConfig()
        default_loader_cfg = WinMLLoaderConfig(task="text-generation")

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=components,
            ),
            patch("winml.modelkit.loader.load_hf_model") as mock_load,
            patch("winml.modelkit.export.resolve_export_config") as mock_resolve_cfg,
            patch("winml.modelkit.export.export_pytorch", side_effect=fake_export_onnx),
        ):
            mock_load.side_effect = lambda _model, task=None: (MagicMock(), None, task)
            mock_resolve_cfg.return_value = (default_export_cfg, default_loader_cfg)
            result = runner.invoke(
                export,
                ["--model", "Qwen/Qwen3-0.6B", "--output", str(output_path)],
                obj={"debug": False},
            )

        assert result.exit_code != 0
        # We must NOT delete artifacts — the completed sub-model is kept so the user
        # (who may have --overwritten a pre-existing file) decides what to do.
        assert first_out.exists()
        # The user is warned that the export did not finish (and how many were written).
        assert "did not finish" in result.output
        assert "1 sub-model" in result.output


class TestExportManifest:
    """Export writes a export_manifest.json alongside each ONNX artifact."""

    def test_single_export_writes_manifest(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """A single (non-composite) export produces one export_manifest.json."""
        from winml.modelkit.commands.export import export
        from winml.modelkit.export import WinMLExportConfig
        from winml.modelkit.loader import WinMLLoaderConfig

        output_path = tmp_path / "model.onnx"

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                side_effect=OSError("not composite"),
            ),
            patch("winml.modelkit.loader.load_hf_model") as mock_load,
            patch("winml.modelkit.export.resolve_export_config") as mock_resolve_cfg,
            patch("winml.modelkit.export.export_pytorch") as mock_export,
        ):
            mock_load.return_value = (MagicMock(), None, "image-classification")
            mock_resolve_cfg.return_value = (
                WinMLExportConfig(),
                WinMLLoaderConfig(task="image-classification"),
            )
            mock_export.return_value = {"input_count": 1}
            result = runner.invoke(
                export,
                ["--model", "microsoft/resnet-50", "--output", str(output_path)],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        manifest_path = tmp_path / "export_manifest.json"
        assert manifest_path.exists(), "export_manifest.json was not created"
        data = json.loads(manifest_path.read_text())
        assert data["source"] == "export"
        assert data["final_artifact"] == "model.onnx"
        assert data["model_id"] == "microsoft/resnet-50"
        assert data["schema_version"] == 1
        assert len(data["stages"]) == 1
        assert data["stages"][0]["name"] == "export"
        assert data["stages"][0]["status"] == "completed"

    def test_composite_export_writes_prefixed_manifests(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Each composite sub-model gets a <stem>_export_manifest.json."""
        from winml.modelkit.commands.export import export
        from winml.modelkit.export import WinMLExportConfig
        from winml.modelkit.loader import WinMLLoaderConfig

        components = {
            "decoder_prefill": "feature-extraction",
            "decoder_gen": "text2text-generation",
        }
        output_path = tmp_path / "qwen3.onnx"

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=components,
            ),
            patch("winml.modelkit.loader.load_hf_model") as mock_load,
            patch("winml.modelkit.export.resolve_export_config") as mock_resolve_cfg,
            patch("winml.modelkit.export.export_pytorch") as mock_export,
        ):
            mock_load.side_effect = lambda _model, task=None: (MagicMock(), None, task)
            mock_resolve_cfg.return_value = (
                WinMLExportConfig(),
                WinMLLoaderConfig(task="text-generation"),
            )
            mock_export.return_value = {"input_count": 2}
            result = runner.invoke(
                export,
                ["--model", "Qwen/Qwen3-0.6B", "--output", str(output_path)],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output

        for name in components:
            stem = f"{output_path.stem}_{name}"
            manifest = tmp_path / f"{stem}_export_manifest.json"
            assert manifest.exists(), f"{manifest.name} was not created"
            data = json.loads(manifest.read_text())
            assert data["source"] == "export"
            assert data["final_artifact"] == f"{stem}.onnx"
            assert data["schema_version"] == 1

        # No un-prefixed manifest for composite exports
        assert not (tmp_path / "export_manifest.json").exists()
