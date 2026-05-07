# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI integration tests for winml command.

Tests the CLI interface using Click's CliRunner to ensure commands work
correctly without executing actual model exports (which are slow).

Test Categories:
1. Basic CLI functionality (version, help)
2. Command discovery
3. Export command argument validation
4. Sysinfo command output formats
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch


if TYPE_CHECKING:
    from pathlib import Path

import pytest
from click.testing import CliRunner

from winml.modelkit.cli import main


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_version(self, runner: CliRunner) -> None:
        """Test --version flag shows version info."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "winml" in result.output.lower()

    def test_help(self, runner: CliRunner) -> None:
        """Test --help shows usage information."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "WML ModelKit" in result.output
        assert "export" in result.output.lower()

    def test_debug_flag(self, runner: CliRunner) -> None:
        """Test --debug flag is accepted."""
        result = runner.invoke(main, ["--debug", "--help"])
        assert result.exit_code == 0


class TestCommandDiscovery:
    """Test command auto-discovery from commands/ directory."""

    def test_export_command_discovered(self, runner: CliRunner) -> None:
        """Test export command is discovered and available."""
        result = runner.invoke(main, ["export", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()
        assert "output" in result.output.lower()

    def test_sys_command_discovered(self, runner: CliRunner) -> None:
        """Test sys command is discovered and available."""
        result = runner.invoke(main, ["sys", "--help"])
        assert result.exit_code == 0
        assert "format" in result.output.lower()


class TestExportCommand:
    """Test export command functionality."""

    def test_export_requires_model(self, runner: CliRunner) -> None:
        """Test export fails without --model argument."""
        result = runner.invoke(main, ["export", "--output", "test.onnx"])
        assert result.exit_code != 0
        assert "model" in result.output.lower() or "required" in result.output.lower()

    def test_export_requires_output(self, runner: CliRunner) -> None:
        """Test export fails without --output argument."""
        result = runner.invoke(main, ["export", "--model", "test-model"])
        assert result.exit_code != 0
        assert "output" in result.output.lower() or "required" in result.output.lower()

    def test_export_help(self, runner: CliRunner) -> None:
        """Test export --help shows all options."""
        result = runner.invoke(main, ["export", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--output" in result.output
        assert "--verbose" in result.output

    def test_export_short_flags(self, runner: CliRunner) -> None:
        """Test export short flags are documented."""
        result = runner.invoke(main, ["export", "--help"])
        assert result.exit_code == 0
        assert "-m" in result.output
        assert "-o" in result.output
        assert "-v" in result.output

    @patch("winml.modelkit.loader.load_hf_model")
    @patch("winml.modelkit.export.export_pytorch")
    def test_export_calls_api(
        self,
        mock_export_onnx: MagicMock,
        mock_load_hf_model: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test export command delegates to export_onnx correctly."""
        # Setup mock model loader
        mock_model = MagicMock()
        mock_load_hf_model.return_value = (mock_model, None, "image-classification")

        # Setup mock export_onnx
        output_path = tmp_path / "model.onnx"
        mock_export_onnx.return_value = output_path

        runner.invoke(
            main,
            [
                "export",
                "--model",
                "test-model",
                "--output",
                str(output_path),
            ],
        )

        # Verify export_onnx was called correctly
        assert mock_export_onnx.called
        call_kwargs = mock_export_onnx.call_args.kwargs
        assert call_kwargs["model_id"] == "test-model"
        assert call_kwargs["task"] == "image-classification"


class TestSysCommand:
    """Test sys command functionality.

    Device and EP detection use WMI/PowerShell queries that are slow on CI,
    so we mock _gather_device_info and _gather_ep_info to avoid timeouts.
    """

    @pytest.fixture(autouse=True)
    def _mock_hw_detection(self):
        """Mock slow hardware detection to prevent CI timeouts."""
        mock_devices = [{"priority": 1, "type": "CPU", "name": "Mock CPU", "details": {}}]
        # _gather_ep_info returns dict[ep_name, {compatible, device_types, entries}].
        mock_eps = {
            "CPUExecutionProvider": {
                "compatible": True,
                "device_types": "CPU",
                "entries": [
                    {"status": "primary", "source_kind": "built-in", "dll_path": None}
                ],
            }
        }
        with (
            patch("winml.modelkit.commands.sys._gather_device_info", return_value=mock_devices),
            patch("winml.modelkit.commands.sys._gather_ep_info", return_value=mock_eps),
        ):
            yield

    def test_sys_help(self, runner: CliRunner) -> None:
        """Test sys --help shows options."""
        result = runner.invoke(main, ["sys", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "json" in result.output.lower()
        assert "compact" in result.output.lower()

    def test_sys_default_format(self, runner: CliRunner) -> None:
        """Test sys with default (text) format."""
        result = runner.invoke(main, ["sys"])
        assert result.exit_code == 0
        assert "Python" in result.output or "python" in result.output.lower()

    def test_sys_json_format(self, runner: CliRunner) -> None:
        """Test sys with JSON format."""
        result = runner.invoke(main, ["sys", "--format", "json"])
        assert result.exit_code == 0
        # Should be valid JSON
        import json

        data = json.loads(result.output)
        assert "python" in data
        assert "libraries" in data

    def test_sys_compact_format(self, runner: CliRunner) -> None:
        """Test sys with compact format."""
        result = runner.invoke(main, ["sys", "--format", "compact"])
        assert result.exit_code == 0
        assert "Python" in result.output

    def test_sys_verbose(self, runner: CliRunner) -> None:
        """Test sys with verbose flag."""
        result = runner.invoke(main, ["sys", "--verbose"])
        assert result.exit_code == 0

    def test_sys_list_device_list_ep_json_is_valid_single_object(self, runner: CliRunner) -> None:
        """--list-device --list-ep --format json must emit one valid JSON object."""
        import json

        result = runner.invoke(main, ["sys", "--list-device", "--list-ep", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "devices" in data
        assert "executionProviders" in data
        assert isinstance(data["devices"], list)
        # executionProviders is dict[ep_name, {compatible, device_types, entries}]
        # per the comprehensive inventory shape; one key per detected EP.
        assert isinstance(data["executionProviders"], dict)
        assert "CPUExecutionProvider" in data["executionProviders"]

    def test_sys_list_device_compact(self, runner: CliRunner) -> None:
        """--list-device --format compact must produce compact output, not text table."""
        result = runner.invoke(main, ["sys", "--list-device", "--format", "compact"])
        assert result.exit_code == 0
        assert "CPU" in result.output
        # Compact output is a single line; no Rich panel borders
        assert "Available Devices" not in result.output

    def test_sys_list_ep_compact(self, runner: CliRunner) -> None:
        """--list-ep --format compact must produce compact output, not text table."""
        result = runner.invoke(main, ["sys", "--list-ep", "--format", "compact"])
        assert result.exit_code == 0
        assert "CPUExecutionProvider" in result.output
        # Compact output is a single line; no Rich panel headers
        assert "Available Execution Providers" not in result.output


class TestDisabledCommands:
    """Test that disabled commands (run, serve) are hidden and reject invocations."""

    def test_run_not_in_help(self, runner: CliRunner) -> None:
        """Disabled 'run' command must not appear in --help output."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        # 'run' should not be listed as a command
        command_lines = result.output.split("Commands:")[1] if "Commands:" in result.output else ""
        assert "run" not in command_lines.split()

    def test_serve_not_in_help(self, runner: CliRunner) -> None:
        """Disabled 'serve' command must not appear in --help output."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        command_lines = result.output.split("Commands:")[1] if "Commands:" in result.output else ""
        assert "serve" not in command_lines.split()

    def test_run_invocation_rejected(self, runner: CliRunner) -> None:
        """Invoking 'winml run' must fail with a clear disabled message."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code != 0
        assert "disabled" in result.output.lower()

    def test_serve_invocation_rejected(self, runner: CliRunner) -> None:
        """Invoking 'winml serve' must fail with a clear disabled message."""
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code != 0
        assert "disabled" in result.output.lower()


class TestModuleExecution:
    """Test python -m winml.modelkit execution."""

    def test_module_imports(self) -> None:
        """Test __main__ module can be imported."""
        from winml.modelkit import __main__

        assert hasattr(__main__, "main")

    def test_cli_imports(self) -> None:
        """Test cli module can be imported."""
        from winml.modelkit import cli

        assert hasattr(cli, "main")
        assert callable(cli.main)
