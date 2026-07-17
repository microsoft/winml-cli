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
        assert "WinML CLI" in result.output
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


class TestCommandTypoSuggestion:
    """Test did-you-mean hints for mistyped subcommand names (issue #508).

    ``LazyGroup.resolve_command`` seeds ``self.commands`` from the
    filesystem-discovered names so Click 8.4+'s built-in
    :class:`click.exceptions.NoSuchCommand` suggester can find candidates.
    The output matches the UX of ``git`` / ``gh`` / ``cargo`` / ``kubectl``.
    """

    def test_typo_suggests_closest_command(self, runner: CliRunner) -> None:
        """`winml exprt` -> suggest 'export'."""
        result = runner.invoke(main, ["exprt"])
        assert result.exit_code != 0
        assert "No such command 'exprt'." in result.output
        assert "Did you mean 'export'?" in result.output

    def test_typo_suggests_for_transposition(self, runner: CliRunner) -> None:
        """`winml exoprt` (transposition) -> suggest 'export'."""
        result = runner.invoke(main, ["exoprt"])
        assert result.exit_code != 0
        assert "Did you mean 'export'?" in result.output

    def test_unknown_command_with_no_close_match_is_unchanged(
        self, runner: CliRunner
    ) -> None:
        """Garbage input -> original error, no spurious suggestion."""
        result = runner.invoke(main, ["xyzzy"])
        assert result.exit_code != 0
        assert "No such command 'xyzzy'." in result.output
        assert "Did you mean" not in result.output


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
        # _gather_ep_info returns dict[ep_name, {entries}].
        mock_eps = {
            "CPUExecutionProvider": {
                "entries": [
                    # source_kind carries the EPSource subclass name (matches
                    # what _describe_source emits in production), so the
                    # renderer's _SOURCE_KIND_LABEL["BuiltinSource"] lookup is
                    # exercised end-to-end through this fixture.
                    {"status": "primary", "source_kind": "BuiltinSource", "dll_path": None}
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
        # executionProviders is dict[ep_name, {entries}] per the
        # comprehensive inventory shape; one key per detected EP.
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


class TestSysListEpEndToEnd:
    """End-to-end coverage for ``_gather_ep_info`` shape (review I-5).

    Other ``TestSysCommand`` tests mock ``_gather_ep_info`` to return a
    canned dict. Here we mock only the slow boundary calls
    (``_get_pkg_manager``, ``_get_catalog``) and let the real handler
    walk the default EP source list, derive per-entry ``status``,
    and emit the JSON shape.
    """

    @pytest.fixture(autouse=True)
    def _isolate_for_e2e(self, monkeypatch):
        """Make _gather_ep_info hermetic without mocking it out entirely."""
        from winml.modelkit import ep_path as _ep

        monkeypatch.setattr(_ep, "_default_ep_sources", list)
        monkeypatch.setattr(_ep, "_get_catalog", lambda: None)
        monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: None)
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        # Force compat detection to pretend nothing is detected so
        # vendor-constrained EPs come out as `compatible=False`.
        _ep._get_detected_vendors.cache_clear()
        monkeypatch.setattr(
            _ep,
            "_get_detected_vendors",
            lambda: frozenset({"Qualcomm Inc"}),
        )

    def test_json_shape_has_all_required_fields(self, runner: CliRunner) -> None:
        import json

        result = runner.invoke(main, ["sys", "--list-ep", "--format", "json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        eps = data["executionProviders"]
        assert isinstance(eps, dict)
        # Built-in CPUExecutionProvider should always be present (no
        # vendor requirement, never incompatible).
        assert "CPUExecutionProvider" in eps
        cpu = eps["CPUExecutionProvider"]
        entries = cpu["entries"]
        assert isinstance(entries, list)
        assert len(entries) >= 1
        first = entries[0]
        assert first["status"] == "primary"
        # v2.9: built-ins flow through the unified EPSource pipeline.
        # source_kind carries the class name like every other source.
        assert first["source_kind"] == "BuiltinSource"

    def test_incompatible_ep_section_marks_entries(
        self, runner: CliRunner, monkeypatch
    ) -> None:
        # Inject a PyPISource for OpenVINO into the default EP source list;
        # with detected vendors = {"Qualcomm Inc"}, OpenVINO must be marked
        # incompatible at the section level AND at every entry level.
        from winml.modelkit import ep_path as _ep
        from winml.modelkit.ep_path import PyPISource

        # Provide a real installed distribution so the source resolves;
        # onnxruntime-ep-openvino is installed in this venv.
        ov_source = PyPISource(
            distribution="onnxruntime-ep-openvino",
            relative_dll=(
                "onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll"
            ),
            eps=("OpenVINOExecutionProvider",),
        )
        monkeypatch.setattr(_ep, "_default_ep_sources", lambda: [ov_source])

        import json

        result = runner.invoke(main, ["sys", "--list-ep", "--format", "json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        ov = data["executionProviders"].get("OpenVINOExecutionProvider")
        if ov is None:
            pytest.skip("onnxruntime-ep-openvino not installed in this venv")
        # Section incompatible -> every entry status overridden to incompatible.
        # (EP-level "compatible" is now derived at render time from row
        # statuses; per-row "compatible" was dead and is no longer emitted.)
        for entry in ov["entries"]:
            assert entry["status"] == "incompatible"


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
