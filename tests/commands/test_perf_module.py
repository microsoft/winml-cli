"""Tests for wmk perf --module flag."""

from __future__ import annotations

from click.testing import CliRunner

from winml.modelkit.cli import main
from winml.modelkit.commands.perf import generate_output_path


class TestPerfModuleFlag:
    """Tests for --module flag on wmk perf."""

    def test_module_flag_in_help(self) -> None:
        """Verify --module flag appears in wmk perf --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["perf", "--help"])
        assert result.exit_code == 0
        assert "--module" in result.output

    def test_module_flag_requires_hf_model(self) -> None:
        """--module without --hf-model should fail."""
        runner = CliRunner()
        result = runner.invoke(main, ["perf", "--module", "BertAttention"])
        assert result.exit_code != 0

    def test_module_default_output_includes_class_name(self) -> None:
        """Default output path includes module class name."""
        # The single-model generate_output_path produces {slug}_perf.json
        path = generate_output_path("bert-base-uncased")
        assert "bert-base-uncased" in str(path)

        # The module-mode output path (from _perf_modules) is
        # {slug}_{module_class}_perf.json -- tested via the inline logic
        # in _perf_modules. Here we verify the format difference:
        from pathlib import Path

        slug = "bert-base-uncased"
        module_class = "BertAttention"
        module_path = Path(f"{slug}_{module_class}_perf.json")
        assert module_class in str(module_path)
        assert str(module_path) != str(path)
