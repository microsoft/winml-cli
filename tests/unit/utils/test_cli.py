# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for shared CLI helpers in winml.modelkit.utils.cli."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import click
import pytest
from click.testing import CliRunner

from winml.modelkit.utils.cli import (
    analyze_option,
    build_pipeline_extra_kwargs,
    guard_output,
    ignored_build_flags_warning,
    max_optim_iterations_option,
    no_color_option,
    optimize_option,
    overwrite_option,
    parse_ep_options,
    precision_option,
    quant_option,
)


if TYPE_CHECKING:
    from pathlib import Path


class TestParseEpOptions:
    """Tests for parse_ep_options()."""

    def test_empty_returns_none(self) -> None:
        """No values -> None so callers leave the session default untouched."""
        assert parse_ep_options(()) is None

    def test_single_pair(self) -> None:
        assert parse_ep_options(("htp_performance_mode=burst",)) == {
            "htp_performance_mode": "burst"
        }

    def test_multiple_pairs(self) -> None:
        result = parse_ep_options(
            ("htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3")
        )
        assert result == {
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        }

    def test_value_may_contain_equals(self) -> None:
        """Only the first '=' splits key from value."""
        assert parse_ep_options(("key=a=b=c",)) == {"key": "a=b=c"}

    def test_last_wins_on_duplicate_key(self) -> None:
        assert parse_ep_options(("k=1", "k=2")) == {"k": "2"}

    def test_whitespace_stripped_from_key_and_value(self) -> None:
        """Surrounding whitespace (e.g. from shell quoting) is stripped."""
        assert parse_ep_options(("htp_performance_mode= burst ",)) == {
            "htp_performance_mode": "burst"
        }
        assert parse_ep_options(("  k  =  v  ",)) == {"k": "v"}

    def test_missing_equals_raises(self) -> None:
        with pytest.raises(click.BadParameter):
            parse_ep_options(("no_equals_sign",))

    def test_empty_key_raises(self) -> None:
        with pytest.raises(click.BadParameter):
            parse_ep_options(("=value",))


class TestNoColorOption:
    """Tests for the shared no_color_option() decorator."""

    @staticmethod
    def _make_cmd() -> click.Command:
        @click.command()
        @no_color_option()
        def cmd() -> None:
            click.echo("NO_COLOR" in os.environ)

        return cmd

    def test_no_flag_leaves_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without --no-color, NO_COLOR is not set by the callback."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        result = CliRunner().invoke(self._make_cmd(), [])
        assert result.exit_code == 0
        assert result.output.strip() == "False"

    def test_flag_sets_no_color_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--no-color sets NO_COLOR=1 so Rich disables color for the run."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        result = CliRunner().invoke(self._make_cmd(), ["--no-color"])
        assert result.exit_code == 0
        assert result.output.strip() == "True"

    def test_flag_not_exposed_as_param(self) -> None:
        """expose_value=False: the command takes no extra parameter."""
        result = CliRunner().invoke(self._make_cmd(), ["--no-color"])
        assert result.exit_code == 0

    def test_help_documents_env_vars(self) -> None:
        """Help mentions the NO_COLOR / CI environment fallbacks."""
        result = CliRunner().invoke(self._make_cmd(), ["--help"])
        assert "--no-color" in result.output
        assert "NO_COLOR" in result.output


class TestPrecisionOption:
    """Tests for the shared precision_option() decorator."""

    @staticmethod
    def _make_cmd(**kwargs: object) -> click.Command:
        @click.command()
        @precision_option(**kwargs)  # type: ignore[arg-type]
        def cmd(precision: str) -> None:
            click.echo(precision)

        return cmd

    def test_default_is_auto(self) -> None:
        """No --precision yields the default 'auto'."""
        result = CliRunner().invoke(self._make_cmd(), [])
        assert result.exit_code == 0
        assert result.output.strip() == "auto"

    def test_short_flag_p(self) -> None:
        """-p is registered by default and maps to the precision param."""
        result = CliRunner().invoke(self._make_cmd(), ["-p", "fp16"])
        assert result.exit_code == 0
        assert result.output.strip() == "fp16"

    def test_long_flag(self) -> None:
        """--precision sets the value."""
        result = CliRunner().invoke(self._make_cmd(), ["--precision", "w8a16"])
        assert result.exit_code == 0
        assert result.output.strip() == "w8a16"

    def test_include_short_false_drops_p(self) -> None:
        """include_short=False removes the -p alias but keeps --precision."""
        cmd = self._make_cmd(include_short=False)
        assert CliRunner().invoke(cmd, ["-p", "fp16"]).exit_code != 0
        assert CliRunner().invoke(cmd, ["--precision", "fp16"]).exit_code == 0

    def test_custom_default(self) -> None:
        """default overrides the resolved value when flag is omitted."""
        result = CliRunner().invoke(self._make_cmd(default="int8"), [])
        assert result.output.strip() == "int8"

    def test_optional_message_appended_to_help(self) -> None:
        """optional_message is appended after the base help text."""
        result = CliRunner().invoke(self._make_cmd(optional_message="Extra note."), ["--help"])
        assert "w8a16" in result.output  # base help present
        assert "Extra note." in result.output

    def test_accepts_arbitrary_string(self) -> None:
        """type=str: validation is deferred downstream, so any string parses."""
        result = CliRunner().invoke(self._make_cmd(), ["--precision", "bf16"])
        assert result.exit_code == 0
        assert result.output.strip() == "bf16"

    def test_default_none(self) -> None:
        """default=None (e.g. quantize) yields no value when the flag is omitted."""
        result = CliRunner().invoke(self._make_cmd(default=None), [])
        assert result.exit_code == 0
        assert result.output.strip() == ""

    def test_help_text_override_replaces_base(self) -> None:
        """help_text replaces the default float+int help; optional_message still appends."""
        cmd = self._make_cmd(
            help_text="Quantization precision: int8, int16",
            optional_message="Overridden by --weight-type",
        )
        result = CliRunner().invoke(cmd, ["--help"])
        # Collapse whitespace so wrapped help lines compare as one string.
        flat = " ".join(result.output.split())
        assert "Quantization precision: int8, int16" in flat
        assert "Overridden by --weight-type" in flat
        assert "fp16" not in flat  # base float help is gone


class TestQuantOption:
    """Tests for the shared quant_option() decorator (with --quantize alias)."""

    @staticmethod
    def _make_cmd(**kwargs: object) -> click.Command:
        @click.command()
        @quant_option(**kwargs)  # type: ignore[arg-type]
        def cmd(quant: bool) -> None:
            click.echo(str(quant))

        return cmd

    def test_default_true(self) -> None:
        result = CliRunner().invoke(self._make_cmd(), [])
        assert result.output.strip() == "True"

    @pytest.mark.parametrize(
        ("flag", "expected"),
        [
            ("--quant", "True"),
            ("--no-quant", "False"),
            ("--quantize", "True"),
            ("--no-quantize", "False"),
        ],
    )
    def test_canonical_and_alias_flags(self, flag: str, expected: str) -> None:
        """--quant/--no-quant and the --quantize/--no-quantize alias map to ``quant``."""
        result = CliRunner().invoke(self._make_cmd(), [flag])
        assert result.exit_code == 0
        assert result.output.strip() == expected

    def test_custom_default_false(self) -> None:
        result = CliRunner().invoke(self._make_cmd(default=False), [])
        assert result.output.strip() == "False"

    def test_help_override_and_optional_message(self) -> None:
        cmd = self._make_cmd(help_text="Config quant section", optional_message="Note.")
        flat = " ".join(CliRunner().invoke(cmd, ["--help"]).output.split())
        assert "Config quant section" in flat
        assert "Note." in flat


class TestOptimizeAnalyzeOptions:
    """Tests for the shared optimize_option() / analyze_option() toggles."""

    @staticmethod
    def _make_cmd(decorator) -> click.Command:
        @click.command()
        @decorator()
        def cmd(optimize: bool = True, analyze: bool = True) -> None:
            # Echo whichever flag the decorator registered.
            click.echo(f"optimize={optimize} analyze={analyze}")

        return cmd

    @pytest.mark.parametrize(
        ("decorator", "flag", "needle"),
        [
            (optimize_option, "--no-optimize", "optimize=False"),
            (optimize_option, "--optimize", "optimize=True"),
            (analyze_option, "--no-analyze", "analyze=False"),
            (analyze_option, "--analyze", "analyze=True"),
        ],
    )
    def test_toggle(self, decorator, flag: str, needle: str) -> None:
        result = CliRunner().invoke(self._make_cmd(decorator), [flag])
        assert result.exit_code == 0
        assert needle in result.output


class TestMaxOptimIterationsOption:
    """Tests for the shared max_optim_iterations_option() decorator."""

    @staticmethod
    def _make_cmd(optional_message: str | None = None) -> click.Command:
        @click.command()
        @max_optim_iterations_option(optional_message=optional_message)
        def cmd(max_optim_iterations: int | None) -> None:
            click.echo(repr(max_optim_iterations))

        return cmd

    def test_default_none(self) -> None:
        assert CliRunner().invoke(self._make_cmd(), []).output.strip() == "None"

    def test_int_value(self) -> None:
        result = CliRunner().invoke(self._make_cmd(), ["--max-optim-iterations", "5"])
        assert result.output.strip() == "5"

    def test_rejects_non_int(self) -> None:
        assert CliRunner().invoke(self._make_cmd(), ["--max-optim-iterations", "x"]).exit_code != 0

    def test_optional_message_appended_with_period_separator(self) -> None:
        """optional_message joins the base help with '. ', matching sibling helpers."""
        result = CliRunner().invoke(self._make_cmd(optional_message="Extra note."), ["--help"])
        # Click reflows --help text, so assert on the joined token rather than line layout.
        joined = " ".join(result.output.split())
        assert "to 0. Extra note." in joined


class TestBuildPipelineExtraKwargs:
    """Tests for the shared build_pipeline_extra_kwargs() translator."""

    def test_defaults_are_empty(self) -> None:
        """All-default flags carry the pipeline default, so no keys are emitted."""
        assert build_pipeline_extra_kwargs() == {}

    def test_no_optimize_sets_skip_optimize(self) -> None:
        assert build_pipeline_extra_kwargs(optimize=False) == {"skip_optimize": True}

    def test_no_analyze_zeroes_iterations(self) -> None:
        assert build_pipeline_extra_kwargs(analyze=False) == {"hack_max_optim_iterations": 0}

    def test_max_optim_iterations_forwarded_when_analyzing(self) -> None:
        result = build_pipeline_extra_kwargs(max_optim_iterations=7)
        assert result == {"hack_max_optim_iterations": 7}

    def test_no_analyze_wins_over_explicit_iterations(self) -> None:
        """--no-analyze takes precedence over an explicit --max-optim-iterations."""
        result = build_pipeline_extra_kwargs(analyze=False, max_optim_iterations=7)
        assert result == {"hack_max_optim_iterations": 0}

    def test_combined_optimize_and_analyze(self) -> None:
        result = build_pipeline_extra_kwargs(optimize=False, analyze=False)
        assert result == {"skip_optimize": True, "hack_max_optim_iterations": 0}


class TestIgnoredBuildFlagsWarning:
    """Tests for the shared ignored_build_flags_warning() helper."""

    def test_returns_none_when_build_runs(self) -> None:
        """No warning when a build will run, even with flags set."""
        assert (
            ignored_build_flags_warning(
                skip_build_onnx=False,
                quant=False,
                optimize=False,
                analyze=False,
                max_optim_iterations=5,
            )
            is None
        )

    def test_returns_none_when_no_flags_set(self) -> None:
        """No warning when all flags are at their defaults."""
        assert ignored_build_flags_warning(skip_build_onnx=True) is None

    def test_names_each_set_flag(self) -> None:
        msg = ignored_build_flags_warning(
            skip_build_onnx=True,
            quant=False,
            optimize=False,
            analyze=False,
            max_optim_iterations=5,
        )
        assert msg is not None
        for flag in ("--no-quant", "--no-optimize", "--no-analyze", "--max-optim-iterations"):
            assert flag in msg
        assert "pre-built ONNX" in msg
        assert "--no-skip-build" in msg

    def test_only_includes_set_flags(self) -> None:
        """Unset flags are not named."""
        msg = ignored_build_flags_warning(skip_build_onnx=True, quant=False)
        assert msg is not None
        assert "--no-quant" in msg
        assert "--no-optimize" not in msg
        assert "--no-analyze" not in msg
        assert "--max-optim-iterations" not in msg

    def test_max_optim_zero_counts_as_set(self) -> None:
        """An explicit 0 is a user-set value (only None is the default)."""
        msg = ignored_build_flags_warning(skip_build_onnx=True, max_optim_iterations=0)
        assert msg is not None
        assert "--max-optim-iterations" in msg


class TestOverwriteOption:
    """Tests for the shared overwrite_option() decorator."""

    @staticmethod
    def _make_cmd(optional_message: str | None = None) -> click.Command:
        @click.command()
        @overwrite_option(optional_message=optional_message)
        def cmd(overwrite: bool) -> None:
            click.echo(repr(overwrite))

        return cmd

    def test_default_is_false(self) -> None:
        assert CliRunner().invoke(self._make_cmd(), []).output.strip() == "False"

    def test_overwrite_flag_sets_true(self) -> None:
        assert CliRunner().invoke(self._make_cmd(), ["--overwrite"]).output.strip() == "True"

    def test_no_overwrite_flag_sets_false(self) -> None:
        assert CliRunner().invoke(self._make_cmd(), ["--no-overwrite"]).output.strip() == "False"

    def test_optional_message_appended(self) -> None:
        result = CliRunner().invoke(self._make_cmd(optional_message="Extra note."), ["--help"])
        joined = " ".join(result.output.split())
        assert "Overwrite an existing output" in joined
        assert "Extra note." in joined


class TestGuardOutput:
    """Tests for the shared guard_output() existence check."""

    def test_none_path_is_noop(self) -> None:
        guard_output(None, overwrite=False)  # must not raise

    def test_missing_file_is_noop(self, tmp_path: Path) -> None:
        guard_output(tmp_path / "nope.onnx", overwrite=False)  # must not raise

    def test_existing_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "model.onnx"
        f.write_text("x")
        with pytest.raises(click.ClickException) as exc:
            guard_output(f, overwrite=False)
        assert "already exists" in str(exc.value)
        assert "--overwrite" in str(exc.value)

    def test_existing_file_with_overwrite_is_noop(self, tmp_path: Path) -> None:
        f = tmp_path / "model.onnx"
        f.write_text("x")
        guard_output(f, overwrite=True)  # must not raise

    def test_empty_dir_is_noop(self, tmp_path: Path) -> None:
        d = tmp_path / "out"
        d.mkdir()
        guard_output(d, overwrite=False)  # empty dir must not raise

    def test_non_empty_dir_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "out"
        d.mkdir()
        (d / "artifact.onnx").write_text("x")
        with pytest.raises(click.ClickException) as exc:
            guard_output(d, overwrite=False)
        assert "not empty" in str(exc.value)

    def test_non_empty_dir_with_overwrite_is_noop(self, tmp_path: Path) -> None:
        d = tmp_path / "out"
        d.mkdir()
        (d / "artifact.onnx").write_text("x")
        guard_output(d, overwrite=True)  # must not raise

    def test_custom_label_in_message(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.json"
        f.write_text("x")
        with pytest.raises(click.ClickException) as exc:
            guard_output(f, overwrite=False, label="Optimization config")
        assert "Optimization config" in str(exc.value)

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        f = tmp_path / "model.onnx"
        f.write_text("x")
        with pytest.raises(click.ClickException):
            guard_output(str(f), overwrite=False)
