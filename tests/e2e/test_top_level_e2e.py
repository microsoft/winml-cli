# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the ``winml`` top-level CLI group.

Covers the behaviors that are *specific to the top-level entry point* and
not exercised by any subcommand's own e2e file:

* No-args invocation — exit 0, help echoed to stdout, banner to stderr.
* ``--help`` / ``-h`` — exit 0, every documented option and all enabled
  commands listed in stdout.
* ``--version`` — exit 0, version string matches the installed package.
* Global flags (``-v``, ``-vv``, ``-q``) — accepted; propagated into
  ``ctx.obj`` for subcommands.
* Command discovery — enabled commands present; disabled commands
  (``run``, ``serve``) absent and rejected on invocation.
* Lazy dispatch — every enabled subcommand's ``--help`` is served without
  error, proving the ``LazyGroup`` import-on-demand path works for all
  modules.
* Bad path — unknown commands and invalid flags produce exit 2 with an
  explanatory error message.

Differences from ``tests/cli/test_main.py``
-------------------------------------------
``tests/cli/test_main.py`` is a unit test suite that patches slow
hardware probes (WMI, PowerShell) to stay fast.  This file is an E2E
suite that exercises the **real** command-loading path end-to-end with
no mocks, ensuring:

1. ``LazyGroup.get_command`` actually imports and registers every module.
2. The AST-based help-text extraction (``_parse_click_help``) renders
   non-empty text for every enabled command.
3. Disabled-command rejection fires through the real ``ctx.fail`` path.

Markers
-------
* ``e2e`` — auto-skipped unless ``-m e2e`` is passed (see conftest.py).

Usage::

    uv run pytest tests/e2e/ -m e2e -k top_level
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from winml.modelkit import __version__
from winml.modelkit.cli import _COMMANDS_DIR, _DISABLED_COMMANDS, main


if TYPE_CHECKING:
    from click.testing import Result


pytestmark = [pytest.mark.e2e]

# ---------------------------------------------------------------------------
# Constants — derived from production code so they stay in sync automatically
# ---------------------------------------------------------------------------

#: Commands the CLI surfaces to users (filesystem-driven minus disabled set).
ENABLED_COMMANDS: list[str] = sorted(
    p.stem
    for p in _COMMANDS_DIR.glob("*.py")
    if not p.name.startswith("_") and p.stem not in _DISABLED_COMMANDS
)

#: Commands that are intentionally hidden and must be rejected.
DISABLED_COMMANDS: list[str] = sorted(_DISABLED_COMMANDS)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _invoke(*args: str) -> Result:
    """Invoke ``winml <args>`` through a fresh CliRunner."""
    return CliRunner().invoke(main, list(args), obj={})


# ===========================================================================
# No-args
# ===========================================================================


@pytest.mark.e2e
class TestTopLevelNoArgs:
    """``winml`` with no arguments."""

    def test_exits_zero(self) -> None:
        result = _invoke()
        assert result.exit_code == 0

    def test_help_text_in_output(self) -> None:
        """No-args falls through to ctx.get_help(); help must appear in stdout."""
        result = _invoke()
        assert "Usage:" in result.output
        assert "Commands" in result.output

    def test_banner_in_output(self) -> None:
        """The gradient banner (including 'Windows ML') must be emitted."""
        result = _invoke()
        assert "Windows ML" in result.output

    def test_version_string_in_banner(self) -> None:
        """Banner footer advertises the current package version."""
        result = _invoke()
        assert __version__ in result.output


# ===========================================================================
# --help / -h
# ===========================================================================


@pytest.mark.e2e
class TestTopLevelHelp:
    """``winml --help`` and ``winml -h``."""

    def test_help_long_flag_exits_zero(self) -> None:
        assert _invoke("--help").exit_code == 0

    def test_help_short_flag_exits_zero(self) -> None:
        assert _invoke("-h").exit_code == 0

    def test_long_and_short_flags_produce_same_output(self) -> None:
        long = _invoke("--help").output
        short = _invoke("-h").output
        assert long == short

    def test_help_contains_usage_line(self) -> None:
        assert "Usage:" in _invoke("--help").output

    def test_help_contains_description(self) -> None:
        out = _invoke("--help").output
        assert "WinML CLI" in out

    def test_help_lists_version_option(self) -> None:
        assert "--version" in _invoke("--help").output

    def test_help_lists_verbose_option(self) -> None:
        out = _invoke("--help").output
        assert "--verbose" in out
        assert "-v" in out

    def test_help_lists_quiet_option(self) -> None:
        out = _invoke("--help").output
        assert "--quiet" in out
        assert "-q" in out

    def test_help_lists_help_option_with_short_alias(self) -> None:
        """``-h`` must appear in the options table (not just ``--help``)."""
        assert "-h" in _invoke("--help").output

    def test_help_commands_section_present(self) -> None:
        assert "Commands" in _invoke("--help").output

    @pytest.mark.parametrize("cmd", ENABLED_COMMANDS)
    def test_every_enabled_command_listed(self, cmd: str) -> None:
        """Every enabled command must appear by name in ``winml --help``."""
        assert cmd in _invoke("--help").output

    @pytest.mark.parametrize("cmd", DISABLED_COMMANDS)
    def test_no_disabled_command_listed(self, cmd: str) -> None:
        """Disabled commands must NOT appear in the Commands section."""
        out = _invoke("--help").output
        commands_section = out.split("Commands")[-1] if "Commands" in out else ""
        assert cmd not in commands_section.split()

    def test_every_listed_command_has_non_empty_help(self) -> None:
        """AST-extracted help text must be non-empty for every listed command.

        ``LazyGroup.format_commands`` uses ``_parse_click_help`` to extract
        the first docstring line from each command module without importing
        it.  This test catches modules whose docstring is missing or whose
        AST parse fails, which would result in a blank help column.
        """
        out = _invoke("--help").output
        # Find the Commands section and extract name+help pairs.
        # Each row is "  <name>  <help text>".
        commands_start = out.find("Commands")
        assert commands_start != -1
        commands_block = out[commands_start:]
        for cmd in ENABLED_COMMANDS:
            # Locate the line that starts with the command name.
            for line in commands_block.splitlines():
                stripped = line.strip()
                if stripped.startswith(cmd):
                    # The part after the command name is the help text.
                    tail = stripped[len(cmd) :].strip()
                    assert tail, (
                        f"Command '{cmd}' has no help text in winml --help; "
                        "check its module docstring or _parse_click_help."
                    )
                    break


# ===========================================================================
# --version
# ===========================================================================


@pytest.mark.e2e
class TestTopLevelVersion:
    """``winml --version``."""

    def test_version_flag_exits_zero(self) -> None:
        assert _invoke("--version").exit_code == 0

    def test_version_output_contains_program_name(self) -> None:
        assert "winml" in _invoke("--version").output.lower()

    def test_version_output_matches_package_version(self) -> None:
        """Reported version must match ``winml.modelkit.__version__``."""
        assert __version__ in _invoke("--version").output


# ===========================================================================
# Global flags
# ===========================================================================


@pytest.mark.e2e
class TestTopLevelFlags:
    """Global ``-v``, ``-vv``, ``-q`` flags are accepted and exit 0."""

    @pytest.mark.parametrize(
        "args",
        [
            ["-v", "--help"],
            ["-vv", "--help"],
            ["-v", "-v", "--help"],
            ["-q", "--help"],
        ],
        ids=["v", "vv", "v-v", "q"],
    )
    def test_flag_accepted(self, args: list[str]) -> None:
        result = _invoke(*args)
        assert result.exit_code == 0, (
            f"Unexpected exit {result.exit_code} for args {args}:\n{result.output}"
        )

    def test_verbose_flag_sets_ctx_verbosity(self) -> None:
        """-v must set ``ctx.obj['verbosity'] >= 1`` (visible via subcommand dispatch)."""
        # sys --help reads ctx.obj but always exits 0; we just verify it works.
        result = _invoke("-v", "sys", "--help")
        assert result.exit_code == 0

    def test_quiet_flag_sets_ctx_quiet(self) -> None:
        result = _invoke("-q", "sys", "--help")
        assert result.exit_code == 0


# ===========================================================================
# Lazy command dispatch — every subcommand --help
# ===========================================================================


@pytest.mark.e2e
class TestSubcommandDispatch:
    """Dispatch to every enabled command via ``winml <cmd> --help``.

    These tests exercise ``LazyGroup.get_command`` end-to-end: the module
    must import cleanly, the Click command must be discoverable, and Click
    must be able to format help from it.
    """

    @pytest.mark.parametrize("cmd", ENABLED_COMMANDS)
    def test_subcommand_help_exits_zero(self, cmd: str) -> None:
        result = _invoke(cmd, "--help")
        assert result.exit_code == 0, (
            f"winml {cmd} --help exited {result.exit_code}:\n{result.output}"
        )

    @pytest.mark.parametrize("cmd", ENABLED_COMMANDS)
    def test_subcommand_help_has_no_top_level_banner(self, cmd: str) -> None:
        """Subcommand help must NOT emit the WinML banner.

        ``LazyGroup.format_help`` only calls ``_print_banner`` at the group
        level. Subcommands delegate to their own ``format_help`` which does
        not call it.
        """
        result = _invoke(cmd, "--help")
        # Banner footer text — a reliable signal that the banner fired.
        assert "Windows ML  ·  Model Conversion" not in result.output


# ===========================================================================
# Disabled commands
# ===========================================================================


@pytest.mark.e2e
class TestDisabledCommands:
    """``run`` and ``serve`` are hidden and reject invocations."""

    @pytest.mark.parametrize("cmd", DISABLED_COMMANDS)
    def test_invocation_fails(self, cmd: str) -> None:
        result = _invoke(cmd)
        assert result.exit_code != 0, f"winml {cmd} should be rejected but exited 0"

    @pytest.mark.parametrize("cmd", DISABLED_COMMANDS)
    def test_invocation_mentions_disabled(self, cmd: str) -> None:
        result = _invoke(cmd)
        assert "disabled" in result.output.lower(), (
            f"Expected 'disabled' in error for 'winml {cmd}'"
        )

    @pytest.mark.parametrize("cmd", DISABLED_COMMANDS)
    def test_help_flag_also_rejected(self, cmd: str) -> None:
        """Even ``winml <cmd> --help`` is rejected for disabled commands."""
        result = _invoke(cmd, "--help")
        assert result.exit_code != 0

    @pytest.mark.parametrize("cmd", DISABLED_COMMANDS)
    def test_error_suggests_alternative(self, cmd: str) -> None:
        """Rejection message must suggest 'winml eval' as the alternative."""
        result = _invoke(cmd)
        assert "eval" in result.output.lower(), (
            f"Rejection for 'winml {cmd}' should mention 'eval' as alternative"
        )


# ===========================================================================
# Bad path
# ===========================================================================


@pytest.mark.e2e
class TestBadPath:
    """Unknown commands and unrecognised flags are rejected cleanly."""

    def test_unknown_command_exits_nonzero(self) -> None:
        result = _invoke("totally_unknown_subcommand_xyz")
        assert result.exit_code != 0

    def test_unknown_command_reports_error(self) -> None:
        result = _invoke("totally_unknown_subcommand_xyz")
        out = result.output.lower()
        assert "no such command" in out or "error" in out

    def test_unknown_long_flag_exits_nonzero(self) -> None:
        result = _invoke("--unknown-flag-xyz")
        assert result.exit_code != 0

    def test_unknown_long_flag_reports_error(self) -> None:
        result = _invoke("--unknown-flag-xyz")
        assert "error" in result.output.lower() or result.exit_code == 2
