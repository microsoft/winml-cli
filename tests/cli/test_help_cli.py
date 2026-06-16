# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI surface tests for ``winml`` (no args) and ``winml --help``.

Both invocations follow the same contract: exit 0 and render the full
help page, which consists of the gradient banner on stderr and the Click
help text (Usage / Options / Commands) on stdout.  The tests here pin the
*observable output contract* of these two entry points — no mocks, no
subcommand execution.

Coverage
--------
``TestWinmlNoArgs``
    ``winml`` with no arguments — exit code, banner, help content.

``TestWinmlHelp``
    ``winml --help`` and its ``-h`` short alias — exit code, parity with
    no-args output, content completeness.

``TestCommandList``
    The Commands section lists exactly the right commands: every enabled
    command appears with non-empty AST-extracted help text; disabled
    commands (``run``, ``serve``) do not appear and are also rejected at
    invocation time.

``TestOptionsSection``
    Every documented top-level option is present in the help output,
    including ``--version`` which is also verified to execute correctly.

These tests run under the default CI filter (no special marker required).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner, Result

from winml.modelkit import __version__
from winml.modelkit.cli import _COMMANDS_DIR, _DISABLED_COMMANDS, main


# ---------------------------------------------------------------------------
# Expected command sets — derived from production code, stay in sync
# ---------------------------------------------------------------------------

ENABLED_COMMANDS: list[str] = sorted(
    p.stem
    for p in _COMMANDS_DIR.glob("*.py")
    if not p.name.startswith("_") and p.stem not in _DISABLED_COMMANDS
)
assert ENABLED_COMMANDS, (
    f"No enabled commands discovered under {_COMMANDS_DIR}; "
    "refusing to silently turn parametrized tests into zero cases."
)

_DISABLED_LIST: list[str] = sorted(_DISABLED_COMMANDS)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _invoke(*args: str) -> Result:
    return CliRunner().invoke(main, list(args), obj={})


# ===========================================================================
# winml  (no args)
# ===========================================================================


class TestWinmlNoArgs:
    """``winml`` invoked with no arguments."""

    def test_exits_zero(self) -> None:
        assert _invoke().exit_code == 0

    def test_usage_line_present(self) -> None:
        assert "Usage:" in _invoke().output

    def test_commands_section_present(self) -> None:
        assert "Commands" in _invoke().output

    def test_options_section_present(self) -> None:
        assert "Options" in _invoke().output

    def test_banner_present(self) -> None:
        """The 'Windows ML' footer line of the gradient banner must appear on stderr."""
        assert "Windows ML" in _invoke().stderr

    def test_banner_shows_current_version(self) -> None:
        assert __version__ in _invoke().stderr

    def test_description_present(self) -> None:
        assert "WinML CLI" in _invoke().output


# ===========================================================================
# winml --help  /  winml -h
# ===========================================================================


class TestWinmlHelp:
    """``winml --help`` and its ``-h`` short alias."""

    def test_long_flag_exits_zero(self) -> None:
        assert _invoke("--help").exit_code == 0

    def test_short_flag_exits_zero(self) -> None:
        assert _invoke("-h").exit_code == 0

    def test_short_and_long_produce_identical_output(self) -> None:
        assert _invoke("--help").output == _invoke("-h").output

    def test_no_args_and_help_produce_identical_output(self) -> None:
        """``winml`` and ``winml --help`` must render the same help page."""
        assert _invoke().output == _invoke("--help").output

    def test_usage_line_present(self) -> None:
        assert "Usage:" in _invoke("--help").output

    def test_banner_present(self) -> None:
        assert "Windows ML" in _invoke("--help").stderr

    def test_description_present(self) -> None:
        assert "WinML CLI" in _invoke("--help").output

    def test_subcommand_help_has_no_banner(self) -> None:
        """Subcommand ``--help`` must NOT render the top-level banner."""
        result = _invoke("sys", "--help")
        assert result.exit_code == 0
        assert "Windows ML" not in result.stderr


# ===========================================================================
# Commands section
# ===========================================================================


class TestCommandList:
    """The Commands section of ``winml --help`` lists exactly the right set."""

    @pytest.mark.parametrize("cmd", ENABLED_COMMANDS)
    def test_enabled_command_listed(self, cmd: str) -> None:
        """Every enabled command must appear by name in the Commands section."""
        assert cmd in _invoke("--help").output

    @pytest.mark.parametrize("cmd", _DISABLED_LIST)
    def test_disabled_command_not_listed(self, cmd: str) -> None:
        """Disabled commands must not appear in the Commands section."""
        out = _invoke("--help").output
        commands_block = out.split("Commands")[-1] if "Commands" in out else ""
        assert not any(
            line.strip().startswith((cmd + " ", cmd + "\t")) for line in commands_block.splitlines()
        ), f"Disabled command '{cmd}' appears as a row in the Commands section"

    @pytest.mark.parametrize("cmd", _DISABLED_LIST)
    def test_disabled_command_invocation_fails(self, cmd: str) -> None:
        """Invoking a disabled command must exit non-zero with a 'disabled' message."""
        result = _invoke(cmd)
        assert result.exit_code != 0
        assert "disabled" in result.output.lower()

    @pytest.mark.parametrize("cmd", ENABLED_COMMANDS)
    def test_enabled_command_has_help_text(self, cmd: str) -> None:
        """Each command row must show non-empty AST-extracted help text.

        ``LazyGroup.format_commands`` parses each module's docstring via
        AST without importing it.  A blank column means the docstring is
        missing or the parse failed.

        Note: relies on Click's HelpFormatter rendering each command as
        ``<name>  <help>`` on a single line.
        """
        out = _invoke("--help").output
        commands_start = out.find("Commands")
        assert commands_start != -1, "Commands section not found"
        commands_block = out[commands_start:]
        for line in commands_block.splitlines():
            stripped = line.strip()
            if stripped.startswith((cmd + " ", cmd + "\t")):
                tail = stripped[len(cmd) :].strip()
                assert tail, f"'{cmd}' has no help text in winml --help"
                break
        else:
            pytest.fail(
                f"Command '{cmd}' row not found in Commands section — "
                "docstring missing or AST extraction failed"
            )


# ===========================================================================
# Options section
# ===========================================================================


class TestOptionsSection:
    """Every documented top-level option appears in ``winml --help``."""

    @pytest.mark.parametrize(
        "opt",
        ["--version", "--verbose", "-v", "--quiet", "-q", "--help", "-h"],
    )
    def test_option_present(self, opt: str) -> None:
        assert opt in _invoke("--help").output

    def test_version_flag_executes(self) -> None:
        """``winml --version`` must exit 0 and print the current version string."""
        result = _invoke("--version")
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_uses_canonical_product_name(self) -> None:
        """``winml --version`` must brand the product as 'WinML CLI' (issue #510)."""
        assert "WinML CLI" in _invoke("--version").output
