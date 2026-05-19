# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for ``winml`` (no args) and ``winml --help``.

Both invocations follow the same contract: exit 0 and render the full
help page, which consists of the gradient banner on stderr and the Click
help text (Usage / Options / Commands) on stdout.  The tests here pin the
*observable output contract* of these two entry points end-to-end — no
mocks, no subcommand execution.

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
    commands (``run``, ``serve``) do not appear.

``TestOptionsSection``
    Every documented top-level option is present in the help output.

Marker
------
``e2e`` — auto-skipped unless ``-m e2e`` is passed (see conftest.py).

Usage::

    uv run pytest tests/e2e/ -m e2e -k top_level
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from winml.modelkit import __version__
from winml.modelkit.cli import _COMMANDS_DIR, _DISABLED_COMMANDS, main


pytestmark = [pytest.mark.e2e]

# ---------------------------------------------------------------------------
# Expected command sets — derived from production code, stay in sync
# ---------------------------------------------------------------------------

ENABLED_COMMANDS: list[str] = sorted(
    p.stem
    for p in _COMMANDS_DIR.glob("*.py")
    if not p.name.startswith("_") and p.stem not in _DISABLED_COMMANDS
)

DISABLED_COMMANDS: list[str] = sorted(_DISABLED_COMMANDS)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _invoke(*args: str):
    return CliRunner().invoke(main, list(args), obj={})


# ===========================================================================
# winml  (no args)
# ===========================================================================


@pytest.mark.e2e
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
        """The 'Windows ML' footer line of the gradient banner must appear."""
        assert "Windows ML" in _invoke().output

    def test_banner_shows_current_version(self) -> None:
        assert __version__ in _invoke().output

    def test_description_present(self) -> None:
        assert "WinML CLI" in _invoke().output


# ===========================================================================
# winml --help  /  winml -h
# ===========================================================================


@pytest.mark.e2e
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
        assert "Windows ML" in _invoke("--help").output

    def test_description_present(self) -> None:
        assert "WinML CLI" in _invoke("--help").output


# ===========================================================================
# Commands section
# ===========================================================================


@pytest.mark.e2e
class TestCommandList:
    """The Commands section of ``winml --help`` lists exactly the right set."""

    @pytest.mark.parametrize("cmd", ENABLED_COMMANDS)
    def test_enabled_command_listed(self, cmd: str) -> None:
        """Every enabled command must appear by name in the Commands section."""
        assert cmd in _invoke("--help").output

    @pytest.mark.parametrize("cmd", DISABLED_COMMANDS)
    def test_disabled_command_not_listed(self, cmd: str) -> None:
        """Disabled commands must not appear in the Commands section."""
        out = _invoke("--help").output
        commands_block = out.split("Commands")[-1] if "Commands" in out else ""
        assert cmd not in commands_block.split()

    @pytest.mark.parametrize("cmd", ENABLED_COMMANDS)
    def test_enabled_command_has_help_text(self, cmd: str) -> None:
        """Each command row must show non-empty AST-extracted help text.

        ``LazyGroup.format_commands`` parses each module's docstring via
        AST without importing it.  A blank column means the docstring is
        missing or the parse failed.
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


# ===========================================================================
# Options section
# ===========================================================================


@pytest.mark.e2e
class TestOptionsSection:
    """Every documented top-level option appears in ``winml --help``."""

    @pytest.mark.parametrize(
        "opt",
        ["--version", "--verbose", "-v", "--quiet", "-q", "--help", "-h"],
    )
    def test_option_present(self, opt: str) -> None:
        assert opt in _invoke("--help").output
