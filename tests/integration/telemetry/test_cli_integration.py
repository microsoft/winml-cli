# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Integration tests for telemetry wiring in the top-level CLI."""

from click.testing import CliRunner

from winml.modelkit.cli import main
from winml.modelkit.telemetry import ActionGroup
from winml.modelkit.telemetry import telemetry as telemetry_mod


# `_reset_singleton` (autouse) comes from
# tests/integration/telemetry/conftest.py.


def test_top_level_help_works():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0


def test_main_group_is_action_group():
    """Regression: ``winml`` must be an ``ActionGroup`` (directly or via
    a subclass like ``LazyGroup``) so every subcommand is auto-instrumented.
    Catches a future regression where someone replaces the Group class
    with a plain ``click.Group``."""
    assert isinstance(main, ActionGroup)


def test_no_telemetry_subcommand():
    """Regression: there is no ``winml telemetry`` subcommand. Consent
    is managed by editing ``%USERPROFILE%\\.modelkit\\config.json``."""
    runner = CliRunner()
    result = runner.invoke(main, ["telemetry", "--help"])
    assert result.exit_code != 0


def test_help_path_does_not_materialize_telemetry_on_shutdown():
    """Regression: ``ctx.call_on_close(_shutdown_telemetry)`` must not
    materialize the Telemetry singleton when no subcommand actually ran.

    Without the guard in ``_shutdown_telemetry``, a Click usage error
    after the group callback registers the close-callback would build a
    fresh Telemetry on the way out — risking a first-run consent prompt
    during process shutdown in production builds with a real iKey.
    """
    runner = CliRunner()
    runner.invoke(main, ["--help"])
    assert telemetry_mod._INSTANCE is None
