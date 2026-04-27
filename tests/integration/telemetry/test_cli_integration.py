# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Integration tests for telemetry wiring in the top-level CLI."""

from click.testing import CliRunner

from winml.modelkit.cli import main
from winml.modelkit.telemetry import ActionGroup


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
