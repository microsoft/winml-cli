# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Integration tests for telemetry wiring in the top-level CLI."""

from click.testing import CliRunner

from winml.modelkit.cli import main
from winml.modelkit.telemetry import ActionGroup
from winml.modelkit.telemetry import telemetry as telemetry_mod


# `_reset_telemetry_singleton` (autouse) comes from tests/conftest.py.


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
    is managed by editing ``%USERPROFILE%\\.winml\\config.json``."""
    runner = CliRunner()
    result = runner.invoke(main, ["telemetry", "--help"])
    assert result.exit_code != 0


def test_shutdown_telemetry_does_not_materialize_singleton():
    """Regression: ``_shutdown_telemetry`` (registered via
    ``ctx.call_on_close``) must not build a fresh Telemetry on the way
    out. Without the guard, a path that runs ``main``'s body but never
    reaches ``wrapped_invoke`` (e.g. a Click usage error after the
    group callback) would trigger first-run consent resolution during
    process shutdown in production builds with a real iKey.

    Calling the function directly exercises the guard regardless of the
    Click invocation path.
    """
    from winml.modelkit.cli import _shutdown_telemetry

    assert telemetry_mod._INSTANCE is None
    _shutdown_telemetry()
    assert telemetry_mod._INSTANCE is None
