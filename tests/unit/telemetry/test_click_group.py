# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for ``ActionGroup`` — the Click ``Group`` subclass that
auto-instruments every registered subcommand with ModelKit telemetry."""

from unittest.mock import MagicMock

import click
from click.testing import CliRunner

from winml.modelkit.telemetry import ActionGroup, Telemetry
from winml.modelkit.telemetry import telemetry as telemetry_mod


# `_reset_singleton` (autouse) and `enabled_telemetry` come from
# tests/unit/telemetry/conftest.py.


def _with_mock_logger(t: Telemetry) -> MagicMock:
    """Replace ``t._logger`` with a ``MagicMock`` and return it."""
    t._logger = MagicMock()
    return t._logger


def test_action_group_registers_subcommand(enabled_telemetry):
    @click.group(cls=ActionGroup)
    def cli():
        pass

    @cli.command()
    def build():
        click.echo("built")

    # Pre-create the singleton and mock the logger so this test does not
    # spin up a real BatchLogRecordProcessor thread / network exporter.
    telemetry = Telemetry.get_or_init()
    _with_mock_logger(telemetry)

    runner = CliRunner()
    result = runner.invoke(cli, ["build"])
    assert result.exit_code == 0
    assert "built" in result.output


def test_heartbeat_and_action_emitted_on_success(enabled_telemetry):
    @click.group(cls=ActionGroup)
    def cli():
        pass

    @cli.command()
    @click.option("--device")
    @click.option("--ep")
    def build(device, ep):
        click.echo("built")

    telemetry = Telemetry.get_or_init()
    mock_logger = _with_mock_logger(telemetry)

    runner = CliRunner()
    result = runner.invoke(cli, ["build", "--device", "NPU", "--ep", "QNN"])
    assert result.exit_code == 0

    emit_calls = mock_logger.emit.call_args_list
    event_names = [str(c.args[0].body) for c in emit_calls]
    assert event_names == ["ModelKitHeartbeat", "ModelKitAction"]

    action_record = emit_calls[1].args[0]
    attrs = dict(action_record.attributes)
    assert attrs["action_name"] == "build"
    assert attrs["device"] == "NPU"
    assert attrs["ep"] == "QNN"
    assert attrs["success"] is True
    assert isinstance(attrs["duration_ms"], int)


def test_command_without_device_or_ep_params_sends_null(enabled_telemetry):
    @click.group(cls=ActionGroup)
    def cli():
        pass

    @cli.command()
    def analyze():
        click.echo("analyzed")

    telemetry = Telemetry.get_or_init()
    mock_logger = _with_mock_logger(telemetry)

    runner = CliRunner()
    runner.invoke(cli, ["analyze"])
    action_record = mock_logger.emit.call_args_list[1].args[0]
    attrs = dict(action_record.attributes)
    assert attrs["device"] is None
    assert attrs["ep"] is None


def test_exception_emits_error_and_action_failure(enabled_telemetry):
    @click.group(cls=ActionGroup)
    def cli():
        pass

    @cli.command()
    def blowup():
        raise ValueError("boom")

    telemetry = Telemetry.get_or_init()
    mock_logger = _with_mock_logger(telemetry)

    runner = CliRunner()
    result = runner.invoke(cli, ["blowup"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)

    event_names = [str(c.args[0].body) for c in mock_logger.emit.call_args_list]
    assert event_names == ["ModelKitHeartbeat", "ModelKitError", "ModelKitAction"]

    action_record = mock_logger.emit.call_args_list[2].args[0]
    assert dict(action_record.attributes)["success"] is False


def test_disabled_telemetry_emits_nothing(monkeypatch):
    """Empty iKey -> Telemetry disabled -> no emits, no crash."""
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "")

    @click.group(cls=ActionGroup)
    def cli():
        pass

    @cli.command()
    def build():
        click.echo("built")

    runner = CliRunner()
    result = runner.invoke(cli, ["build"])
    assert result.exit_code == 0


def test_group_help_does_not_init_telemetry(enabled_telemetry):
    @click.group(cls=ActionGroup)
    def cli():
        pass

    @cli.command()
    def build():
        click.echo("built")

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    # Telemetry singleton must not even have been materialized — that is
    # what proves no prompt and no emit would ever happen for --help.
    assert telemetry_mod._INSTANCE is None


def test_subcommand_help_does_not_emit(enabled_telemetry):
    @click.group(cls=ActionGroup)
    def cli():
        pass

    @cli.command()
    def build():
        click.echo("built")

    runner = CliRunner()
    result = runner.invoke(cli, ["build", "--help"])
    assert result.exit_code == 0
    # Subcommand --help short-circuits inside Click's parsing before
    # the wrapped invoke runs, so no emits.
    if telemetry_mod._INSTANCE is not None:
        logger = telemetry_mod._INSTANCE._logger
        assert logger is None or not logger.emit.called
