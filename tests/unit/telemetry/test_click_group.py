# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for ``ActionGroup`` — the Click ``Group`` subclass that
auto-instruments every registered subcommand with ModelKit telemetry."""

from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner

from winml.modelkit.telemetry import consent as consent_mod
from winml.modelkit.telemetry import telemetry as telemetry_mod
from winml.modelkit.telemetry.click_group import ActionGroup
from winml.modelkit.telemetry.telemetry import Telemetry


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the Telemetry singleton between tests so each one starts clean.

    Calls ``shutdown()`` on any pre-existing instance so a real
    ``BatchLogRecordProcessor`` thread (created when a test exercises the
    real LoggerProvider path) does not leak across tests.
    """
    if telemetry_mod._INSTANCE is not None:
        try:
            telemetry_mod._INSTANCE.shutdown()
        except Exception:
            # Best-effort cleanup: a half-initialized singleton from a
            # prior test must not block resetting state for this test.
            pass
    telemetry_mod._INSTANCE = None
    yield
    if telemetry_mod._INSTANCE is not None:
        try:
            telemetry_mod._INSTANCE.shutdown()
        except Exception:
            # Same rationale as above; teardown must always reach the
            # _INSTANCE = None reset below.
            pass
    telemetry_mod._INSTANCE = None


@pytest.fixture
def enabled_telemetry(monkeypatch, isolated_config, clean_env):
    """Set up the environment for a fully-enabled Telemetry singleton.

    The singleton itself is constructed lazily by ``ActionGroup.invoke``
    on the first CLI invocation. Tests that want to capture emits should
    eagerly call :func:`Telemetry.get_or_init` and replace ``_logger``
    with a mock before invoking the CLI.

    ``isolated_config`` and ``clean_env`` come from
    ``tests/unit/telemetry/conftest.py``.
    """
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "o:test-key")
    consent_mod._write_stored_consent("enabled")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)


def _capture_logger(telemetry):
    """Replace the underlying logger with a mock that records emit() calls."""
    telemetry._logger = MagicMock()
    return telemetry._logger


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
    _capture_logger(telemetry)

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
    mock_logger = _capture_logger(telemetry)

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
    mock_logger = _capture_logger(telemetry)

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
    mock_logger = _capture_logger(telemetry)

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
