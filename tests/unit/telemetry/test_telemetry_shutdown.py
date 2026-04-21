# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from unittest.mock import MagicMock

import pytest

from winml.modelkit.telemetry import consent as consent_mod
from winml.modelkit.telemetry import telemetry as telemetry_mod
from winml.modelkit.telemetry.telemetry import Telemetry


@pytest.fixture(autouse=True)
def _reset_singleton():
    telemetry_mod._INSTANCE = None
    yield
    telemetry_mod._INSTANCE = None


@pytest.fixture
def enabled_telemetry(monkeypatch, isolated_config, clean_env):
    """Fully-enabled Telemetry with a mock provider (for shutdown assertions)."""
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "o:test-key")
    consent_mod._write_stored_consent("enabled")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    t = Telemetry.get_or_init()
    assert t.disabled is False
    t._provider = MagicMock()
    return t


def test_shutdown_flushes_provider(enabled_telemetry):
    t = enabled_telemetry
    provider = t._provider
    t.shutdown()
    provider.shutdown.assert_called_once()


def test_shutdown_clears_logger_for_post_shutdown_noop(enabled_telemetry):
    t = enabled_telemetry
    provider = t._provider
    t.shutdown()
    # After shutdown the instance must be in a disabled state.
    assert t.disabled is True
    assert t._logger is None
    # And the provider reference is dropped so we can't double-shutdown the
    # same object by accident.
    assert t._provider is None
    # Further emits are no-ops (no crash, no provider interaction).
    t.log_heartbeat()
    provider.shutdown.assert_called_once()  # still just one call


def test_shutdown_is_idempotent(enabled_telemetry):
    t = enabled_telemetry
    provider = t._provider
    t.shutdown()
    t.shutdown()  # must not raise
    assert provider.shutdown.call_count == 1


def test_shutdown_on_disabled_is_noop(monkeypatch):
    telemetry_mod._INSTANCE = None
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "")
    t = Telemetry.get_or_init()
    assert t.disabled is True
    t.shutdown()  # no crash; provider is None to begin with


def test_public_methods_never_raise_even_if_logger_broken(enabled_telemetry):
    """Regression: a telemetry subsystem failure must never take down the
    CLI. If the logger / provider object starts raising, the public
    methods continue to return normally."""
    t = enabled_telemetry
    t._logger = MagicMock()
    t._logger.emit.side_effect = RuntimeError("downstream blew up")

    # None of these should raise.
    t.log_heartbeat()
    t.log_action(
        action_name="x",
        device=None,
        ep=None,
        duration_ms=0,
        success=True,
    )
    try:
        raise ValueError("y")
    except ValueError as e:
        t.log_error(e)
    t.shutdown()
