# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from unittest.mock import MagicMock

from winml.modelkit.telemetry import Telemetry


# `_reset_singleton` (autouse), `enabled_telemetry`, `running_telemetry`
# come from tests/unit/telemetry/conftest.py.


def _with_mock_provider(t: Telemetry) -> MagicMock:
    """Replace ``t._provider`` with a ``MagicMock`` and return it."""
    t._provider = MagicMock()
    return t._provider


def test_shutdown_flushes_provider(running_telemetry):
    provider = _with_mock_provider(running_telemetry)
    running_telemetry.shutdown()
    provider.shutdown.assert_called_once()


def test_shutdown_clears_logger_for_post_shutdown_noop(running_telemetry):
    provider = _with_mock_provider(running_telemetry)
    running_telemetry.shutdown()
    # After shutdown the instance must be in a disabled state.
    assert running_telemetry.disabled is True
    assert running_telemetry._logger is None
    # And the provider reference is dropped so we can't double-shutdown the
    # same object by accident.
    assert running_telemetry._provider is None
    # Further emits are no-ops (no crash, no provider interaction).
    running_telemetry.log_heartbeat()
    provider.shutdown.assert_called_once()  # still just one call


def test_shutdown_is_idempotent(running_telemetry):
    provider = _with_mock_provider(running_telemetry)
    running_telemetry.shutdown()
    running_telemetry.shutdown()  # must not raise
    assert provider.shutdown.call_count == 1


def test_shutdown_on_disabled_is_noop(monkeypatch):
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "")
    t = Telemetry.get_or_init()
    assert t.disabled is True
    t.shutdown()  # no crash; provider is None to begin with


def test_public_methods_never_raise_even_if_logger_broken(running_telemetry):
    """Regression: a telemetry subsystem failure must never take down the
    CLI. If the logger / provider object starts raising, the public
    methods continue to return normally."""
    running_telemetry._logger = MagicMock()
    running_telemetry._logger.emit.side_effect = RuntimeError("downstream blew up")

    # None of these should raise.
    running_telemetry.log_heartbeat()
    running_telemetry.log_action(
        action_name="x",
        device=None,
        ep=None,
        duration_ms=0,
        success=True,
    )
    try:
        raise ValueError("y")
    except ValueError as e:
        running_telemetry.log_error(e)
    running_telemetry.shutdown()
