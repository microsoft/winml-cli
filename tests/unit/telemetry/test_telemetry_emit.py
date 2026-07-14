# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from unittest.mock import MagicMock

from winml.modelkit.telemetry import Telemetry


# `_reset_telemetry_singleton` (autouse) comes from tests/conftest.py.
# `enabled_telemetry`, `running_telemetry`, `isolated_config`, and
# `clean_env` come from tests/unit/telemetry/conftest.py.


def _with_mock_logger(t: Telemetry) -> MagicMock:
    """Replace ``t._logger`` with a ``MagicMock`` and return it."""
    t._logger = MagicMock()
    return t._logger


def test_log_heartbeat_emits_event_with_no_data(running_telemetry):
    logger = _with_mock_logger(running_telemetry)
    running_telemetry.log_heartbeat()
    logger.emit.assert_called_once()
    log_record = logger.emit.call_args.args[0]
    assert str(log_record.body) == "WinMLCLIHeartbeat"
    assert dict(log_record.attributes) == {}


def test_log_action_emits_with_whitelisted_attrs(running_telemetry):
    logger = _with_mock_logger(running_telemetry)
    running_telemetry.log_action(
        action_name="build",
        device="NPU",
        ep="QNNExecutionProvider",
        duration_ms=1234,
        success=True,
    )
    logger.emit.assert_called_once()
    log_record = logger.emit.call_args.args[0]
    assert str(log_record.body) == "WinMLCLIAction"
    attrs = dict(log_record.attributes)
    assert attrs["action_name"] == "build"
    assert attrs["device"] == "NPU"
    assert attrs["ep"] == "QNNExecutionProvider"
    assert attrs["duration_ms"] == 1234
    assert attrs["success"] is True
    assert attrs["invoked_from"] in ("Script", "Interactive")


def test_log_action_drops_unknown_attrs(running_telemetry):
    """Regression: attributes outside the whitelist are silently dropped,
    never sent."""
    logger = _with_mock_logger(running_telemetry)
    running_telemetry.log_action(
        action_name="build",
        device=None,
        ep=None,
        duration_ms=10,
        success=True,
        # Extra kwarg NOT in the whitelist - should be dropped.
        leaked_field=r"C:\Users\Alice\somewhere",
    )
    log_record = logger.emit.call_args.args[0]
    attrs = dict(log_record.attributes)
    assert "leaked_field" not in attrs


def test_log_error_scrubs_message_and_extracts_stack(running_telemetry):
    logger = _with_mock_logger(running_telemetry)
    try:
        raise ValueError(
            "failed on alice@example.com at 10.0.0.1 for GUID 12345678-1234-5678-1234-567812345678"
        )
    except ValueError as exc:
        running_telemetry.log_error(exc)

    logger.emit.assert_called_once()
    log_record = logger.emit.call_args.args[0]
    assert str(log_record.body) == "WinMLCLIError"
    attrs = dict(log_record.attributes)
    assert attrs["exception_type"] == "ValueError"
    # Message scrubbed: no email, no IP, no GUID
    assert "alice@example.com" not in attrs["exception_message"]
    assert "10.0.0.1" not in attrs["exception_message"]
    assert "12345678-1234-5678-1234-567812345678" not in attrs["exception_message"]
    assert "<scrubbed>" in attrs["exception_message"]
    # Stack is a list of {file, line, function}
    stack = attrs["exception_stack"]
    assert isinstance(stack, list)
    assert stack  # at least one frame
    for frame in stack:
        assert set(frame.keys()) == {"file", "line", "function"}


def test_log_error_records_root_cause_for_chained(running_telemetry):
    logger = _with_mock_logger(running_telemetry)
    try:
        try:
            raise ValueError("qwen3_5 not recognized")
        except ValueError as e:
            raise RuntimeError("Inspection error") from e
    except RuntimeError as outer:
        running_telemetry.log_error(outer)

    log_record = logger.emit.call_args.args[0]
    attrs = dict(log_record.attributes)
    assert attrs["exception_type"] == "RuntimeError"
    assert attrs["root_cause_type"] == "ValueError"
    assert "qwen3_5 not recognized" in attrs["root_cause_message"]


def test_log_error_unchained_has_null_root_cause(running_telemetry):
    logger = _with_mock_logger(running_telemetry)
    try:
        raise RuntimeError("Quantization failed")
    except RuntimeError as exc:
        running_telemetry.log_error(exc)

    log_record = logger.emit.call_args.args[0]
    attrs = dict(log_record.attributes)
    assert attrs["exception_type"] == "RuntimeError"
    assert attrs["root_cause_type"] is None
    assert attrs["root_cause_message"] is None


def test_log_error_scrubs_root_cause_message(running_telemetry):
    logger = _with_mock_logger(running_telemetry)
    try:
        try:
            raise ValueError("leaked alice@example.com in root")
        except ValueError as e:
            raise RuntimeError("wrapper") from e
    except RuntimeError as outer:
        running_telemetry.log_error(outer)

    log_record = logger.emit.call_args.args[0]
    attrs = dict(log_record.attributes)
    assert "alice@example.com" not in attrs["root_cause_message"]
    assert "<scrubbed>" in attrs["root_cause_message"]


def test_disabled_telemetry_noops_all_emits(monkeypatch):
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "")
    t = Telemetry.get_or_init()
    assert t.disabled is True
    # These must not raise and must not hit a logger (there is none).
    t.log_heartbeat()
    t.log_action(
        action_name="build",
        device=None,
        ep=None,
        duration_ms=0,
        success=True,
    )
    try:
        raise RuntimeError("x")
    except RuntimeError as e:
        t.log_error(e)
