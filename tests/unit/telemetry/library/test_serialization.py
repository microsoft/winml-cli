# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from datetime import UTC, datetime

import pytest

from winml.modelkit.telemetry.library.serialization import _build_envelope, _serialize_batch


def test_build_envelope_basic_shape():
    ts = datetime(2026, 4, 17, 10, 30, 0, 123456, tzinfo=UTC)
    envelope = _build_envelope(
        name="ModelKitAction",
        ikey="o:abc-def",
        timestamp=ts,
        data={"action_name": "build", "success": True},
        ext={"app": {"ver": "0.0.1"}},
    )
    assert envelope["ver"] == "4.0"
    assert envelope["name"] == "ModelKitAction"
    assert envelope["iKey"] == "o:abc-def"
    assert envelope["data"] == {"action_name": "build", "success": True}
    assert envelope["ext"] == {"app": {"ver": "0.0.1"}}
    # ISO8601 with millisecond precision, trailing Z for UTC
    assert envelope["time"] == "2026-04-17T10:30:00.123Z"


def test_serialize_batch_emits_json_array():
    ts = datetime(2026, 4, 17, 10, 30, 0, 0, tzinfo=UTC)
    envelopes = [
        _build_envelope("ModelKitHeartbeat", "o:key", ts, {}, {}),
        _build_envelope("ModelKitAction", "o:key", ts, {"success": True}, {}),
    ]
    body = _serialize_batch(envelopes)
    # Compact JSON, no whitespace
    assert body.startswith(b"[")
    assert body.endswith(b"]")
    # Both events present
    assert b'"ModelKitHeartbeat"' in body
    assert b'"ModelKitAction"' in body
    # UTF-8 encoded
    assert isinstance(body, bytes)


def test_serialize_batch_preserves_unicode():
    ts = datetime(2026, 4, 17, 10, 30, 0, 0, tzinfo=UTC)
    envelope = _build_envelope("ModelKitAction", "o:key", ts, {"note": "café λ"}, {})
    body = _serialize_batch([envelope])
    # ensure_ascii=False keeps unicode readable
    assert "café".encode() in body
    assert "λ".encode() in body


@pytest.mark.parametrize(
    "microsecond,expected_ms",
    [
        (0, "000"),
        (1000, "001"),
        (999999, "999"),
        (500000, "500"),
    ],
)
def test_timestamp_millisecond_precision(microsecond, expected_ms):
    ts = datetime(2026, 4, 17, 10, 30, 0, microsecond, tzinfo=UTC)
    envelope = _build_envelope("X", "o:k", ts, {}, {})
    assert envelope["time"] == f"2026-04-17T10:30:00.{expected_ms}Z"
