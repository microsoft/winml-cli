# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from datetime import datetime, timezone

import pytest

from winml.modelkit.telemetry.library.serialization import _build_envelope, _serialize_batch


def test_build_envelope_basic_shape():
    ts = datetime(2026, 4, 17, 10, 30, 0, 123456, tzinfo=timezone.utc)
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


def test_serialize_batch_emits_ndjson():
    ts = datetime(2026, 4, 17, 10, 30, 0, 0, tzinfo=timezone.utc)
    envelopes = [
        _build_envelope("ModelKitHeartbeat", "o:key", ts, {}, {}),
        _build_envelope("ModelKitAction", "o:key", ts, {"success": True}, {}),
    ]
    body = _serialize_batch(envelopes)
    # NDJSON: one envelope per line, no enclosing array.
    assert not body.startswith(b"[")
    assert not body.endswith(b"]")
    lines = body.split(b"\n")
    assert len(lines) == 2
    # Each line is a standalone JSON document.
    import json

    json.loads(lines[0])
    json.loads(lines[1])
    # Both events present
    assert b'"ModelKitHeartbeat"' in lines[0]
    assert b'"ModelKitAction"' in lines[1]
    # UTF-8 encoded
    assert isinstance(body, bytes)


def test_serialize_batch_preserves_unicode():
    ts = datetime(2026, 4, 17, 10, 30, 0, 0, tzinfo=timezone.utc)
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
    ts = datetime(2026, 4, 17, 10, 30, 0, microsecond, tzinfo=timezone.utc)
    envelope = _build_envelope("X", "o:k", ts, {}, {})
    assert envelope["time"] == f"2026-04-17T10:30:00.{expected_ms}Z"
