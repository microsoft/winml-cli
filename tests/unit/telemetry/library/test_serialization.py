# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from datetime import UTC, datetime

import pytest

from winml.modelkit.telemetry.library.serialization import (
    _build_envelope,
    _envelope_ikey,
    _serialize_batch,
)


def test_build_envelope_basic_shape():
    ts = datetime(2026, 4, 17, 10, 30, 0, 123456, tzinfo=UTC)
    envelope = _build_envelope(
        name="WinMLCLIAction",
        ikey="o:abc-def",
        timestamp=ts,
        data={"action_name": "build", "success": True},
        ext={"app": {"ver": "0.0.1"}},
    )
    assert envelope["ver"] == "4.0"
    assert envelope["name"] == "WinMLCLIAction"
    assert envelope["iKey"] == "o:abc-def"
    assert envelope["data"] == {"action_name": "build", "success": True}
    assert envelope["ext"] == {"app": {"ver": "0.0.1"}}
    # ISO8601 with millisecond precision, trailing Z for UTC
    assert envelope["time"] == "2026-04-17T10:30:00.123Z"


def test_serialize_batch_emits_ndjson():
    ts = datetime(2026, 4, 17, 10, 30, 0, 0, tzinfo=UTC)
    envelopes = [
        _build_envelope("WinMLCLIHeartbeat", "o:key", ts, {}, {}),
        _build_envelope("WinMLCLIAction", "o:key", ts, {"success": True}, {}),
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
    assert b'"WinMLCLIHeartbeat"' in lines[0]
    assert b'"WinMLCLIAction"' in lines[1]
    # UTF-8 encoded
    assert isinstance(body, bytes)


def test_serialize_batch_preserves_unicode():
    ts = datetime(2026, 4, 17, 10, 30, 0, 0, tzinfo=UTC)
    envelope = _build_envelope("WinMLCLIAction", "o:key", ts, {"note": "café λ"}, {})
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


@pytest.mark.parametrize(
    "full_ikey,expected",
    [
        # Realistic OneCollector iKey shape: <32hex>-<guid>-<ingestion_token>.
        (
            "abc123abc123abc123abc123abc12345-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee-1234",
            "o:abc123abc123abc123abc123abc12345",
        ),
        # Minimal valid form: anything non-empty before the first dash.
        ("abc-def", "o:abc"),
        ("token-rest-of-key", "o:token"),
    ],
)
def test_envelope_ikey_extracts_tenant_token_and_prefixes(full_ikey, expected):
    """The envelope iKey is ``o:<part-before-first-dash>``; the suffix
    (ingestion token + GUID) only goes in the ``x-apikey`` header."""
    assert _envelope_ikey(full_ikey) == expected


@pytest.mark.parametrize(
    "bad_ikey",
    [
        "noseparator",  # no dash at all
        "-leading-dash",  # empty tenant_token portion
        "",  # empty (defense in depth; exporter rejects this earlier)
    ],
)
def test_envelope_ikey_rejects_malformed(bad_ikey):
    with pytest.raises(ValueError, match="tenant_token"):
        _envelope_ikey(bad_ikey)
