# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# NOTE: The plan was written against opentelemetry-sdk 1.39.1 which exported
# LogData, LogExporter, and LogExportResult.  The installed SDK (1.41+) has
# renamed these to ReadableLogRecord, LogRecordExporter, and
# LogRecordExportResult (the old names are deprecated aliases).  Tests use the
# current names throughout.

import logging
import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from opentelemetry._logs import LogRecord
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExportResult
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope

from winml.modelkit.telemetry._cache import _PersistentCache
from winml.modelkit.telemetry.library import OneCollectorLogExporter


def _make_log_data(body: str, attrs: dict, resource: Resource | None = None) -> ReadableLogRecord:
    record = LogRecord(
        timestamp=1_712_678_400_000_000_000,  # ns
        body=body,
        attributes=attrs,
    )
    return ReadableLogRecord(
        log_record=record,
        resource=resource or Resource.get_empty(),
        instrumentation_scope=InstrumentationScope("modelkit"),
    )


@pytest.fixture
def exporter(tmp_path):
    # Use a tmp_path-scoped cache so the test never reads or writes the
    # real user-scoped persistent cache (which can be polluted by prior
    # modelkit runs on the dev machine and would inject extra POSTs into
    # the test).
    # Yield + shutdown so the underlying ``requests.Session`` (and its
    # connection pool) is closed at teardown rather than leaked across
    # tests.
    cache = _PersistentCache(path=tmp_path / "winmlcli.cache")
    exp = OneCollectorLogExporter(
        ikey="abc-def",
        endpoint="https://example.invalid/OneCollector/1.0/",
        cache=cache,
    )
    yield exp
    exp.shutdown()


@pytest.mark.parametrize(
    "ikey,endpoint",
    [
        ("", "https://example.invalid/"),
        ("abc-def", ""),
        ("", ""),
    ],
)
def test_constructor_rejects_empty_ikey_or_endpoint(ikey, endpoint):
    """Defense-in-depth: no accidental POST with an empty iKey, even if
    the Telemetry singleton's gating is bypassed (e.g. direct instantiation
    from a test or a future callsite)."""
    with pytest.raises(ValueError):
        OneCollectorLogExporter(ikey=ikey, endpoint=endpoint)


@pytest.mark.parametrize(
    "ikey",
    [
        "noseparator",  # no dash at all
        "-leading-dash",  # empty tenant_token portion
    ],
)
def test_constructor_rejects_malformed_ikey(ikey):
    """The full ikey must contain a non-empty tenant_token portion before
    the first '-', otherwise the envelope iKey can't be derived."""
    with pytest.raises(ValueError):
        OneCollectorLogExporter(ikey=ikey, endpoint="https://example.invalid/OneCollector/1.0/")


def test_export_success_returns_success(exporter):
    ld = _make_log_data(
        body="WinMLCLIAction",
        attrs={"action_name": "build", "success": True},
    )
    mock_response = MagicMock(status_code=200)
    with patch.object(exporter._session, "post", return_value=mock_response) as mock_post:
        result = exporter.export([ld])

    assert result == LogRecordExportResult.SUCCESS
    mock_post.assert_called_once()
    # OneCollector /OneCollector/1.0/ ingest only accepts x-json-stream
    # (NDJSON) or bond-compact-binary; application/json is rejected with
    # HTTP 415. Auth is via the x-apikey header (full ikey), and the
    # envelope iKey field carries the "o:<tenant_token>" form -- the two
    # values are intentionally different on the wire.
    headers = exporter._session.headers
    assert headers["Content-Type"] == "application/x-json-stream; charset=utf-8"
    assert headers["x-apikey"] == "abc-def"
    # Body is NDJSON (one envelope per line, no enclosing array).
    _, kwargs = mock_post.call_args
    body = kwargs["data"]
    assert not body.startswith(b"[")
    assert b'"WinMLCLIAction"' in body
    # Regression guard: envelope iKey is "o:<tenant_token>", NOT the full
    # ikey. Sending the full ikey here triggers
    # ``Collector-Error: Invalid Tenant Token`` from OneCollector.
    assert b'"iKey":"o:abc"' in body
    assert b'"iKey":"abc-def"' not in body


def test_export_connection_error_returns_failure(exporter):
    ld = _make_log_data("WinMLCLIHeartbeat", {})
    with patch.object(exporter._session, "post", side_effect=requests.ConnectionError("no route")):
        result = exporter.export([ld])
    assert result == LogRecordExportResult.FAILURE


def test_export_timeout_returns_failure(exporter):
    ld = _make_log_data("WinMLCLIHeartbeat", {})
    with patch.object(exporter._session, "post", side_effect=requests.Timeout("slow")):
        result = exporter.export([ld])
    assert result == LogRecordExportResult.FAILURE


def test_export_4xx_5xx_returns_failure(exporter):
    ld = _make_log_data("WinMLCLIHeartbeat", {})
    for status in (400, 500, 503):
        with patch.object(exporter._session, "post", return_value=MagicMock(status_code=status)):
            result = exporter.export([ld])
        assert result == LogRecordExportResult.FAILURE, f"status {status}"


def test_export_translates_resource_to_ext(exporter):
    # Resource carries the legacy `os.release` and `initTs` attributes that
    # an older version of the exporter mapped into the envelope. The
    # OneCollector CS 4.0 backend rejects any envelope that includes those
    # fields (they are not part of the documented `ext.os` / `ext.app`
    # slots) with `InvalidEventFormat: all`. The exporter must NOT translate
    # them, even if they appear in the Resource.
    resource = Resource.create(
        {
            "device_id": "hash-abc",
            "id_status": "EXISTING",
            "os.name": "Windows",
            "os.version": "10.0.26200",
            "os.release": "11",
            "os.arch": "AMD64",
            "app_version": "0.0.1",
            "app_instance_id": "sesid-xyz",
            "initTs": 1712678400.0,
        }
    )
    ld = _make_log_data("WinMLCLIHeartbeat", {}, resource=resource)

    captured_body = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured_body["data"] = data
        return MagicMock(status_code=200)

    with patch.object(exporter._session, "post", side_effect=fake_post):
        exporter.export([ld])

    import json

    # NDJSON: one envelope per line.
    lines = captured_body["data"].decode("utf-8").splitlines()
    envelopes = [json.loads(line) for line in lines if line]
    ext = envelopes[0]["ext"]
    assert ext["device"] == {
        "localId": "hash-abc",
        "authId": "EXISTING",
        "deviceClass": "AMD64",
    }
    assert ext["os"] == {"name": "Windows", "ver": "10.0.26200"}
    assert ext["app"] == {"ver": "0.0.1", "sesId": "sesid-xyz"}


def test_export_omits_undocumented_cs40_ext_fields(exporter):
    """Regression guard for https://github.com/microsoft/winml-cli/issues/635.

    `ext.os.release` and `ext.app.initTs` are not part of the documented
    CS 4.0 `os` / `app` extension slots. Including them causes the
    OneCollector backend to reject the entire batch with
    ``{"acc":0,"efi":{"InvalidEventFormat":"all"}}``. This test fails
    fast if anyone re-introduces either mapping in
    :func:`_resource_to_ext`.
    """
    resource = Resource.create(
        {
            "os.name": "Windows",
            "os.release": "11",
            "app_version": "0.0.1",
            "initTs": 1712678400.0,
        }
    )
    ld = _make_log_data("WinMLCLIHeartbeat", {}, resource=resource)

    captured = {}

    def fake_post(url, data=None, **_kw):
        captured["data"] = data
        return MagicMock(status_code=200)

    with patch.object(exporter._session, "post", side_effect=fake_post):
        exporter.export([ld])

    import json

    envelope = json.loads(captured["data"].splitlines()[0])
    assert "release" not in envelope["ext"].get("os", {})
    assert "initTs" not in envelope["ext"].get("app", {})


def test_export_serializes_multiple_envelopes_as_ndjson(exporter):
    """Two envelopes → two lines joined by \\n, no enclosing array."""
    a = _make_log_data("WinMLCLIHeartbeat", {})
    b = _make_log_data("WinMLCLIAction", {"action_name": "build", "success": True})

    captured = {}

    def fake_post(url, data=None, **_kw):
        captured["data"] = data
        return MagicMock(status_code=200)

    with patch.object(exporter._session, "post", side_effect=fake_post):
        exporter.export([a, b])

    body = captured["data"]
    assert not body.startswith(b"[")
    assert not body.endswith(b"]")
    lines = body.split(b"\n")
    assert len(lines) == 2
    # Each line is a standalone JSON document (no trailing comma, no brackets).
    import json

    json.loads(lines[0])
    json.loads(lines[1])


def test_shutdown_closes_session_and_blocks_further_exports(exporter):
    close_called = MagicMock()
    exporter._session.close = close_called

    exporter.shutdown()
    close_called.assert_called_once()

    # After shutdown, export is a no-op that returns SUCCESS
    ld = _make_log_data("WinMLCLIHeartbeat", {})
    with patch.object(exporter._session, "post") as mock_post:
        assert exporter.export([ld]) == LogRecordExportResult.SUCCESS
        mock_post.assert_not_called()


def test_export_empty_batch_is_noop(exporter):
    with patch.object(exporter._session, "post") as mock_post:
        assert exporter.export([]) == LogRecordExportResult.SUCCESS
        mock_post.assert_not_called()


# --- _resource_to_ext direct unit tests ---


from winml.modelkit.telemetry.library.exporter import _resource_to_ext  # noqa: E402


def test_resource_to_ext_none_returns_empty_dict():
    assert _resource_to_ext(None) == {}


def test_resource_to_ext_empty_resource_returns_empty_dict():
    assert _resource_to_ext(Resource.create({})) == {}


def test_resource_to_ext_partial_only_populates_present_slots():
    # Only an os.name is present → only the "os" sub-dict should appear.
    ext = _resource_to_ext(Resource.create({"os.name": "Windows"}))
    assert ext == {"os": {"name": "Windows"}}


def test_resource_to_ext_unknown_attributes_are_ignored():
    # Attributes outside the mapping table do not leak into ext.
    ext = _resource_to_ext(Resource.create({"os.name": "Windows", "custom.attr": "x"}))
    assert ext == {"os": {"name": "Windows"}}


# --- _parse_kill_tokens direct unit tests ---


from winml.modelkit.telemetry.library.exporter import _parse_kill_tokens  # noqa: E402


@pytest.mark.parametrize(
    "header,expected",
    [
        ("", set()),
        ("o:abc:all", {"abc"}),
        ("o:abc", {"abc"}),  # no reason suffix
        ("o:abc:all,o:def:event_x", {"abc", "def"}),
        ("  o:abc:all  ,  o:def  ", {"abc", "def"}),  # whitespace tolerant
        ("garbage,o:abc:all", {"abc"}),  # entries without "o:" prefix are skipped
        ("o::all", set()),  # empty tenant_token portion is rejected
    ],
)
def test_parse_kill_tokens(header, expected):
    assert _parse_kill_tokens(header) == expected


# --- kill-tokens / DEBUG-log behavior on failed POSTs ---


def _killed_response(tenant: str = "abc", duration: int = 86_400):
    """Build a 401 response that mimics OneCollector's tenant-killed reply."""
    resp = MagicMock(status_code=401)
    resp.headers = {
        "Collector-Error": "Invalid Tenant Token.",
        "kill-tokens": f"o:{tenant}:all",
        "kill-duration": str(duration),
    }
    resp.text = '{"acc":0,"efi":{"InvalidTenantToken":"all"}}'
    return resp


def test_kill_tokens_recorded_on_failure(exporter):
    """On a 401 with kill-tokens for our tenant, exporter records the
    kill window and ``_is_killed()`` returns True until it expires."""
    ld = _make_log_data("WinMLCLIHeartbeat", {})
    assert exporter._is_killed() is False

    with patch.object(exporter._session, "post", return_value=_killed_response()):
        exporter.export([ld])

    assert exporter._is_killed() is True
    assert exporter._killed_until is not None
    assert exporter._killed_until > time.time()


def test_kill_tokens_for_other_tenant_is_ignored(exporter):
    """If kill-tokens names a different tenant, our exporter is unaffected."""
    ld = _make_log_data("WinMLCLIHeartbeat", {})
    other = _killed_response(tenant="not-our-tenant")
    with patch.object(exporter._session, "post", return_value=other):
        exporter.export([ld])
    assert exporter._is_killed() is False


def test_export_skipped_during_kill_window(exporter):
    """While killed, export() is a no-op that returns SUCCESS without
    even touching the HTTP session."""
    ld = _make_log_data("WinMLCLIHeartbeat", {})

    # First export: triggers the kill window.
    with patch.object(exporter._session, "post", return_value=_killed_response()) as p1:
        exporter.export([ld])
    assert p1.call_count == 1
    assert exporter._is_killed()

    # Second export: must not POST at all.
    with patch.object(exporter._session, "post") as p2:
        result = exporter.export([ld])
    assert result == LogRecordExportResult.SUCCESS
    p2.assert_not_called()


def test_kill_drops_envelopes_instead_of_caching(exporter, tmp_path):
    """A failed POST that triggered a kill must NOT enqueue the batch in
    the persistent cache — caching it just guarantees forever-failure on
    future startups within the kill window."""
    ld = _make_log_data("WinMLCLIHeartbeat", {})
    cache_path = tmp_path / "winmlcli.cache"
    with patch.object(exporter._session, "post", return_value=_killed_response()):
        exporter.export([ld])
    assert exporter._is_killed()
    assert not cache_path.exists(), "kill-induced failures must not persist to cache"


def test_cache_flush_kill_skips_new_batch_post(tmp_path):
    """If the cache-flush POST triggers a kill, the new-batch POST in
    the same export() call must be skipped — otherwise we waste a
    network round-trip just to re-confirm the kill, and the new
    envelopes would either be dropped or wrongly re-cached."""
    cache_path = tmp_path / "winmlcli.cache"
    cache = _PersistentCache(path=cache_path)
    cache.append([{"name": "WinMLCLIHeartbeat", "iKey": "o:abc"}])

    exp = OneCollectorLogExporter(
        ikey="abc-def",
        endpoint="https://example.invalid/OneCollector/1.0/",
        cache=cache,
    )
    try:
        ld = _make_log_data("WinMLCLIHeartbeat", {})
        with patch.object(exp._session, "post", return_value=_killed_response()) as p:
            result = exp.export([ld])

        # Exactly one POST: the cache flush. The new-batch POST is
        # short-circuited by the mid-export kill check.
        assert p.call_count == 1
        assert exp._is_killed()
        # Returned SUCCESS so BatchLogRecordProcessor doesn't re-queue.
        assert result == LogRecordExportResult.SUCCESS
    finally:
        exp.shutdown()


def test_kill_window_expiry_re_enables_post(exporter):
    """Past the kill window, export() resumes normal POST behavior."""
    ld = _make_log_data("WinMLCLIHeartbeat", {})

    # Use a short window so we can fast-forward past it via monkeypatching.
    short_kill = _killed_response(duration=10)
    with patch.object(exporter._session, "post", return_value=short_kill):
        exporter.export([ld])
    assert exporter._is_killed()

    # Fast-forward: pretend we're 11s past the kill window.
    fake_now = (exporter._killed_until or 0) + 1
    with (
        patch("winml.modelkit.telemetry.library.exporter.time.time", return_value=fake_now),
        patch.object(exporter._session, "post", return_value=MagicMock(status_code=200)) as p,
    ):
        result = exporter.export([ld])

    assert result == LogRecordExportResult.SUCCESS
    p.assert_called_once()


@pytest.mark.parametrize(
    "kill_duration_value",
    [
        None,  # header absent entirely
        "",  # header present but empty
        "0",  # non-positive
        "abc",  # non-numeric
    ],
)
def test_kill_tokens_with_unusable_duration_is_ignored(exporter, kill_duration_value):
    """``kill-tokens`` is meaningless without a positive integer
    ``kill-duration``. Any of: absent, empty, non-positive, or non-numeric
    must leave the exporter unkilled."""
    ld = _make_log_data("WinMLCLIHeartbeat", {})
    resp = MagicMock(status_code=401)
    headers = {"kill-tokens": "o:abc:all"}
    if kill_duration_value is not None:
        headers["kill-duration"] = kill_duration_value
    resp.headers = headers
    resp.text = ""
    with patch.object(exporter._session, "post", return_value=resp):
        exporter.export([ld])
    assert exporter._is_killed() is False


def test_post_failure_logs_collector_error_and_body_excerpt(exporter, caplog):
    """The DEBUG log on non-2xx must capture both the ``Collector-Error``
    header and a body excerpt — the two pieces OneCollector uses to
    communicate the actual rejection reason. Without these in the log,
    diagnosing tenant/format misconfigurations requires a live probe."""
    ld = _make_log_data("WinMLCLIHeartbeat", {})
    resp = MagicMock(status_code=401)
    resp.headers = {"Collector-Error": "Invalid Tenant Token."}
    resp.text = '{"acc":0,"rej":1,"efi":{"InvalidTenantToken":[0]}}'

    caplog.set_level(logging.DEBUG, logger="winml.modelkit.telemetry.library.exporter")
    with patch.object(exporter._session, "post", return_value=resp):
        exporter.export([ld])

    backend_logs = [
        r.getMessage() for r in caplog.records if "telemetry backend returned" in r.getMessage()
    ]
    assert backend_logs, "expected a DEBUG log line for the 401"
    msg = backend_logs[0]
    assert "401" in msg
    assert "Invalid Tenant Token." in msg
    assert "InvalidTenantToken" in msg


def test_post_failure_log_truncates_long_body(exporter, caplog):
    """A backend that returns a huge body shouldn't flood the DEBUG log."""
    ld = _make_log_data("WinMLCLIHeartbeat", {})
    resp = MagicMock(status_code=500)
    resp.headers = {}
    resp.text = "x" * 10_000

    caplog.set_level(logging.DEBUG, logger="winml.modelkit.telemetry.library.exporter")
    with patch.object(exporter._session, "post", return_value=resp):
        exporter.export([ld])

    msg = next(
        r.getMessage() for r in caplog.records if "telemetry backend returned" in r.getMessage()
    )
    # The truncation cap is _RESPONSE_BODY_LOG_LIMIT (200) bytes.
    assert "x" * 200 in msg
    assert "x" * 1_000 not in msg
