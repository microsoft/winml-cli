# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# NOTE: The plan was written against opentelemetry-sdk 1.39.1 which exported
# LogData, LogExporter, and LogExportResult.  The installed SDK (1.41+) has
# renamed these to ReadableLogRecord, LogRecordExporter, and
# LogRecordExportResult (the old names are deprecated aliases).  Tests use the
# current names throughout.

from unittest.mock import MagicMock, patch

import pytest
import requests
from opentelemetry._logs import LogRecord
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExportResult
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope

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
def exporter():
    return OneCollectorLogExporter(
        ikey="o:abc",
        endpoint="https://example.invalid/OneCollector/1.0/",
    )


@pytest.mark.parametrize(
    "ikey,endpoint",
    [
        ("", "https://example.invalid/"),
        ("o:abc", ""),
        ("", ""),
    ],
)
def test_constructor_rejects_empty_ikey_or_endpoint(ikey, endpoint):
    """Defense-in-depth: no accidental POST with an empty iKey, even if
    the Telemetry singleton's gating is bypassed (e.g. direct instantiation
    from a test or a future callsite)."""
    with pytest.raises(ValueError):
        OneCollectorLogExporter(ikey=ikey, endpoint=endpoint)


def test_export_success_returns_success(exporter):
    ld = _make_log_data(
        body="ModelKitAction",
        attrs={"action_name": "build", "success": True},
    )
    mock_response = MagicMock(status_code=200)
    with patch.object(exporter._session, "post", return_value=mock_response) as mock_post:
        result = exporter.export([ld])

    assert result == LogRecordExportResult.SUCCESS
    mock_post.assert_called_once()
    # Content-Type is set on the session (applied to every request).
    assert exporter._session.headers["Content-Type"].startswith("application/json")
    # Body is JSON array containing the event
    _, kwargs = mock_post.call_args
    body = kwargs["data"]
    assert b'"ModelKitAction"' in body
    assert b'"iKey":"o:abc"' in body


def test_export_connection_error_returns_failure(exporter):
    ld = _make_log_data("ModelKitHeartbeat", {})
    with patch.object(exporter._session, "post", side_effect=requests.ConnectionError("no route")):
        result = exporter.export([ld])
    assert result == LogRecordExportResult.FAILURE


def test_export_timeout_returns_failure(exporter):
    ld = _make_log_data("ModelKitHeartbeat", {})
    with patch.object(exporter._session, "post", side_effect=requests.Timeout("slow")):
        result = exporter.export([ld])
    assert result == LogRecordExportResult.FAILURE


def test_export_4xx_5xx_returns_failure(exporter):
    ld = _make_log_data("ModelKitHeartbeat", {})
    for status in (400, 500, 503):
        with patch.object(exporter._session, "post", return_value=MagicMock(status_code=status)):
            result = exporter.export([ld])
        assert result == LogRecordExportResult.FAILURE, f"status {status}"


def test_export_translates_resource_to_ext(exporter):
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
    ld = _make_log_data("ModelKitHeartbeat", {}, resource=resource)

    captured_body = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured_body["data"] = data
        return MagicMock(status_code=200)

    with patch.object(exporter._session, "post", side_effect=fake_post):
        exporter.export([ld])

    import json

    envelopes = json.loads(captured_body["data"].decode("utf-8"))
    ext = envelopes[0]["ext"]
    assert ext["device"] == {
        "localId": "hash-abc",
        "authId": "EXISTING",
        "deviceClass": "AMD64",
    }
    assert ext["os"] == {"name": "Windows", "ver": "10.0.26200", "release": "11"}
    assert ext["app"] == {
        "ver": "0.0.1",
        "sesId": "sesid-xyz",
        "initTs": 1712678400.0,
    }


def test_shutdown_closes_session_and_blocks_further_exports(exporter):
    close_called = MagicMock()
    exporter._session.close = close_called

    exporter.shutdown()
    close_called.assert_called_once()

    # After shutdown, export is a no-op that returns SUCCESS
    ld = _make_log_data("ModelKitHeartbeat", {})
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
