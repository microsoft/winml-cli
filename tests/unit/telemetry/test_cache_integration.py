# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""End-to-end tests proving the cache wiring delivers the design's
'no telemetry data is lost due to transient network issues' guarantee.

These tests exist specifically to lock in the integration contract that
slipped through Phase 2 review (encoding helpers landed but were never
wired into the exporter / Telemetry singleton). If the wiring ever
regresses, these tests fail loudly.
"""

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


def _make_log_data(body: str, attrs: dict) -> ReadableLogRecord:
    record = LogRecord(
        timestamp=1_712_678_400_000_000_000,
        body=body,
        attributes=attrs,
    )
    return ReadableLogRecord(
        log_record=record,
        resource=Resource.get_empty(),
        instrumentation_scope=InstrumentationScope("modelkit"),
    )


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "modelkit.cache"


@pytest.fixture
def cache(cache_path):
    return _PersistentCache(path=cache_path)


def test_network_failure_persists_envelopes_to_cache(cache, cache_path):
    """Process 1: net is down. The exporter must write the envelope to
    disk so process 2 can recover it."""
    exporter = OneCollectorLogExporter(
        ikey="o:abc",
        endpoint="https://example.invalid/",
        cache=cache,
    )
    ld = _make_log_data("ModelKitAction", {"action_name": "build", "success": True})

    with patch.object(exporter._session, "post", side_effect=requests.ConnectionError("net down")):
        result = exporter.export([ld])

    assert result == LogRecordExportResult.FAILURE
    assert cache_path.exists(), "exporter must have persisted the failed batch to cache"
    persisted = cache.drain()
    assert len(persisted) == 1
    assert persisted[0]["name"] == "ModelKitAction"


def test_next_process_flushes_cached_envelopes_on_first_export(cache, cache_path):
    """Process 2 (next CLI invocation): cache has leftover envelopes from
    a previous failed run. The first export() must drain and resend them."""
    # Seed the cache as if a prior process had failed.
    seeded = [
        {"name": "ModelKitHeartbeat", "iKey": "o:abc"},
        {"name": "ModelKitAction", "iKey": "o:abc", "data": {"a": 1}},
    ]
    cache.append(seeded)
    assert cache_path.exists()

    exporter = OneCollectorLogExporter(
        ikey="o:abc",
        endpoint="https://example.invalid/",
        cache=cache,
    )
    new_ld = _make_log_data("ModelKitError", {"exception_type": "ValueError"})

    mock_response = MagicMock(status_code=200)
    with patch.object(exporter._session, "post", return_value=mock_response) as post:
        result = exporter.export([new_ld])

    assert result == LogRecordExportResult.SUCCESS
    # Two POSTs: one for the cached batch, one for the new batch.
    assert post.call_count == 2
    assert not cache_path.exists(), "cache must be cleared after successful drain"

    # The first POST is the seeded cached envelopes.
    first_body = post.call_args_list[0].kwargs["data"]
    assert b"ModelKitHeartbeat" in first_body
    # The second POST is the live batch.
    second_body = post.call_args_list[1].kwargs["data"]
    assert b"ModelKitError" in second_body


def test_cache_flush_only_runs_on_first_export(cache):
    """The drain must not hammer the cache file on every export call —
    only the first one per process triggers a flush."""
    cache.append([{"name": "stale", "iKey": "o:abc"}])

    exporter = OneCollectorLogExporter(
        ikey="o:abc",
        endpoint="https://example.invalid/",
        cache=cache,
    )
    ld = _make_log_data("ModelKitAction", {"action_name": "build", "success": True})

    mock_response = MagicMock(status_code=200)
    with patch.object(exporter._session, "post", return_value=mock_response) as post:
        exporter.export([ld])  # flushes cache + sends current
        post.reset_mock()
        exporter.export([ld])  # cache already drained; only current sent

    assert post.call_count == 1


def test_cached_envelopes_re_persisted_on_recovery_failure(cache, cache_path):
    """If the recovery POST itself fails, the cached envelopes must go
    BACK into the cache so the next process can try again. Otherwise a
    cache flush attempt destroys the data."""
    seeded = [{"name": "ModelKitHeartbeat", "iKey": "o:abc"}]
    cache.append(seeded)

    exporter = OneCollectorLogExporter(
        ikey="o:abc",
        endpoint="https://example.invalid/",
        cache=cache,
    )
    ld = _make_log_data("ModelKitAction", {"action_name": "build", "success": True})

    with patch.object(
        exporter._session, "post", side_effect=requests.ConnectionError("still down")
    ):
        exporter.export([ld])

    # Both the cached envelope AND the new batch must be on disk now.
    persisted = cache.drain()
    names = [e["name"] for e in persisted]
    assert "ModelKitHeartbeat" in names, "cached envelope must survive recovery failure"
    assert "ModelKitAction" in names, "current batch must be persisted on POST failure"


def test_telemetry_disabled_clears_existing_cache(monkeypatch, tmp_path):
    """Regression: a session that resolves to disabled (e.g. the user
    just opted out via config.json) must delete any pending cache so
    we never resend events the user has since opted out of."""
    import winml.modelkit.telemetry._cache as cache_mod
    from winml.modelkit.telemetry import telemetry as telemetry_mod
    from winml.modelkit.telemetry.telemetry import Telemetry

    monkeypatch.setenv("MODELKIT_TELEMETRY_CACHE_DIR", str(tmp_path))
    cache_path = tmp_path / "modelkit.cache"

    # Seed a cache from a previous (enabled) session.
    cache_mod._PersistentCache().append([{"name": "stale", "iKey": "o:abc"}])
    assert cache_path.exists()

    # Now construct a Telemetry that resolves to disabled (empty iKey).
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "")
    telemetry_mod._INSTANCE = None
    t = Telemetry.get_or_init()
    assert t.disabled is True
    assert not cache_path.exists(), "disabled init must clear existing cache"
