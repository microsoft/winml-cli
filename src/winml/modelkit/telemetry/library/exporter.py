# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""OneCollector log exporter: serializes OTel log records as CS 4.0 envelopes and POSTs them."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import requests
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult

from .._cache import _PersistentCache
from .serialization import _build_envelope, _envelope_ikey, _serialize_batch


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.sdk._logs import ReadableLogRecord

_LOGGER = logging.getLogger(__name__)

_HTTP_TIMEOUT = 10.0
# Truncate response body excerpts in DEBUG logs so a backend that decides
# to dump a large diagnostic payload doesn't flood the log.
_RESPONSE_BODY_LOG_LIMIT = 200


class OneCollectorLogExporter(LogRecordExporter):
    """Post Common Schema 4.0 event envelopes to the OneCollector endpoint."""

    def __init__(
        self,
        ikey: str,
        endpoint: str,
        cache: _PersistentCache | None = None,
    ) -> None:
        # Fail loudly rather than silently POST ``{"iKey": ""}`` to the
        # endpoint. In dev installs ``constants.INSTRUMENTATION_KEY`` is
        # empty; the Telemetry singleton guards against that, and this
        # second guard keeps the invariant a property of the library
        # itself (defense in depth).
        if not ikey:
            raise ValueError("ikey must be non-empty")
        if not endpoint:
            raise ValueError("endpoint must be non-empty")
        self._ikey = ikey
        # OneCollector requires the envelope's iKey field to be
        # ``o:<tenant_token>`` (just the prefix portion of the full ikey),
        # while the x-apikey HTTP header carries the full ikey. Compute
        # the envelope form once and cache it; a malformed ikey raises
        # ValueError, which Telemetry._try_init catches to disable
        # telemetry rather than crash the CLI.
        self._envelope_ikey = _envelope_ikey(ikey)
        # Bare tenant_token (no "o:" prefix) is what the ``kill-tokens``
        # response header lists, so we keep it around for membership checks.
        self._tenant_token = self._envelope_ikey[2:]
        self._endpoint = endpoint
        self._cache = cache if cache is not None else _PersistentCache()
        # First export() flushes the cache before sending the new batch;
        # subsequent exports go straight through.
        self._cache_flushed = False
        # OneCollector's ``kill-tokens`` directive: when set, our tenant
        # is on the backend's deny list and we must stop sending until
        # this epoch second. ``None`` means not killed.
        self._killed_until: float | None = None
        # _shutdown is read on the BatchLogRecordProcessor export thread and
        # written on the shutdown thread; bool assignment is atomic under the
        # CPython GIL, so no lock is needed.
        self._shutdown = False
        # Close the session if post-creation setup fails, so we never leak a
        # Session object (and its connection pool) when init raises.
        session = requests.Session()
        try:
            # The /OneCollector/1.0/ ingest only accepts x-json-stream
            # (NDJSON) or bond-compact-binary; application/json is rejected
            # with HTTP 415. Auth is via the x-apikey header — embedding the
            # iKey only inside each envelope is not equivalent.
            session.headers.update(
                {
                    "Content-Type": "application/x-json-stream; charset=utf-8",
                    "x-apikey": ikey,
                }
            )
        except Exception:
            session.close()
            raise
        self._session = session

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        """Serialize *batch* and POST to the configured OneCollector endpoint.

        Recovers any envelopes cached from prior failed runs on the first
        call. Persists the current batch to the cache on POST failure so
        the next process can retry — :class:`BatchLogRecordProcessor` only
        re-queues in memory and loses the queue on process exit.

        While the tenant is under ``kill-tokens``, this is a no-op that
        returns ``SUCCESS`` to keep the BatchLogRecordProcessor from
        re-queueing events the backend has explicitly told us to stop
        sending; envelopes for that window are dropped, not cached.
        """
        if self._shutdown or not batch:
            return LogRecordExportResult.SUCCESS

        if self._is_killed():
            _LOGGER.debug(
                "telemetry export skipped: tenant under kill-tokens for %.0fs more",
                (self._killed_until or 0) - time.time(),
            )
            return LogRecordExportResult.SUCCESS

        try:
            envelopes = [self._to_envelope(ld) for ld in batch]
        except Exception:
            _LOGGER.debug("telemetry serialization failed", exc_info=True)
            return LogRecordExportResult.FAILURE

        # First-call cache flush: try to send anything left over from a
        # previous run. Best-effort, single shot — don't loop if the
        # backend is still down. If the failure is because we just got
        # killed, drop the cached batch instead of looping it forever.
        if not self._cache_flushed:
            self._cache_flushed = True
            cached = self._cache.drain()
            if cached and not self._post_envelopes(cached) and not self._is_killed():
                self._cache.append(cached)

        if not self._post_envelopes(envelopes):
            if not self._is_killed():
                self._cache.append(envelopes)
            return LogRecordExportResult.FAILURE
        return LogRecordExportResult.SUCCESS

    def _post_envelopes(self, envelopes: list[dict]) -> bool:
        """POST a list of envelopes; return True on 2xx, False otherwise.

        On non-2xx, parses ``kill-tokens`` / ``kill-duration`` to honor the
        backend's tenant-level backoff, and emits a DEBUG log line that
        captures the ``Collector-Error`` header and a body excerpt — the
        two pieces of info the OneCollector backend uses to communicate
        the actual rejection reason.
        """
        if not envelopes:
            return True
        try:
            body = _serialize_batch(envelopes)
        except Exception:
            _LOGGER.debug("telemetry serialization failed", exc_info=True)
            return False
        try:
            response = self._session.post(
                self._endpoint,
                data=body,
                timeout=_HTTP_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout):
            _LOGGER.debug("telemetry network failure", exc_info=True)
            return False
        if 200 <= response.status_code < 300:
            return True
        self._record_kill_if_present(response)
        _LOGGER.debug(
            "telemetry backend returned %s: error=%r body=%s",
            response.status_code,
            response.headers.get("Collector-Error"),
            (response.text or "")[:_RESPONSE_BODY_LOG_LIMIT].replace("\n", " "),
        )
        return False

    def _is_killed(self) -> bool:
        """True iff our tenant is currently under a ``kill-tokens`` window."""
        return self._killed_until is not None and time.time() < self._killed_until

    def _record_kill_if_present(self, response: requests.Response) -> None:
        """Honor an inbound ``kill-tokens`` directive that names our tenant.

        No-op for any other 4xx/5xx response.
        """
        kill_header = response.headers.get("kill-tokens")
        duration_header = response.headers.get("kill-duration")
        if not kill_header or not duration_header:
            return
        try:
            duration_s = int(duration_header)
        except ValueError:
            return
        if duration_s <= 0:
            return
        if self._tenant_token not in _parse_kill_tokens(kill_header):
            return
        self._killed_until = time.time() + duration_s
        _LOGGER.debug(
            "telemetry tenant under kill-tokens for %ss (until epoch %.0f)",
            duration_s,
            self._killed_until,
        )

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """No-op: all exports are synchronous."""
        return True

    def shutdown(self) -> None:
        """Mark exporter as shut down and close the underlying HTTP session."""
        self._shutdown = True
        try:
            self._session.close()
        except Exception:
            _LOGGER.debug("session close failed", exc_info=True)

    # --- internal ---

    def _to_envelope(self, ld: ReadableLogRecord) -> dict:
        record = ld.log_record
        timestamp = _ns_to_datetime(record.timestamp)
        data = dict(record.attributes or {})
        ext = _resource_to_ext(ld.resource)
        return _build_envelope(
            name=str(record.body),
            ikey=self._envelope_ikey,
            timestamp=timestamp,
            data=data,
            ext=ext,
        )


def _parse_kill_tokens(header_value: str) -> set[str]:
    """Parse the OneCollector ``kill-tokens`` header into a set of tenant_token strings.

    The header is a comma-separated list. Each entry is in the form
    ``o:<tenant_token>`` optionally followed by ``:<reason>`` (e.g.
    ``o:abc:all`` or ``o:abc:event_name``). We treat any entry naming
    a tenant as a full kill for that tenant — per-event kills aren't
    something we exploit today.
    """
    if not header_value:
        return set()
    tokens: set[str] = set()
    for raw in header_value.split(","):
        entry = raw.strip()
        if not entry.startswith("o:"):
            continue
        rest = entry[2:]
        # Strip optional ":<reason>" suffix.
        tenant = rest.split(":", 1)[0]
        if tenant:
            tokens.add(tenant)
    return tokens


def _ns_to_datetime(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)


def _resource_to_ext(resource) -> dict:
    """Translate OpenTelemetry Resource attributes to CS 4.0 ext.* slots.

    Attribute name → CS slot mapping:
        device_id       → ext.device.localId
        id_status       → ext.device.authId
        os.arch         → ext.device.deviceClass
        os.name         → ext.os.name
        os.version      → ext.os.ver
        os.release      → ext.os.release
        app_version     → ext.app.ver
        app_instance_id → ext.app.sesId
        initTs          → ext.app.initTs
    """
    if resource is None:
        return {}
    attrs = dict(resource.attributes or {})
    ext: dict[str, dict] = {}
    device: dict = {}
    os_: dict = {}
    app: dict = {}

    if "device_id" in attrs:
        device["localId"] = attrs["device_id"]
    if "id_status" in attrs:
        device["authId"] = attrs["id_status"]
    if "os.arch" in attrs:
        device["deviceClass"] = attrs["os.arch"]
    if "os.name" in attrs:
        os_["name"] = attrs["os.name"]
    if "os.version" in attrs:
        os_["ver"] = attrs["os.version"]
    if "os.release" in attrs:
        os_["release"] = attrs["os.release"]
    if "app_version" in attrs:
        app["ver"] = attrs["app_version"]
    if "app_instance_id" in attrs:
        app["sesId"] = attrs["app_instance_id"]
    if "initTs" in attrs:
        app["initTs"] = attrs["initTs"]

    if device:
        ext["device"] = device
    if os_:
        ext["os"] = os_
    if app:
        ext["app"] = app
    return ext
