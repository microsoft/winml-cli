# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""OneCollector log exporter: serializes OTel log records as CS 4.0 envelopes and POSTs them."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import requests
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult

from .serialization import _build_envelope, _serialize_batch


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.sdk._logs import ReadableLogRecord

_LOGGER = logging.getLogger(__name__)

_HTTP_TIMEOUT = 10.0


class OneCollectorLogExporter(LogRecordExporter):
    """Post Common Schema 4.0 event envelopes to the OneCollector endpoint."""

    def __init__(self, ikey: str, endpoint: str) -> None:
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
        self._endpoint = endpoint
        # _shutdown is read on the BatchLogRecordProcessor export thread and
        # written on the shutdown thread; bool assignment is atomic under the
        # CPython GIL, so no lock is needed.
        self._shutdown = False
        # Close the session if post-creation setup fails, so we never leak a
        # Session object (and its connection pool) when init raises.
        session = requests.Session()
        try:
            session.headers.update(
                {
                    "Content-Type": "application/json; charset=utf-8",
                }
            )
        except Exception:
            session.close()
            raise
        self._session = session

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        """Serialize *batch* and POST to the configured OneCollector endpoint.

        Retries are **not** implemented at this layer; the upstream
        ``BatchLogRecordProcessor`` re-queues batches for which ``export``
        returns ``FAILURE``.
        """
        if self._shutdown or not batch:
            return LogRecordExportResult.SUCCESS

        try:
            envelopes = [self._to_envelope(ld) for ld in batch]
            body = _serialize_batch(envelopes)
        except Exception:
            _LOGGER.debug("telemetry serialization failed", exc_info=True)
            return LogRecordExportResult.FAILURE

        try:
            response = self._session.post(
                self._endpoint,
                data=body,
                timeout=_HTTP_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout):
            _LOGGER.debug("telemetry network failure", exc_info=True)
            return LogRecordExportResult.FAILURE

        if 200 <= response.status_code < 300:
            return LogRecordExportResult.SUCCESS
        _LOGGER.debug("telemetry backend returned %s", response.status_code)
        return LogRecordExportResult.FAILURE

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
            ikey=self._ikey,
            timestamp=timestamp,
            data=data,
            ext=ext,
        )


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
