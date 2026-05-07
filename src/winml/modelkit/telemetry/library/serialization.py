# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Common Schema 4.0 envelope builders for OneCollector events."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from datetime import datetime


def _envelope_ikey(full_ikey: str) -> str:
    """Derive the envelope ``iKey`` value from a OneCollector instrumentation key.

    OneCollector expects two distinct iKey forms on the wire:

    - ``x-apikey`` HTTP header: the full instrumentation key
      (``<tenant_token>-<guid>-<ingestion_token>``).
    - Envelope ``iKey`` field: ``o:<tenant_token>``, where ``tenant_token``
      is the portion of the ikey before the first ``-``.

    Embedding the full ikey in the envelope's ``iKey`` field is what causes
    the backend to reject the request with
    ``Collector-Error: Invalid Tenant Token``.
    """
    dash = full_ikey.find("-")
    if dash <= 0:
        raise ValueError(
            "OneCollector instrumentation key must contain a non-empty "
            "tenant_token portion before the first '-'"
        )
    return f"o:{full_ikey[:dash]}"


def _build_envelope(
    name: str,
    ikey: str,
    timestamp: datetime,
    data: dict[str, Any],
    ext: dict[str, Any],
) -> dict[str, Any]:
    """Build one CS 4.0 event envelope.

    Fields:
        ver: schema version, always "4.0"
        name: event name (e.g. "ModelKitAction")
        time: ISO8601 UTC, millisecond precision, trailing Z
        iKey: envelope iKey value -- caller is responsible for supplying
            it in the ``o:<tenant_token>`` form (use
            :func:`_envelope_ikey` to derive it from the full ikey)
        data: event-specific flat payload
        ext: common context slots (os, app, device)
    """
    return {
        "ver": "4.0",
        "name": name,
        "time": _format_time(timestamp),
        "iKey": ikey,
        "data": data,
        "ext": ext,
    }


def _format_time(ts: datetime) -> str:
    # CS 4.0 expects millisecond precision, trailing Z (UTC).
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def _serialize_batch(envelopes: list[dict[str, Any]]) -> bytes:
    r"""Serialize a batch of envelopes as NDJSON (x-json-stream).

    One envelope per line, ``\n`` separated, no enclosing array. This is the
    wire format the /OneCollector/1.0/ ingest endpoint expects under
    ``application/x-json-stream``; a JSON array is rejected with HTTP 415.
    """
    return b"\n".join(
        json.dumps(env, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        for env in envelopes
    )
