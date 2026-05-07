# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Stable-per-device ID used as the telemetry device_id.

Derived from a random UUID4, hashed with SHA256, and persisted so the same
machine reports the same id across sessions. Users can reset by removing the
stored value (registry on Windows, state file elsewhere).
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from enum import StrEnum

from . import _store


_LOGGER = logging.getLogger(__name__)
_STORAGE_KEY = "deviceid"


class IdStatus(StrEnum):
    """Outcome of :func:`get_or_create_device_id`.

    Subclassing ``str`` keeps the enum serialization-compatible with
    OpenTelemetry resource attributes (which require str values) and
    with the CS 4.0 ``ext.device.authId`` slot on the wire.
    """

    EXISTING = "EXISTING"
    NEW = "NEW"
    FAILED = "FAILED"


def get_or_create_device_id() -> tuple[str, IdStatus]:
    """Return ``(device_id, status)``.

    - :attr:`IdStatus.EXISTING`: read from persistent storage
    - :attr:`IdStatus.NEW`:      freshly generated and persisted
    - :attr:`IdStatus.FAILED`:   storage unavailable; caller should proceed with empty id
    """
    try:
        existing = _store.read_key(_STORAGE_KEY)
    except Exception:  # defensive: any storage error means we treat as fresh
        _LOGGER.debug("deviceid read failed", exc_info=True)
        existing = None

    if existing:
        return existing, IdStatus.EXISTING

    new_id = _hash_uuid(uuid.uuid4())
    try:
        _store.write_key(_STORAGE_KEY, new_id)
    except Exception:
        _LOGGER.debug("deviceid write failed", exc_info=True)
        return "", IdStatus.FAILED
    return new_id, IdStatus.NEW


def _hash_uuid(value: uuid.UUID) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
