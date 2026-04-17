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

from . import _store


_LOGGER = logging.getLogger(__name__)
_STORAGE_KEY = "deviceid"


def get_or_create_device_id() -> tuple[str, str]:
    """Return (device_id, status) where status is one of.

    Status values:
    - "EXISTING": read from persistent storage
    - "NEW":      freshly generated and persisted
    - "FAILED":   storage unavailable; caller should proceed with empty id
    """
    try:
        existing = _store.read_key(_STORAGE_KEY)
    except Exception:  # defensive: any storage error means we treat as fresh
        _LOGGER.debug("deviceid read failed", exc_info=True)
        existing = None

    if existing:
        return existing, "EXISTING"

    new_id = _hash_uuid(uuid.uuid4())
    try:
        _store.write_key(_STORAGE_KEY, new_id)
    except Exception:
        _LOGGER.debug("deviceid write failed", exc_info=True)
        return "", "FAILED"
    return new_id, "NEW"


def _hash_uuid(value: uuid.UUID) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
