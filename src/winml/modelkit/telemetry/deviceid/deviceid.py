# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

r"""Stable-per-device ID used as the telemetry device_id.

A random UUID4 in CS 4.0 ``ext.device.localId`` form (``r:<canonical-uuid>``,
where ``r`` is the random scope). Persisted so the same machine reports the
same id across sessions. Users can reset by removing the stored value
(``HKCU\SOFTWARE\Microsoft\DeveloperTools\.modelkit``).
"""

from __future__ import annotations

import logging
import re
import uuid
from enum import Enum

from . import _store


_LOGGER = logging.getLogger(__name__)
_STORAGE_KEY = "deviceid"
# OneCollector rejects any localId that isn't <scope>:<canonical-uuid>; the
# 'r' (random) scope is what we generate. Stored values that don't match
# (e.g. SHA256 hex digests written by releases <= 0.0.4) are treated as
# absent and regenerated — see issue #691.
_LOCAL_ID_RE = re.compile(r"^r:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class IdStatus(str, Enum):
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

    if existing and _LOCAL_ID_RE.match(existing):
        return existing, IdStatus.EXISTING

    new_id = f"r:{uuid.uuid4()}"
    try:
        _store.write_key(_STORAGE_KEY, new_id)
    except Exception:
        _LOGGER.debug("deviceid write failed", exc_info=True)
        return "", IdStatus.FAILED
    return new_id, IdStatus.NEW
