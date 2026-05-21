# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Device ID: a stable CS 4.0 ``r:<uuid>`` localId persisted per user."""

from .deviceid import IdStatus, get_or_create_device_id


__all__ = ["IdStatus", "get_or_create_device_id"]
