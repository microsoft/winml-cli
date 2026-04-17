# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Device ID: a stable SHA256-hashed UUID4 persisted per user."""

from .deviceid import get_or_create_device_id


__all__ = ["get_or_create_device_id"]
