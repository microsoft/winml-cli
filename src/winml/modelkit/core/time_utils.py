# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Simple timestamp formatting utility."""

from datetime import datetime, timezone
from typing import overload


@overload
def format_timestamp_iso(epoch_time: float) -> str: ...
@overload
def format_timestamp_iso(epoch_time: None) -> None: ...
@overload
def format_timestamp_iso(epoch_time: float | None) -> str | None: ...
def format_timestamp_iso(epoch_time: float | None) -> str | None:
    """Format Unix epoch timestamp to ISO 8601 with Z suffix.

    Args:
        epoch_time: Unix epoch timestamp as float, or None

    Returns:
        ISO 8601 string with milliseconds and Z suffix, or None if input is None
    """
    if epoch_time is None:
        return None
    dt = datetime.fromtimestamp(epoch_time, tz=timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
