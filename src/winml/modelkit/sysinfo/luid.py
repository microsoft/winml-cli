# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Platform-neutral helpers for ORT and PDH adapter LUIDs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from collections.abc import Mapping


class _HardwareDeviceWithMetadata(Protocol):
    metadata: Mapping[str, str]


class _EPDeviceWithMetadata(Protocol):
    device: _HardwareDeviceWithMetadata


def format_pdh_luid(decimal_luid: str) -> str:
    """Format an ORT decimal LUID as a PDH high/low hexadecimal pair."""
    value = int(decimal_luid)
    return f"0x{(value >> 32) & 0xFFFFFFFF:08X}_0x{value & 0xFFFFFFFF:08X}"


def get_ep_device_luid(ep_device: _EPDeviceWithMetadata) -> str | None:
    """Return the PDH-formatted LUID published by one concrete ORT EP device."""
    try:
        metadata = dict(ep_device.device.metadata)
    except (TypeError, ValueError):
        return None

    decimal_luid = metadata.get("LUID")
    if not decimal_luid:
        return None
    try:
        return format_pdh_luid(decimal_luid)
    except (TypeError, ValueError):
        return None
