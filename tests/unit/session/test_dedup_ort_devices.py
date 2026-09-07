# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Only identical LUID-backed EP routes may be deduplicated."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest

from winml.modelkit.session.ep_registry import _dedup_ort_devices


def _fake_device(vendor_id: int, device_id: int, type_name: str) -> MagicMock:
    """Build a MagicMock matching the OrtEpDevice introspection shape."""
    d = MagicMock()
    d.device.vendor_id = vendor_id
    d.device.device_id = device_id
    d.device.type.name = type_name
    d.device.metadata = {"LUID": str((vendor_id << 32) | device_id)}
    d.ep_name = "DmlExecutionProvider"
    d.ep_options = {}
    return d


def test_collapses_same_luid_and_route() -> None:
    """Duplicate routes to the same LUID collapse to one."""
    dup_a = _fake_device(0x8086, 0x0001, "GPU")
    dup_b = _fake_device(0x8086, 0x0001, "GPU")  # same key as dup_a
    distinct_device = _fake_device(0x8086, 0x0002, "GPU")
    distinct_vendor = _fake_device(0x10DE, 0x0001, "GPU")

    out = _dedup_ort_devices([dup_a, dup_b, distinct_device, distinct_vendor])

    assert len(out) == 3
    # First-occurrence-wins: dup_a survives, dup_b is dropped.
    assert dup_a in out
    assert dup_b not in out
    assert distinct_device in out
    assert distinct_vendor in out


@pytest.mark.parametrize("metadata_kind", ["distinct-luids", "missing", "invalid"])
def test_identical_products_are_preserved(metadata_kind: str) -> None:
    devices = [_fake_device(0x8086, 0x0001, "GPU") for _ in range(2)]
    for index, device in enumerate(devices):
        device.device.metadata = (
            {"LUID": str(index + 1)}
            if metadata_kind == "distinct-luids"
            else ({} if metadata_kind == "missing" else {"LUID": "invalid"})
        )
    assert _dedup_ort_devices(devices) == devices


@pytest.mark.parametrize("axis", ["ep", "type", "options"])
def test_distinct_routes_to_same_luid_are_preserved(axis: str) -> None:
    devices = [_fake_device(0x8086, 0x0001, "GPU") for _ in range(2)]
    if axis == "ep":
        devices[1].ep_name = "OpenVINOExecutionProvider"
    elif axis == "type":
        devices[1].device.type.name = "NPU"
    else:
        devices[1].ep_options = {"device_id": "1"}
    assert _dedup_ort_devices(devices) == devices


def test_attribute_error_passthrough() -> None:
    """A handle that raises ``AttributeError`` on ``.device`` is preserved.

    The defensive ``except AttributeError: out.append(d)`` branch must
    fire so an introspection bug never silently drops a device.
    """
    broken = MagicMock()
    type(broken).device = PropertyMock(side_effect=AttributeError("no device attr"))

    out = _dedup_ort_devices([broken])

    assert out == [broken]


def test_empty_input() -> None:
    """Empty input yields empty output."""
    assert _dedup_ort_devices([]) == []
