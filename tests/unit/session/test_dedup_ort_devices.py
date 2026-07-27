# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``_dedup_ort_devices`` in ep_registry.

The helper collapses :class:`ort.OrtEpDevice` handles that share
``(vendor_id, device_id, type.name)`` — a workaround for hosts (dual-iGPU
listings, OpenVINO on Intel) that emit duplicate handles for the same
physical device. Handles missing the introspection attributes pass through
unchanged so an introspection failure can never silently drop a device.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

from winml.modelkit.session.ep_registry import _dedup_ort_devices


def _fake_device(vendor_id: int, device_id: int, type_name: str) -> MagicMock:
    """Build a MagicMock matching the OrtEpDevice introspection shape."""
    d = MagicMock()
    d.device.vendor_id = vendor_id
    d.device.device_id = device_id
    d.device.type.name = type_name
    return d


def test_collapses_by_vendor_device_type() -> None:
    """Devices sharing ``(vendor_id, device_id, type.name)`` collapse to one."""
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
