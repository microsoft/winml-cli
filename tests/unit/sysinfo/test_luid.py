# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for platform-neutral adapter LUID helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from winml.modelkit.sysinfo import format_pdh_luid, get_ep_device_luid


@pytest.mark.parametrize(
    ("decimal_luid", "expected"),
    [
        ("0", "0x00000000_0x00000000"),
        ("99219", "0x00000000_0x00018393"),
        ("4295066515", "0x00000001_0x00018393"),
        (str((1 << 64) - 1), "0xFFFFFFFF_0xFFFFFFFF"),
    ],
)
def test_format_pdh_luid(decimal_luid: str, expected: str) -> None:
    assert format_pdh_luid(decimal_luid) == expected


def test_get_ep_device_luid_formats_concrete_device_metadata() -> None:
    ep_device = SimpleNamespace(device=SimpleNamespace(metadata={"LUID": "99219"}))

    assert get_ep_device_luid(ep_device) == "0x00000000_0x00018393"


@pytest.mark.parametrize("metadata", [{}, {"LUID": ""}, {"LUID": "invalid"}, None])
def test_get_ep_device_luid_returns_none_when_metadata_has_no_valid_luid(
    metadata: object,
) -> None:
    ep_device = SimpleNamespace(device=SimpleNamespace(metadata=metadata))

    assert get_ep_device_luid(ep_device) is None
