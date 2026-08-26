# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for Windows-native DXCore adapter discovery helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from winml.modelkit.sysinfo import dxcore_adapters
from winml.modelkit.sysinfo.dxcore_adapters import _LUID, _format_luid


if TYPE_CHECKING:
    import pytest


def test_format_luid_uses_unsigned_high_and_low_halves() -> None:
    assert _format_luid(_LUID(low_part=0x135AA, high_part=0)) == (
        "0x00000000_0x000135AA"
    )
    assert _format_luid(_LUID(low_part=1, high_part=-1)) == (
        "0xFFFFFFFF_0x00000001"
    )


def test_enumeration_is_empty_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dxcore_adapters.sys, "platform", "linux")

    assert dxcore_adapters.enumerate_compute_adapters() == []
