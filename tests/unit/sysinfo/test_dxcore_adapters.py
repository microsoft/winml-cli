# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for Windows-native DXCore adapter discovery helpers."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from winml.modelkit.sysinfo import dxcore_adapters
from winml.modelkit.sysinfo.dxcore_adapters import (
    _ATTRIBUTE_NPU,
    _GUID,
    _LUID,
    _enumerate_for_type,
    _format_luid,
)


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


def test_enumeration_filters_by_hardware_type_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, bytes]] = []

    def fake_method(
        instance: ctypes.c_void_p,
        index: int,
        restype: object,
        *argtypes: object,
    ):
        del instance, restype, argtypes

        if index == 3:

            def create_list(
                factory: ctypes.c_void_p,
                attribute_count: int,
                attributes: object,
                iid: object,
                adapter_list: object,
            ) -> int:
                del factory, iid, adapter_list
                guid = ctypes.cast(attributes, ctypes.POINTER(_GUID)).contents
                calls.append((index, attribute_count, bytes(guid)))
                return 0

            return create_list
        if index == 4:
            return lambda adapter_list: 0
        raise AssertionError(f"unexpected vtable method {index}")

    monkeypatch.setattr(dxcore_adapters, "_method", fake_method)
    monkeypatch.setattr(dxcore_adapters, "_release", lambda instance: None)

    assert _enumerate_for_type(ctypes.c_void_p(1), "NPU", _ATTRIBUTE_NPU) == []
    assert calls == [(3, 1, bytes(_ATTRIBUTE_NPU))]
