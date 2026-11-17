# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for Windows-native DXCore adapter discovery helpers."""

from __future__ import annotations

import ctypes
import re

import pytest

from winml.modelkit.sysinfo import (
    DXCoreAdapterError,
    DXCoreAdapterHandle,
    DXCoreAdapterInfo,
    dxcore_adapters,
    open_gpu_adapter,
)
from winml.modelkit.sysinfo.dxcore_adapters import (
    _ATTRIBUTE_GPU,
    _ATTRIBUTE_NPU,
    _GUID,
    _LUID,
    _PREFERENCE_HIGH_PERFORMANCE,
    _enumerate_for_type,
    _format_luid,
    _get_memory_mib,
    _sort_high_performance,
)


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


def test_get_memory_mib_returns_64_bit_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dxcore_adapters, "_is_property_supported", lambda *args: True)

    def set_property(
        adapter: ctypes.c_void_p,
        property_id: int,
        value: ctypes.c_uint64,
    ) -> None:
        del adapter, property_id
        value.value = 24 * 1024 * 1024 * 1024

    monkeypatch.setattr(dxcore_adapters, "_get_property", set_property)

    assert _get_memory_mib(ctypes.c_void_p(1), 7) == 24 * 1024


def test_get_memory_mib_returns_none_when_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dxcore_adapters, "_is_property_supported", lambda *args: False)

    assert _get_memory_mib(ctypes.c_void_p(1), 7) is None


# ---------------------------------------------------------------------------
# High-performance GPU selection: a fake DXCore over adapter pointer values.
# The factory is pointer 1 and the adapter list is pointer 2.
# ---------------------------------------------------------------------------

_INTEGRATED_GPU = DXCoreAdapterInfo(
    device_type="GPU",
    name="Integrated GPU",
    luid="0x00000000_0x0000A001",
    vendor_id=1,
    device_id=1,
)
_DISCRETE_GPU = DXCoreAdapterInfo(
    device_type="GPU",
    name="Discrete GPU",
    luid="0x00000000_0x0000B002",
    vendor_id=2,
    device_id=2,
)
# 12 is a software adapter: DXCore lists it, but it is not hardware.
_ADAPTER_INFO: dict[int, DXCoreAdapterInfo | None] = {
    10: _INTEGRATED_GPU,
    11: _DISCRETE_GPU,
    12: None,
}


class _FakeDXCore:
    """Patch the DXCore seams so the adapter list holds ``adapters``."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        adapters: list[int],
        *,
        sorted_order: list[int] | None = None,
    ) -> None:
        self.adapters = adapters
        self.sorted_order = sorted_order
        self.sort_calls = 0
        self.list_attributes: list[bytes] = []
        self.released: list[int | None] = []
        monkeypatch.setattr(
            dxcore_adapters, "_create_factory", lambda: (object(), ctypes.c_void_p(1))
        )
        monkeypatch.setattr(dxcore_adapters, "_create_adapter_list", self._create_adapter_list)
        monkeypatch.setattr(
            dxcore_adapters, "_adapter_count", lambda adapter_list: len(self.adapters)
        )
        monkeypatch.setattr(
            dxcore_adapters,
            "_get_adapter",
            lambda adapter_list, index: ctypes.c_void_p(self.adapters[index]),
        )
        monkeypatch.setattr(dxcore_adapters, "_sort_high_performance", self._sort)
        monkeypatch.setattr(dxcore_adapters, "_read_adapter_info", self._read_adapter_info)
        monkeypatch.setattr(
            dxcore_adapters, "_release", lambda pointer: self.released.append(pointer.value)
        )

    def _create_adapter_list(self, factory: ctypes.c_void_p, attribute: _GUID) -> ctypes.c_void_p:
        assert factory.value == 1
        self.list_attributes.append(bytes(attribute))
        return ctypes.c_void_p(2)

    def _sort(self, adapter_list: ctypes.c_void_p) -> bool:
        assert adapter_list.value == 2
        self.sort_calls += 1
        if self.sorted_order is None:
            return False
        self.adapters[:] = self.sorted_order
        return True

    def _read_adapter_info(
        self, adapter: ctypes.c_void_p, device_type: str
    ) -> DXCoreAdapterInfo | None:
        assert device_type == "GPU"
        return _ADAPTER_INFO[adapter.value]


def test_high_performance_sort_picks_the_first_hardware_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDXCore(monkeypatch, [10, 11, 12], sorted_order=[12, 11, 10])

    handle = open_gpu_adapter()

    assert isinstance(handle, DXCoreAdapterHandle)
    assert (handle.name, handle.luid, handle.selection) == (
        "Discrete GPU",
        "0x00000000_0x0000B002",
        "high-performance",
    )
    assert handle.info == _DISCRETE_GPU
    assert handle.pointer == 11
    assert sorted(fake.released) == [1, 2, 10, 12]
    assert fake.list_attributes == [bytes(_ATTRIBUTE_GPU)]

    handle.close()
    handle.close()

    assert fake.released.count(11) == 1
    with pytest.raises(RuntimeError, match=r"^The DXCore adapter handle is closed\.$"):
        _ = handle.pointer


def test_without_a_sort_the_only_hardware_gpu_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDXCore(monkeypatch, [12, 11])

    with open_gpu_adapter() as handle:
        assert (handle.name, handle.luid, handle.selection) == (
            "Discrete GPU",
            "0x00000000_0x0000B002",
            "only-gpu",
        )
        assert handle.pointer == 11
        assert sorted(fake.released) == [1, 2, 12]

    assert fake.sort_calls == 1
    assert sorted(fake.released) == [1, 2, 11, 12]


def test_without_a_sort_two_gpus_require_device_luid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDXCore(monkeypatch, [10, 11, 12])

    with pytest.raises(
        DXCoreAdapterError, match=r"^Windows cannot rank the GPUs by performance here\."
    ) as caught:
        open_gpu_adapter()

    message = str(caught.value)
    assert "--device-luid" in message
    assert "0x00000000_0x0000A001" in message
    assert "0x00000000_0x0000B002" in message
    assert sorted(fake.released) == [1, 2, 10, 11, 12]


def test_device_luid_selects_the_matching_gpu_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDXCore(monkeypatch, [10, 11, 12], sorted_order=[12, 10, 11])

    handle = open_gpu_adapter("0x00000000_0x0000b002")

    assert (handle.name, handle.luid, handle.selection) == (
        "Discrete GPU",
        "0x00000000_0x0000B002",
        "device-luid",
    )
    assert handle.pointer == 11
    assert fake.sort_calls == 0
    assert sorted(fake.released) == [1, 2, 10, 12]


def test_unknown_device_luid_lists_the_hardware_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDXCore(monkeypatch, [10, 11, 12])

    with pytest.raises(
        DXCoreAdapterError, match=r"^No hardware GPU has LUID 0x00000000_0x0000C003\."
    ) as caught:
        open_gpu_adapter("0x00000000_0x0000C003")

    message = str(caught.value)
    assert "Integrated GPU (0x00000000_0x0000A001)" in message
    assert "Discrete GPU (0x00000000_0x0000B002)" in message
    assert "winml sys" in message
    assert sorted(fake.released) == [1, 2, 10, 11, 12]


def test_unknown_device_luid_never_falls_back_to_the_only_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDXCore(monkeypatch, [11])

    with pytest.raises(
        DXCoreAdapterError,
        match=(
            r"^No hardware GPU has LUID 0x00000000_0x0000A001\. "
            r"Hardware GPUs: Discrete GPU \(0x00000000_0x0000B002\)\."
        ),
    ):
        open_gpu_adapter("0x00000000_0x0000A001")

    assert sorted(fake.released) == [1, 2, 11]


@pytest.mark.parametrize("device_luid", [None, "0x00000000_0x0000A001"])
def test_no_hardware_gpu_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    device_luid: str | None,
) -> None:
    fake = _FakeDXCore(monkeypatch, [12])

    with pytest.raises(DXCoreAdapterError, match=r"^CGC text generation needs a hardware GPU"):
        open_gpu_adapter(device_luid)

    assert sorted(fake.released) == [1, 2, 12]


@pytest.mark.parametrize(
    ("seam", "error", "expected_released"),
    [
        pytest.param(
            "_create_factory",
            FileNotFoundError("Could not find module 'dxcore.dll' (or one of its dependencies)."),
            [],
            id="dxcore-dll-missing",
        ),
        pytest.param(
            "_create_adapter_list",
            OSError("IDXCoreAdapterFactory.CreateAdapterList failed with HRESULT 0x887A0004"),
            [1],
            id="adapter-list",
        ),
        pytest.param(
            "_read_adapter_info",
            OSError("IDXCoreAdapter.GetProperty(11) failed with HRESULT 0x887A0004"),
            [1, 2, 10],
            id="adapter-property",
        ),
    ],
)
def test_dxcore_failures_become_adapter_errors(
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
    error: OSError,
    expected_released: list[int],
) -> None:
    fake = _FakeDXCore(monkeypatch, [10, 11, 12])

    def fail(*args: object) -> None:
        raise error

    monkeypatch.setattr(dxcore_adapters, seam, fail)

    with pytest.raises(
        DXCoreAdapterError,
        match=rf"^DXCore could not list the GPUs: {re.escape(str(error))}$",
    ) as caught:
        open_gpu_adapter()

    assert caught.value.__cause__ is error
    assert not hasattr(caught.value, "hresult")
    assert sorted(fake.released) == expected_released


@pytest.mark.parametrize(
    ("supported", "sort_result", "expected"),
    [
        pytest.param(True, 0, True, id="sorted"),
        pytest.param(False, 0, False, id="preference-unsupported"),
        pytest.param(True, ctypes.c_long(0x80004005).value, False, id="sort-failed"),
    ],
)
def test_sort_high_performance_sorts_only_when_dxcore_supports_the_preference(
    monkeypatch: pytest.MonkeyPatch,
    supported: bool,
    sort_result: int,
    expected: bool,
) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_method(
        instance: ctypes.c_void_p,
        index: int,
        restype: object,
        *argtypes: object,
    ):
        del instance, restype, argtypes
        if index == 8:

            def is_preference_supported(adapter_list: ctypes.c_void_p, preference: int) -> bool:
                calls.append(("IsAdapterPreferenceSupported", adapter_list.value, preference))
                return supported

            return is_preference_supported
        if index == 7:

            def sort(
                adapter_list: ctypes.c_void_p,
                count: int,
                preferences: ctypes.Array[ctypes.c_uint32],
            ) -> int:
                calls.append(("Sort", adapter_list.value, count, list(preferences[:count])))
                return sort_result

            return sort
        raise AssertionError(f"unexpected vtable method {index}")

    monkeypatch.setattr(dxcore_adapters, "_method", fake_method)

    assert _sort_high_performance(ctypes.c_void_p(2)) is expected

    preference = _PREFERENCE_HIGH_PERFORMANCE
    assert calls[0] == ("IsAdapterPreferenceSupported", 2, preference)
    assert calls[1:] == ([("Sort", 2, 1, [preference])] if supported else [])
