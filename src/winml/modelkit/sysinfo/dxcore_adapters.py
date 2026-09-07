# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Windows-native GPU and NPU discovery through DXCore."""

from __future__ import annotations

import ctypes
import sys
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DXCoreAdapterInfo:
    """Stable system identity and metadata for one compute adapter."""

    device_type: str
    name: str
    luid: str
    vendor_id: int
    device_id: int


class _GUID(ctypes.Structure):
    _fields_ = [
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> _GUID:
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


class _LUID(ctypes.Structure):
    _fields_ = [("low_part", ctypes.c_uint32), ("high_part", ctypes.c_int32)]


class _HardwareIDParts(ctypes.Structure):
    _fields_ = [
        ("vendor_id", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("subsystem_id", ctypes.c_uint32),
        ("subvendor_id", ctypes.c_uint32),
        ("revision_id", ctypes.c_uint32),
    ]


class _HardwareID(ctypes.Structure):
    _fields_ = [
        ("vendor_id", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("subsystem_id", ctypes.c_uint32),
        ("revision_id", ctypes.c_uint32),
    ]


_IID_FACTORY1 = _GUID.from_string("d5682e19-6d21-401c-827a-9a51a4ea35d7")
_IID_ADAPTER_LIST = _GUID.from_string("526c7776-40e9-459b-b711-f32ad76dfc28")
_IID_ADAPTER = _GUID.from_string("f0db4c7f-fe5a-42a2-bd62-f2a6cf6fc83e")

_ATTRIBUTE_GPU = _GUID.from_string("b69eb219-3ded-4464-979f-a00bd4687006")
_ATTRIBUTE_NPU = _GUID.from_string("d46140c4-add7-451b-9e56-06fe8c3b58ed")

_PROPERTY_INSTANCE_LUID = 0
_PROPERTY_DRIVER_DESCRIPTION = 2
_PROPERTY_HARDWARE_ID = 3
_PROPERTY_HARDWARE_ID_PARTS = 14
_PROPERTY_IS_HARDWARE = 11

def _method(
    instance: ctypes.c_void_p,
    index: int,
    restype: Any,
    *argtypes: Any,
) -> Any:
    vtable = ctypes.cast(instance, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[index])


def _release(instance: ctypes.c_void_p) -> None:
    if instance:
        _method(instance, 2, ctypes.c_ulong)(instance)


def _check_hresult(result: int, operation: str) -> None:
    if result < 0:
        raise OSError(f"{operation} failed with HRESULT 0x{result & 0xFFFFFFFF:08X}")


def _get_property(
    adapter: ctypes.c_void_p,
    property_id: int,
    value: ctypes.Structure | ctypes._SimpleCData[Any],
) -> None:
    get_property = _method(
        adapter,
        6,
        ctypes.c_long,
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    result = get_property(
        adapter,
        property_id,
        ctypes.sizeof(value),
        ctypes.byref(value),
    )
    _check_hresult(result, f"IDXCoreAdapter.GetProperty({property_id})")


def _is_property_supported(adapter: ctypes.c_void_p, property_id: int) -> bool:
    is_supported = _method(
        adapter,
        5,
        ctypes.c_bool,
        ctypes.c_uint32,
    )
    return bool(is_supported(adapter, property_id))


def _get_string_property(adapter: ctypes.c_void_p, property_id: int) -> str:
    size = ctypes.c_size_t()
    get_size = _method(
        adapter,
        7,
        ctypes.c_long,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_size_t),
    )
    _check_hresult(
        get_size(adapter, property_id, ctypes.byref(size)),
        f"IDXCoreAdapter.GetPropertySize({property_id})",
    )
    if size.value == 0:
        return ""
    buffer = ctypes.create_string_buffer(size.value)
    get_property = _method(
        adapter,
        6,
        ctypes.c_long,
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    _check_hresult(
        get_property(adapter, property_id, size.value, buffer),
        f"IDXCoreAdapter.GetProperty({property_id})",
    )
    return buffer.value.decode("utf-8", errors="replace")


def _format_luid(luid: _LUID) -> str:
    return f"0x{luid.high_part & 0xFFFFFFFF:08X}_0x{luid.low_part:08X}"


def _enumerate_for_type(
    factory: ctypes.c_void_p,
    device_type: str,
    attribute: _GUID,
) -> list[DXCoreAdapterInfo]:
    adapter_list = ctypes.c_void_p()
    create_list = _method(
        factory,
        3,
        ctypes.c_long,
        ctypes.c_uint32,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    )
    result = create_list(
        factory,
        1,
        ctypes.byref(attribute),
        ctypes.byref(_IID_ADAPTER_LIST),
        ctypes.byref(adapter_list),
    )
    _check_hresult(result, "IDXCoreAdapterFactory.CreateAdapterList")

    adapters: list[DXCoreAdapterInfo] = []
    try:
        get_count = _method(adapter_list, 4, ctypes.c_uint32)
        get_adapter = _method(
            adapter_list,
            3,
            ctypes.c_long,
            ctypes.c_uint32,
            ctypes.POINTER(_GUID),
            ctypes.POINTER(ctypes.c_void_p),
        )
        for index in range(get_count(adapter_list)):
            adapter = ctypes.c_void_p()
            _check_hresult(
                get_adapter(
                    adapter_list,
                    index,
                    ctypes.byref(_IID_ADAPTER),
                    ctypes.byref(adapter),
                ),
                "IDXCoreAdapterList.GetAdapter",
            )
            try:
                is_hardware = ctypes.c_bool()
                _get_property(adapter, _PROPERTY_IS_HARDWARE, is_hardware)
                if not is_hardware.value:
                    continue
                luid = _LUID()
                _get_property(adapter, _PROPERTY_INSTANCE_LUID, luid)
                if _is_property_supported(adapter, _PROPERTY_HARDWARE_ID_PARTS):
                    hardware_id: _HardwareIDParts | _HardwareID = _HardwareIDParts()
                    _get_property(adapter, _PROPERTY_HARDWARE_ID_PARTS, hardware_id)
                else:
                    hardware_id = _HardwareID()
                    _get_property(adapter, _PROPERTY_HARDWARE_ID, hardware_id)
                adapters.append(
                    DXCoreAdapterInfo(
                        device_type=device_type,
                        name=_get_string_property(adapter, _PROPERTY_DRIVER_DESCRIPTION),
                        luid=_format_luid(luid),
                        vendor_id=hardware_id.vendor_id,
                        device_id=hardware_id.device_id,
                    )
                )
            finally:
                _release(adapter)
    finally:
        _release(adapter_list)
    return adapters


def enumerate_compute_adapters() -> list[DXCoreAdapterInfo]:
    """Return Windows-native GPU/NPU identities without loading ORT providers."""
    if sys.platform != "win32":
        return []

    dxcore = ctypes.WinDLL("dxcore.dll")
    create_factory = dxcore.DXCoreCreateAdapterFactory
    create_factory.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    create_factory.restype = ctypes.c_long

    factory = ctypes.c_void_p()
    _check_hresult(
        create_factory(ctypes.byref(_IID_FACTORY1), ctypes.byref(factory)),
        "DXCoreCreateAdapterFactory",
    )
    try:
        return [
            *_enumerate_for_type(factory, "NPU", _ATTRIBUTE_NPU),
            *_enumerate_for_type(factory, "GPU", _ATTRIBUTE_GPU),
        ]
    finally:
        _release(factory)
