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
    dedicated_memory_mib: int | None = None
    shared_memory_mib: int | None = None


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
_PROPERTY_DEDICATED_ADAPTER_MEMORY = 7
_PROPERTY_SHARED_SYSTEM_MEMORY = 9
_PROPERTY_HARDWARE_ID_PARTS = 14
_PROPERTY_IS_HARDWARE = 11

# DXCoreAdapterPreference::HighPerformance (dxcore_interface.h).
_PREFERENCE_HIGH_PERFORMANCE = 2


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


def _get_memory_mib(adapter: ctypes.c_void_p, property_id: int) -> int | None:
    if not _is_property_supported(adapter, property_id):
        return None
    value = ctypes.c_uint64()
    _get_property(adapter, property_id, value)
    return value.value // (1024 * 1024)


def _format_luid(luid: _LUID) -> str:
    return f"0x{luid.high_part & 0xFFFFFFFF:08X}_0x{luid.low_part:08X}"


def _create_factory() -> tuple[Any, ctypes.c_void_p]:
    """Load ``dxcore.dll`` and create an owned ``IDXCoreAdapterFactory1``."""
    dxcore = ctypes.WinDLL("dxcore.dll")
    create_factory = dxcore.DXCoreCreateAdapterFactory
    create_factory.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    create_factory.restype = ctypes.c_long

    factory = ctypes.c_void_p()
    _check_hresult(
        create_factory(ctypes.byref(_IID_FACTORY1), ctypes.byref(factory)),
        "DXCoreCreateAdapterFactory",
    )
    return dxcore, factory


def _create_adapter_list(factory: ctypes.c_void_p, attribute: _GUID) -> ctypes.c_void_p:
    """Return an owned ``IDXCoreAdapterList`` of the adapters that have ``attribute``."""
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
    return adapter_list


def _adapter_count(adapter_list: ctypes.c_void_p) -> int:
    get_count = _method(adapter_list, 4, ctypes.c_uint32)
    return int(get_count(adapter_list))


def _get_adapter(adapter_list: ctypes.c_void_p, index: int) -> ctypes.c_void_p:
    """Return an owned ``IDXCoreAdapter`` reference to entry ``index`` of the list."""
    adapter = ctypes.c_void_p()
    get_adapter = _method(
        adapter_list,
        3,
        ctypes.c_long,
        ctypes.c_uint32,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    )
    _check_hresult(
        get_adapter(
            adapter_list,
            index,
            ctypes.byref(_IID_ADAPTER),
            ctypes.byref(adapter),
        ),
        "IDXCoreAdapterList.GetAdapter",
    )
    return adapter


def _sort_high_performance(adapter_list: ctypes.c_void_p) -> bool:
    """Sort the list by the OS high-performance GPU preference.

    Returns:
        ``False`` when the list does not support the preference or ``Sort``
        fails; the list then keeps DXCore's enumeration order.
    """
    is_preference_supported = _method(adapter_list, 8, ctypes.c_bool, ctypes.c_uint32)
    if not is_preference_supported(adapter_list, _PREFERENCE_HIGH_PERFORMANCE):
        return False
    preferences = (ctypes.c_uint32 * 1)(_PREFERENCE_HIGH_PERFORMANCE)
    sort = _method(
        adapter_list,
        7,
        ctypes.c_long,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    )
    return bool(sort(adapter_list, len(preferences), preferences) >= 0)


def _read_adapter_info(adapter: ctypes.c_void_p, device_type: str) -> DXCoreAdapterInfo | None:
    """Read one adapter's identity, or return ``None`` for a software adapter."""
    is_hardware = ctypes.c_bool()
    _get_property(adapter, _PROPERTY_IS_HARDWARE, is_hardware)
    if not is_hardware.value:
        return None
    luid = _LUID()
    _get_property(adapter, _PROPERTY_INSTANCE_LUID, luid)
    if _is_property_supported(adapter, _PROPERTY_HARDWARE_ID_PARTS):
        hardware_id: _HardwareIDParts | _HardwareID = _HardwareIDParts()
        _get_property(adapter, _PROPERTY_HARDWARE_ID_PARTS, hardware_id)
    else:
        hardware_id = _HardwareID()
        _get_property(adapter, _PROPERTY_HARDWARE_ID, hardware_id)
    return DXCoreAdapterInfo(
        device_type=device_type,
        name=_get_string_property(adapter, _PROPERTY_DRIVER_DESCRIPTION),
        luid=_format_luid(luid),
        vendor_id=hardware_id.vendor_id,
        device_id=hardware_id.device_id,
        dedicated_memory_mib=_get_memory_mib(adapter, _PROPERTY_DEDICATED_ADAPTER_MEMORY),
        shared_memory_mib=_get_memory_mib(adapter, _PROPERTY_SHARED_SYSTEM_MEMORY),
    )


def _enumerate_for_type(
    factory: ctypes.c_void_p,
    device_type: str,
    attribute: _GUID,
) -> list[DXCoreAdapterInfo]:
    adapter_list = _create_adapter_list(factory, attribute)
    adapters: list[DXCoreAdapterInfo] = []
    try:
        for index in range(_adapter_count(adapter_list)):
            adapter = _get_adapter(adapter_list, index)
            try:
                info = _read_adapter_info(adapter, device_type)
            finally:
                _release(adapter)
            if info is not None:
                adapters.append(info)
    finally:
        _release(adapter_list)
    return adapters


def enumerate_compute_adapters() -> list[DXCoreAdapterInfo]:
    """Return Windows-native GPU/NPU identities without loading ORT providers."""
    if sys.platform != "win32":
        return []

    _dxcore, factory = _create_factory()
    try:
        return [
            *_enumerate_for_type(factory, "NPU", _ATTRIBUTE_NPU),
            *_enumerate_for_type(factory, "GPU", _ATTRIBUTE_GPU),
        ]
    finally:
        _release(factory)


class DXCoreAdapterError(OSError):
    """DXCore could not list the GPUs, or no hardware GPU fits the request.

    It deliberately has no ``hresult`` attribute, so Windows ML HRESULT
    translators never mistake it for a ``WinMLError``; a DXCore HRESULT is
    part of the message instead.
    """


class DXCoreAdapterHandle:
    """One owned ``IDXCoreAdapter`` reference for ``Runtime.create_target_from_adapter``.

    The Windows ML Runtime keeps no reference of its own to the adapter, so
    close the handle only after everything built on its execution target is
    closed.
    """

    def __init__(
        self,
        pointer: ctypes.c_void_p,
        module: Any,
        *,
        info: DXCoreAdapterInfo,
        selection: str,
    ) -> None:
        address = pointer.value
        if not address:
            raise ValueError("DXCore adapter pointer must not be null.")
        self._pointer: ctypes.c_void_p | None = pointer
        self._address = address
        self._module = module
        self._info = info
        self._selection = selection

    @property
    def pointer(self) -> int:
        """``IDXCoreAdapter*`` address for ``Runtime.create_target_from_adapter``.

        Raises:
            RuntimeError: The handle is closed.
        """
        if self._pointer is None:
            raise RuntimeError("The DXCore adapter handle is closed.")
        return self._address

    @property
    def info(self) -> DXCoreAdapterInfo:
        """Identity of the selected GPU."""
        return self._info

    @property
    def name(self) -> str:
        """Driver description of the selected GPU."""
        return self._info.name

    @property
    def luid(self) -> str:
        """LUID of the selected GPU, formatted as ``0xHHHHHHHH_0xLLLLLLLL``."""
        return self._info.luid

    @property
    def selection(self) -> str:
        """How the GPU was selected: ``device-luid``, ``high-performance`` or ``only-gpu``."""
        return self._selection

    def close(self) -> None:
        """Release the adapter reference once, then drop the ``dxcore.dll`` reference."""
        pointer, self._pointer = self._pointer, None
        if pointer is not None:
            _release(pointer)
        self._module = None

    def __enter__(self) -> DXCoreAdapterHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _select_gpu(
    gpus: list[DXCoreAdapterInfo],
    device_luid: str | None,
    *,
    sorted_ok: bool,
) -> tuple[int, str]:
    """Return the index of the hardware GPU to own and how it was selected."""
    if device_luid is not None:
        for index, info in enumerate(gpus):
            if info.luid.casefold() == device_luid.casefold():
                return index, "device-luid"
    if not gpus:
        raise DXCoreAdapterError(
            "CGC text generation needs a hardware GPU, but DXCore reports none."
        )
    listing = "; ".join(f"{info.name} ({info.luid})" for info in gpus)
    if device_luid is not None:
        raise DXCoreAdapterError(
            f"No hardware GPU has LUID {device_luid}. Hardware GPUs: {listing}. "
            "Run 'winml sys' to list adapters; a LUID changes after a reboot or a driver update."
        )
    if sorted_ok:
        return 0, "high-performance"
    if len(gpus) == 1:
        return 0, "only-gpu"
    raise DXCoreAdapterError(
        "Windows cannot rank the GPUs by performance here. "
        f"Select one with --device-luid: {listing}."
    )


def open_gpu_adapter(device_luid: str | None = None) -> DXCoreAdapterHandle:
    """Own the hardware GPU that ``device_luid`` names, else the high-performance GPU.

    Without ``device_luid``, DXCore sorts the GPUs by the OS high-performance
    preference and the first hardware GPU is used. When DXCore cannot sort the
    list, the only hardware GPU is used, and several need ``device_luid``.
    The adapter list, the factory and every adapter not selected are released
    before this returns. It needs no ORT and creates no Direct3D device.

    Args:
        device_luid: A ``0xHHHHHHHH_0xLLLLLLLL`` LUID from ``winml sys``,
            matched case-insensitively, or ``None`` for the high-performance GPU.

    Returns:
        The owned adapter. Close it after everything built on it is closed.

    Raises:
        DXCoreAdapterError: DXCore failed, or no hardware GPU fits the request.
    """
    factory: ctypes.c_void_p | None = None
    adapter_list: ctypes.c_void_p | None = None
    # owned[i] is the still-unreleased adapter reference whose identity is gpus[i].
    owned: list[ctypes.c_void_p] = []
    try:
        module, factory = _create_factory()
        adapter_list = _create_adapter_list(factory, _ATTRIBUTE_GPU)
        sorted_ok = device_luid is None and _sort_high_performance(adapter_list)
        gpus: list[DXCoreAdapterInfo] = []
        for index in range(_adapter_count(adapter_list)):
            adapter = _get_adapter(adapter_list, index)
            owned.append(adapter)
            info = _read_adapter_info(adapter, "GPU")
            if info is None:  # a software adapter
                owned.pop()
                _release(adapter)
                continue
            gpus.append(info)
        selected, selection = _select_gpu(gpus, device_luid, sorted_ok=sorted_ok)
        handle = DXCoreAdapterHandle(
            owned[selected], module, info=gpus[selected], selection=selection
        )
        del owned[selected]
        return handle
    except DXCoreAdapterError:
        raise
    except OSError as exc:
        raise DXCoreAdapterError(f"DXCore could not list the GPUs: {exc}") from exc
    finally:
        for unselected in owned:
            _release(unselected)
        if adapter_list is not None:
            _release(adapter_list)
        if factory is not None:
            _release(factory)
