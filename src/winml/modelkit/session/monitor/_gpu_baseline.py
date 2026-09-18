# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Keep a model-free D3D12 device alive on the explicitly selected GPU."""

from __future__ import annotations

import ctypes
import re
import sys
import uuid
from typing import Any, ClassVar


class _Luid(ctypes.Structure):
    _fields_: ClassVar = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]


def _guid(value: str) -> ctypes.Array[ctypes.c_ubyte]:
    return (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le)


def _method(pointer: ctypes.c_void_p, index: int, result: Any, *args: Any) -> Any:
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *args)(table[index])


def _check(status: int) -> None:
    if status < 0:
        raise OSError(f"D3D12 baseline device failed: HRESULT 0x{status & 0xFFFFFFFF:08X}")


class GpuBaselineDevice:
    """A retained device, no model, resources, command queue or submitted GPU work."""

    def __init__(self, luid: str) -> None:
        match = re.fullmatch(r"0x([0-9a-f]{8})_0x([0-9a-f]{8})", luid, re.IGNORECASE)
        if not match:
            raise ValueError("Invalid GPU adapter LUID")
        if sys.platform != "win32":
            raise OSError("D3D12 baseline preparation requires Windows")
        high, low = (int(part, 16) for part in match.groups())
        self._device = ctypes.c_void_p()
        factory, adapter = ctypes.c_void_p(), ctypes.c_void_p()
        dxgi = ctypes.WinDLL("dxgi")
        self._d3d12 = ctypes.WinDLL("d3d12")
        dxgi.CreateDXGIFactory1.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        dxgi.CreateDXGIFactory1.restype = ctypes.c_int32
        self._d3d12.D3D12CreateDevice.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._d3d12.D3D12CreateDevice.restype = ctypes.c_int32
        try:
            # IDXGIFactory4::EnumAdapterByLuid, never adapter ordinal zero/fallback.
            _check(
                dxgi.CreateDXGIFactory1(
                    _guid("1bc6ea02-ef36-464f-bf0c-21ca39e5168a"), ctypes.byref(factory)
                )
            )
            _check(
                _method(
                    factory,
                    26,
                    ctypes.c_int32,
                    _Luid,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                )(
                    factory,
                    _Luid(low, ctypes.c_int32(high).value),
                    _guid("2411e7e1-12ac-4ccf-bd14-9798e8534dc0"),
                    ctypes.byref(adapter),
                )
            )
            _check(
                self._d3d12.D3D12CreateDevice(
                    adapter,
                    0xB000,
                    _guid("189819f1-1db6-4b57-be54-1821339b85f7"),
                    ctypes.byref(self._device),
                )
            )
        except Exception:
            self.close()
            raise
        finally:
            for pointer in (adapter, factory):
                if pointer.value:
                    _method(pointer, 2, ctypes.c_ulong)(pointer)

    def close(self) -> None:
        """Release only our device reference after the last benchmark checkpoint."""
        if self._device.value:
            _method(self._device, 2, ctypes.c_ulong)(self._device)
            self._device = ctypes.c_void_p()
