# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""In-process DXGI application memory accounting for an explicit adapter LUID."""

from __future__ import annotations

import ctypes
from typing import ClassVar

from ._gpu_baseline import _check, _guid, _Luid, _method


class _VideoMemoryInfo(ctypes.Structure):
    _fields_: ClassVar = [
        (name, ctypes.c_uint64)
        for name in ("budget", "usage", "available_reservation", "reservation")
    ]


class DxgiMemoryReader:
    """Retain an IDXGIAdapter3; no D3D device or GPU allocation is created."""

    def __init__(self, luid: str) -> None:
        high, low = (int(part, 16) for part in luid.replace(":", "_").split("_"))
        self.adapter = ctypes.c_void_p()
        factory = ctypes.c_void_p()
        dxgi = ctypes.WinDLL("dxgi")
        create = dxgi.CreateDXGIFactory1
        create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_int32
        try:
            _check(create(_guid("1bc6ea02-ef36-464f-bf0c-21ca39e5168a"), ctypes.byref(factory)))
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
                    _guid("645967a4-1392-4310-a798-8053ce3e93fd"),
                    ctypes.byref(self.adapter),
                )
            )
        finally:
            if factory.value:
                _method(factory, 2, ctypes.c_ulong)(factory)

    def sample(self) -> dict[str, dict]:
        """Return byte counters and per-query HRESULT; failures remain unavailable."""
        if not self.adapter.value:
            raise RuntimeError("DXGI memory reader closed")
        result = {}
        for segment, name in ((0, "local"), (1, "nonlocal")):
            info = _VideoMemoryInfo()
            hr = _method(
                self.adapter,
                14,
                ctypes.c_int32,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.POINTER(_VideoMemoryInfo),
            )(self.adapter, 0, segment, ctypes.byref(info))
            result[name] = {
                "value_bytes": int(info.usage) if hr >= 0 else None,
                "budget_bytes": int(info.budget) if hr >= 0 else None,
                "hresult": f"0x{hr & 0xFFFFFFFF:08X}",
                "source": "IDXGIAdapter3::QueryVideoMemoryInfo CurrentUsage",
            }
        return result

    def close(self) -> None:
        """Release adapter reference."""
        if self.adapter.value:
            _method(self.adapter, 2, ctypes.c_ulong)(self.adapter)
            self.adapter = ctypes.c_void_p()
