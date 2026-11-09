# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CGC lowering compatibility for Runtime wheels with incomplete projections."""

from __future__ import annotations

import ctypes
from typing import Any, ClassVar

import click


def _native_options(options: Any) -> Any:
    # Lazy import: the optional Windows Runtime must not become an import-time
    # dependency of modelkit. IID and method order are from WinMLRuntime.h.
    from comtypes import GUID, STDMETHOD, COMError, IUnknown

    class IWinMLCgcCompileOptions(IUnknown):
        _iid_ = GUID("{2a7d4b91-0f6c-4f8a-9d54-1c3b7e6a8f42}")
        _methods_: ClassVar[list[Any]] = [
            STDMETHOD(ctypes.c_int32, "SetCgcLoweringTarget", [ctypes.c_int32]),
            STDMETHOD(ctypes.c_int32, "GetCgcLoweringTarget", [ctypes.c_void_p]),
        ]

    try:
        return options.interface.QueryInterface(IWinMLCgcCompileOptions)
    except COMError as error:
        raise RuntimeError("Runtime options do not expose IWinMLCgcCompileOptions") from error


def set_cgc_lowering_target(options: Any, runtime: Any, target: str) -> None:
    """Use the public projection first, otherwise negotiate the native interface.

    Runtime 2.8.204 omits the Python enum/property while its native options still
    implement IWinMLCgcCompileOptions. No Runtime classes or modules are patched.
    """
    names = {"auto": "AUTOMATIC", "disable-d3d12": "DML", "d3d12": "D3D12"}
    if target not in names:
        raise ValueError(f"Unknown CGC lowering target: {target!r}")
    enum = getattr(runtime, "CgcLoweringTarget", None)
    if enum is not None and hasattr(options, "cgc_lowering_target"):
        value = getattr(enum, names[target])
        options.cgc_lowering_target = value
        if options.cgc_lowering_target != value:
            raise click.ClickException(
                "Runtime CGC lowering readback differs from the requested target."
            )
        return
    if target == "auto":
        # Every fresh compile-options object defaults to automatic lowering.
        return
    value = {"disable-d3d12": 1, "d3d12": 2}[target]
    try:
        native = _native_options(options)
        status = native.SetCgcLoweringTarget(value)
        if status < 0:
            raise RuntimeError(f"SetCgcLoweringTarget failed: 0x{status & 0xFFFFFFFF:08X}")
        actual = ctypes.c_int32()
        status = native.GetCgcLoweringTarget(ctypes.byref(actual))
        if status < 0:
            raise RuntimeError(f"GetCgcLoweringTarget failed: 0x{status & 0xFFFFFFFF:08X}")
        if actual.value != value:
            raise RuntimeError(f"Lowering readback {actual.value} differs from requested {value}")
    except (OSError, RuntimeError, AttributeError) as error:
        raise click.ClickException(
            f"Cannot apply CGC lowering target {target!r}: {error}. "
            "Automatic lowering was not substituted."
        ) from error
