# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Lazy public facade for system and device discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .device import (
        get_device_ep_map,
        get_ep_device_map,
        resolve_check_device_ep,
        resolve_device,
        resolve_eps,
    )
    from .dxcore_adapters import DXCoreAdapterInfo, enumerate_compute_adapters
    from .hardware import CPU, GPU, NPU, get_available_devices
    from .luid import format_pdh_luid, get_ep_device_luid
    from .software import OS
    from .sysinfo import SysInfo


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "get_device_ep_map": (".device", "get_device_ep_map"),
    "get_ep_device_map": (".device", "get_ep_device_map"),
    "resolve_check_device_ep": (".device", "resolve_check_device_ep"),
    "resolve_device": (".device", "resolve_device"),
    "resolve_eps": (".device", "resolve_eps"),
    "DXCoreAdapterInfo": (".dxcore_adapters", "DXCoreAdapterInfo"),
    "enumerate_compute_adapters": (".dxcore_adapters", "enumerate_compute_adapters"),
    "CPU": (".hardware", "CPU"),
    "GPU": (".hardware", "GPU"),
    "NPU": (".hardware", "NPU"),
    "get_available_devices": (".hardware", "get_available_devices"),
    "format_pdh_luid": (".luid", "format_pdh_luid"),
    "get_ep_device_luid": (".luid", "get_ep_device_luid"),
    "OS": (".software", "OS"),
    "SysInfo": (".sysinfo", "SysInfo"),
}

__all__ = [
    "CPU",
    "GPU",
    "NPU",
    "OS",
    "DXCoreAdapterInfo",
    "SysInfo",
    "enumerate_compute_adapters",
    "format_pdh_luid",
    "get_available_devices",
    "get_device_ep_map",
    "get_ep_device_luid",
    "get_ep_device_map",
    "resolve_check_device_ep",
    "resolve_device",
    "resolve_eps",
]


def __getattr__(name: str) -> Any:
    """Load one public sysinfo symbol without importing unrelated adapters."""
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_path, attr_name = target
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public symbols without resolving them."""
    return sorted(set(globals()) | set(__all__))
