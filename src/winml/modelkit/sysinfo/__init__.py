# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from .device import get_device_ep_map, get_ep_device_map, resolve_device, resolve_eps
from .hardware import CPU, GPU, NPU
from .software import OS
from .sysinfo import SysInfo


__all__ = [
    "CPU",
    "GPU",
    "NPU",
    "OS",
    "SysInfo",
    "get_device_ep_map",
    "get_ep_device_map",
    "resolve_device",
    "resolve_eps",
]
