# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from .device import resolve_device_category
from .hardware import CPU, GPU, NPU
from .software import OS
from .sysinfo import SysInfo


__all__ = [
    "CPU",
    "GPU",
    "NPU",
    "OS",
    "SysInfo",
    "resolve_device_category",
]
