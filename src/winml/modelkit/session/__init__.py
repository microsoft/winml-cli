# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - ONNX Runtime session manager with WinML EP integration."""

from .ep_device import (
    _VALID_DEVICES,
    EP_DEVICE_SPECS,
    VALID_EPS,
    AmbiguousMatch,
    DeviceNotFound,
    EPDevice,
    EPDeviceSpec,
    EPMonitorMismatch,
    EPNotDiscovered,
    EPRegistrationFailed,
    canonicalize_ep_name,
    default_device_for_ep,
    default_ep_for_device,
    ep_to_device,
    expand_ep_name,
    get_provider_for_device,
    lookup_device_spec,
    resolve_device,
    short_ep_name,
)
from .ep_registry import WinMLEPRegistry
from .monitor.ep_monitor import EPMonitor, NullEPMonitor
from .monitor.hw_monitor import HWMonitor
from .monitor.openvino_monitor import OpenVinoMonitor
from .monitor.qnn_monitor import QNNMonitor
from .monitor.vitisai_monitor import VitisAIMonitor
from .qairt.qairt_session import WinMLQairtSession
from .session import InferenceError, SessionState, WinMLSession
from .stats import PerfStats


__all__ = [
    "EP_DEVICE_SPECS",
    "VALID_EPS",
    "_VALID_DEVICES",
    "AmbiguousMatch",
    "DeviceNotFound",
    "EPDevice",
    "EPDeviceSpec",
    "EPMonitor",
    "EPMonitorMismatch",
    "EPNotDiscovered",
    "EPRegistrationFailed",
    "HWMonitor",
    "InferenceError",
    "NullEPMonitor",
    "OpenVinoMonitor",
    "PerfStats",
    "QNNMonitor",
    "SessionState",
    "VitisAIMonitor",
    "WinMLEPRegistry",
    "WinMLQairtSession",
    "WinMLSession",
    "canonicalize_ep_name",
    "default_device_for_ep",
    "default_ep_for_device",
    "ep_to_device",
    "expand_ep_name",
    "get_provider_for_device",
    "lookup_device_spec",
    "resolve_device",
    "short_ep_name",
]
