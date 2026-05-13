# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - ONNX Runtime session manager with WinML EP integration."""

from .ep_device import (
    _VALID_DEVICES,
    VALID_EPS,
    AmbiguousMatch,
    DeviceNotFound,
    EPDevice,
    EPMonitorMismatch,
    EPNotDiscovered,
    EPRegistrationFailed,
    canonicalize_ep_name,
    ep_to_device,
    expand_ep_name,
    get_provider_for_device,
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
    "VALID_EPS",
    "_VALID_DEVICES",
    "AmbiguousMatch",
    "DeviceNotFound",
    "EPDevice",
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
    "ep_to_device",
    "expand_ep_name",
    "get_provider_for_device",
    "resolve_device",
    "short_ep_name",
]
