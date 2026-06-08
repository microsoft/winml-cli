# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - ONNX Runtime session manager with WinML EP integration."""

from .ep_device import (
    EP_DEVICE_SPECS,
    VALID_DEVICES,
    VALID_EPS,
    AmbiguousListingPick,
    AmbiguousMatch,
    DeviceNotFound,
    EPDeviceSpec,
    EPDeviceTarget,
    IncompatibleListingPick,
    UnknownListingPick,
    WinMLEPMonitorMismatch,
    WinMLEPNotDiscovered,
    WinMLEPRegistrationFailed,
    auto_detect_device,
    default_device_for_ep,
    default_ep_for_device,
    ep_to_device,
    eps_for_device,
    expand_ep_name,
    lookup_device_spec,
    resolve_device,
    short_ep_name,
)
from .ep_registry import WinMLEP, WinMLEPDevice, WinMLEPRegistry, available_eps
from .monitor.ep_monitor import NullEPMonitor, WinMLEPMonitor
from .monitor.hw_monitor import HWMonitor
from .monitor.openvino_monitor import OpenVINOMonitor
from .monitor.qnn_monitor import QNNMonitor
from .monitor.vitisai_monitor import VitisAIMonitor
from .qairt.qairt_session import WinMLQairtSession
from .session import InferenceError, SessionState, WinMLSession
from .stats import PerfStats
from .winml_device import WinMLDevice, wrap_ort_device


__all__ = [
    "EP_DEVICE_SPECS",
    "VALID_DEVICES",
    "VALID_EPS",
    "AmbiguousListingPick",
    "AmbiguousMatch",
    "DeviceNotFound",
    "EPDeviceSpec",
    "EPDeviceTarget",
    "HWMonitor",
    "IncompatibleListingPick",
    "InferenceError",
    "NullEPMonitor",
    "OpenVINOMonitor",
    "PerfStats",
    "QNNMonitor",
    "SessionState",
    "UnknownListingPick",
    "VitisAIMonitor",
    "WinMLDevice",
    "WinMLEP",
    "WinMLEPDevice",
    "WinMLEPMonitor",
    "WinMLEPMonitorMismatch",
    "WinMLEPNotDiscovered",
    "WinMLEPRegistrationFailed",
    "WinMLEPRegistry",
    "WinMLQairtSession",
    "WinMLSession",
    "auto_detect_device",
    "available_eps",
    "default_device_for_ep",
    "default_ep_for_device",
    "ep_to_device",
    "eps_for_device",
    "expand_ep_name",
    "lookup_device_spec",
    "resolve_device",
    "short_ep_name",
    "wrap_ort_device",
]
