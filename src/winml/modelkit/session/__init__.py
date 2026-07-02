# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - ONNX Runtime session manager with WinML EP integration."""

from ..ep_path import DirectorySource, EPEntry
from .ep_device import (
    DEVICE_TO_DEVICE_TYPE,
    DEVICE_TYPE_TO_DEVICE,
    EP_DEVICE_SPECS,
    VALID_DEVICES,
    VALID_EPS,
    DeviceNotFound,
    EPDeviceSpec,
    EPDeviceTarget,
    UnknownListingPick,
    WinMLDevice,
    WinMLEPMonitorMismatch,
    WinMLEPNotDiscovered,
    WinMLEPRegistrationFailed,
    _ep_short_or_none,
    auto_detect_device,
    default_device_for_ep,
    default_ep_for_device,
    ep_to_device,
    eps_for_device,
    expand_ep_name,
    known_ep_short_names,
    lookup_device_spec,
    resolve_device,
    short_ep_name,
)
from .ep_registry import WinMLEP, WinMLEPDevice, WinMLEPRegistry
from .monitor.ep_monitor import NullEPMonitor, WinMLEPMonitor
from .monitor.hw_monitor import HWMonitor
from .monitor.openvino_monitor import OpenVINOMonitor
from .monitor.qnn_monitor import QNNMonitor
from .monitor.vitisai_monitor import VitisAIMonitor
from .qairt.qairt_session import WinMLQairtSession
from .session import InferenceError, SessionState, WinMLSession
from .stats import PerfStats


__all__ = [
    "DEVICE_TO_DEVICE_TYPE",
    "DEVICE_TYPE_TO_DEVICE",
    "EP_DEVICE_SPECS",
    "VALID_DEVICES",
    "VALID_EPS",
    "DeviceNotFound",
    "DirectorySource",
    "EPDeviceSpec",
    "EPDeviceTarget",
    "EPEntry",
    "HWMonitor",
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
    "default_device_for_ep",
    "default_ep_for_device",
    "ep_to_device",
    "eps_for_device",
    "expand_ep_name",
    "known_ep_short_names",
    "lookup_device_spec",
    "resolve_device",
    "short_ep_name",
]
