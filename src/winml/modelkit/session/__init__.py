# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - ONNX Runtime session manager with WinML EP integration."""

from ..ep_path import VALID_SOURCE_TAGS, DirectorySource, EPEntry
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
    _format_bytes,
    auto_detect_device,
    available_eps_for_device,
    default_device_for_ep,
    default_ep_for_device,
    device_from_provider_option_hints,
    ep_short_or_none,
    ep_to_device,
    eps_for_device,
    expand_ep_name,
    known_ep_short_names,
    lookup_device_spec,
    resolve_device,
    short_ep_name,
)
from .ep_registry import WinMLEP, WinMLEPDevice, WinMLEPRegistry
from .genai_session import (
    GenaiLoadError,
    GenaiNotInstalledError,
    GenaiSession,
    GenaiSessionError,
    GenerationConfig,
    GenerationTiming,
)
from .monitor.ep_monitor import EPMonitor, NullEPMonitor, WinMLEPMonitor
from .monitor.hw_monitor import HWMonitor
from .monitor.openvino_monitor import OpenVinoMonitor
from .monitor.qnn_monitor import QNNMonitor
from .monitor.vitisai_monitor import VitisAIMonitor
from .qairt.qairt_session import WinMLQairtSession
from .session import InferenceError, PerfContext, SessionState, WinMLSession
from .stats import PerfStats


__all__ = [
    "DEVICE_TO_DEVICE_TYPE",
    "DEVICE_TYPE_TO_DEVICE",
    "EP_DEVICE_SPECS",
    "VALID_DEVICES",
    "VALID_EPS",
    "VALID_SOURCE_TAGS",
    "DeviceNotFound",
    "DirectorySource",
    "EPDeviceSpec",
    "EPDeviceTarget",
    "EPEntry",
    "EPMonitor",
    "GenaiLoadError",
    "GenaiNotInstalledError",
    "GenaiSession",
    "GenaiSessionError",
    "GenerationConfig",
    "GenerationTiming",
    "HWMonitor",
    "InferenceError",
    "NullEPMonitor",
    "OpenVinoMonitor",
    "PerfContext",
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
    "available_eps_for_device",
    "default_device_for_ep",
    "default_ep_for_device",
    "device_from_provider_option_hints",
    "ep_short_or_none",
    "ep_to_device",
    "eps_for_device",
    "expand_ep_name",
    "known_ep_short_names",
    "lookup_device_spec",
    "resolve_device",
    "short_ep_name",
]
