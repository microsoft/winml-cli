# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Lazy public facade for WinML session backends and monitors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
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


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "VALID_SOURCE_TAGS": ("..ep_path", "VALID_SOURCE_TAGS"),
    "DirectorySource": ("..ep_path", "DirectorySource"),
    "EPEntry": ("..ep_path", "EPEntry"),
    "DEVICE_TO_DEVICE_TYPE": (".ep_device", "DEVICE_TO_DEVICE_TYPE"),
    "DEVICE_TYPE_TO_DEVICE": (".ep_device", "DEVICE_TYPE_TO_DEVICE"),
    "EP_DEVICE_SPECS": (".ep_device", "EP_DEVICE_SPECS"),
    "VALID_DEVICES": (".ep_device", "VALID_DEVICES"),
    "VALID_EPS": (".ep_device", "VALID_EPS"),
    "DeviceNotFound": (".ep_device", "DeviceNotFound"),
    "EPDeviceSpec": (".ep_device", "EPDeviceSpec"),
    "EPDeviceTarget": (".ep_device", "EPDeviceTarget"),
    "UnknownListingPick": (".ep_device", "UnknownListingPick"),
    "WinMLDevice": (".ep_device", "WinMLDevice"),
    "WinMLEPMonitorMismatch": (".ep_device", "WinMLEPMonitorMismatch"),
    "WinMLEPNotDiscovered": (".ep_device", "WinMLEPNotDiscovered"),
    "WinMLEPRegistrationFailed": (".ep_device", "WinMLEPRegistrationFailed"),
    "auto_detect_device": (".ep_device", "auto_detect_device"),
    "available_eps_for_device": (".ep_device", "available_eps_for_device"),
    "default_device_for_ep": (".ep_device", "default_device_for_ep"),
    "default_ep_for_device": (".ep_device", "default_ep_for_device"),
    "device_from_provider_option_hints": (
        ".ep_device",
        "device_from_provider_option_hints",
    ),
    "ep_short_or_none": (".ep_device", "ep_short_or_none"),
    "ep_to_device": (".ep_device", "ep_to_device"),
    "eps_for_device": (".ep_device", "eps_for_device"),
    "expand_ep_name": (".ep_device", "expand_ep_name"),
    "known_ep_short_names": (".ep_device", "known_ep_short_names"),
    "lookup_device_spec": (".ep_device", "lookup_device_spec"),
    "resolve_device": (".ep_device", "resolve_device"),
    "short_ep_name": (".ep_device", "short_ep_name"),
    "WinMLEP": (".ep_registry", "WinMLEP"),
    "WinMLEPDevice": (".ep_registry", "WinMLEPDevice"),
    "WinMLEPRegistry": (".ep_registry", "WinMLEPRegistry"),
    "GenaiLoadError": (".genai_session", "GenaiLoadError"),
    "GenaiNotInstalledError": (".genai_session", "GenaiNotInstalledError"),
    "GenaiSession": (".genai_session", "GenaiSession"),
    "GenaiSessionError": (".genai_session", "GenaiSessionError"),
    "GenerationConfig": (".genai_session", "GenerationConfig"),
    "GenerationTiming": (".genai_session", "GenerationTiming"),
    "EPMonitor": (".monitor.ep_monitor", "EPMonitor"),
    "NullEPMonitor": (".monitor.ep_monitor", "NullEPMonitor"),
    "WinMLEPMonitor": (".monitor.ep_monitor", "WinMLEPMonitor"),
    "HWMonitor": (".monitor.hw_monitor", "HWMonitor"),
    "OpenVinoMonitor": (".monitor.openvino_monitor", "OpenVinoMonitor"),
    "QNNMonitor": (".monitor.qnn_monitor", "QNNMonitor"),
    "VitisAIMonitor": (".monitor.vitisai_monitor", "VitisAIMonitor"),
    "WinMLQairtSession": (".qairt.qairt_session", "WinMLQairtSession"),
    "InferenceError": (".session", "InferenceError"),
    "PerfContext": (".session", "PerfContext"),
    "SessionState": (".session", "SessionState"),
    "WinMLSession": (".session", "WinMLSession"),
    "PerfStats": (".stats", "PerfStats"),
}
_PRIVATE_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "_format_bytes": (".ep_device", "_format_bytes"),
}

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


def __getattr__(name: str) -> Any:
    """Load one public session symbol without importing unrelated backends."""
    target = _LAZY_IMPORTS.get(name) or _PRIVATE_LAZY_IMPORTS.get(name)
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
