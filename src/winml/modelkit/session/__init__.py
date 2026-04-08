# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLSession - ONNX Runtime session manager with WinML EP integration."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
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
    "EPMonitor",
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
]

_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "WinMLEPRegistry": (".ep_registry", "WinMLEPRegistry"),
    "EPMonitor": (".monitor.ep_monitor", "EPMonitor"),
    "NullEPMonitor": (".monitor.ep_monitor", "NullEPMonitor"),
    "HWMonitor": (".monitor.hw_monitor", "HWMonitor"),
    "OpenVinoMonitor": (".monitor.openvino_monitor", "OpenVinoMonitor"),
    "QNNMonitor": (".monitor.qnn_monitor", "QNNMonitor"),
    "VitisAIMonitor": (".monitor.vitisai_monitor", "VitisAIMonitor"),
    "WinMLQairtSession": (".qairt.qairt_session", "WinMLQairtSession"),
    "InferenceError": (".session", "InferenceError"),
    "SessionState": (".session", "SessionState"),
    "WinMLSession": (".session", "WinMLSession"),
    "PerfStats": (".stats", "PerfStats"),
}


def __getattr__(name: str) -> object:
    entry = _LAZY_IMPORT_MAP.get(name)
    if entry is not None:
        module_path, attr_name = entry
        mod = importlib.import_module(module_path, __name__)
        attr = getattr(mod, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
