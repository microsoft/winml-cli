# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Plugin-style ONNX Runtime execution provider registration.

Resolves EP plugin DLLs from their pip-installed distribution packages
(``onnxruntime-ep-openvino``, ``onnxruntime-qnn``, ...) and registers them
with ONNX Runtime via ``register_execution_provider_library()`` (added in
ORT 1.24). Built-in EPs (CPU, DML in 1.24+) are registered automatically
by ORT and are not listed here.
"""
from __future__ import annotations

import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def _qnn_arch_dir() -> str:
    """Return the QNN libs subdirectory matching the current architecture."""
    return "arm64ec" if platform.machine().lower() in ("arm64", "aarch64") else "amd64"


# EP name -> (PyPI distribution, relative DLL path under site-packages).
EP_PLUGIN_REGISTRY: dict[str, tuple[str, str]] = {
    "OpenVINOExecutionProvider": (
        "onnxruntime-ep-openvino",
        "onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll",
    ),
    "QNNExecutionProvider": (
        "onnxruntime-qnn",
        f"onnxruntime_qnn/libs/{_qnn_arch_dir()}/onnxruntime_providers_qnn.dll",
    ),
}


def resolve_plugin_dll(ep_name: str) -> Path | None:
    """Resolve the absolute DLL path for a plugin EP, or None if unavailable."""
    entry = EP_PLUGIN_REGISTRY.get(ep_name)
    if entry is None:
        return None
    pkg, rel = entry
    try:
        dist = metadata.distribution(pkg)
    except metadata.PackageNotFoundError:
        return None
    path = Path(str(dist.locate_file(rel)))
    return path if path.exists() else None


_winml_instance: WinML | None = None


class WinML:
    """Singleton class for managing WinML execution providers."""

    _initialized: bool

    def __new__(cls, *args: Any, **kwargs: Any) -> WinML:
        """Create or return the singleton instance."""
        global _winml_instance
        if _winml_instance is None:
            _winml_instance = super().__new__(cls, *args, **kwargs)
            _winml_instance._initialized = False
        return _winml_instance

    def __init__(self) -> None:
        """Initialize WinML execution provider catalog from installed plugin packages."""
        if self._initialized:
            return
        self._initialized = True

        self._ep_paths: dict[str, str] = {}
        for ep_name in EP_PLUGIN_REGISTRY:
            path = resolve_plugin_dll(ep_name)
            if path is not None:
                self._ep_paths[ep_name] = str(path)

        self._registered_eps: dict[str, list[str]] = {
            "onnxruntime": [],
            "onnxruntime_genai": [],
        }

    def register_execution_providers(
        self, ort: bool = True, ort_genai: bool = False
    ) -> dict[str, list[str]]:
        """Register WinML execution providers for ONNX Runtime modules.

        Args:
            ort: Whether to register for ONNX Runtime.
            ort_genai: Whether to register for ONNX Runtime GenAI.

        Returns:
            Dictionary of registered execution provider names by module.
        """
        modules = []
        if ort:
            import onnxruntime

            modules.append(onnxruntime)
        if ort_genai:
            import onnxruntime_genai  # type: ignore[import-not-found]

            modules.append(onnxruntime_genai)
        for name, path in self._ep_paths.items():
            for module in modules:
                if name not in self._registered_eps[module.__name__]:
                    try:
                        module.register_execution_provider_library(name, path)
                        self._registered_eps[module.__name__].append(name)
                    except Exception as e:
                        print(
                            f"Failed to register execution provider {name}: {e}",
                            file=sys.stderr,
                        )
        return self._registered_eps


def register_execution_providers(ort: bool = True, ort_genai: bool = False) -> dict[str, list[str]]:
    """Register WinML execution providers for ONNX Runtime and ONNX Runtime GenAI.

    Args:
        ort (bool): Whether to register for ONNX Runtime.
        ort_genai (bool): Whether to register for ONNX Runtime GenAI.

    Returns:
        dict[str, list[str]]: Dictionary of registered execution provider names
        by module.
    """
    return WinML().register_execution_providers(ort=ort, ort_genai=ort_genai)


def add_ep_for_device(
    session_options: Any,
    ep_name: str,
    device_type: Any,
    ep_options: dict | None = None,
) -> None:
    """Ensures correct EP device selection for WinML. NEVER modify this function.

    ep_name is one of:
        - "CPUExecutionProvider"
        - "DmlExecutionProvider"
        - "WebGpuExecutionProvider"
        - "QNNExecutionProvider"
        - "OpenVINOExecutionProvider"
        - "VitisAIExecutionProvider"
        - "NvTensorRTRTXExecutionProvider"

    device_type is one of:
        - ort.OrtHardwareDeviceType.CPU
        - ort.OrtHardwareDeviceType.GPU
        - ort.OrtHardwareDeviceType.NPU
    """
    import onnxruntime as ort

    ep_devices = ort.get_ep_devices()
    for ep_device in ep_devices:
        if ep_device.ep_name == ep_name and ep_device.device.type == device_type:
            print(f"Adding {ep_name} for {device_type}")
            session_options.add_provider_for_devices(
                [ep_device], {} if ep_options is None else ep_options
            )
            break
