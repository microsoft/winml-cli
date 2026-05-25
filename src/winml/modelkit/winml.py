# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .utils.constants import EPName


logger = logging.getLogger(__name__)

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
        """Initialize WinML execution provider catalog."""
        if self._initialized:
            return
        self._initialized = True

        from windowsml import EpCatalog

        self._catalog = EpCatalog()
        self._providers = self._catalog.find_all_providers()
        self._ep_paths: dict[str, str] = {}
        for provider in self._providers:
            provider.ensure_ready()
            if provider.library_path == "":
                continue
            self._ep_paths[provider.name] = provider.library_path
        self._registered_eps: dict[str, list[str]] = {
            "onnxruntime": [],
            "onnxruntime_genai": [],
        }

        # Workaround: WinMLEpCatalogRelease (called by EpCatalog.close() /
        # EpCatalog.__del__) crashes with ACCESS_VIOLATION (0xC0000005) on some
        # QNN NPU driver configurations — a Windows SEH exception that Python
        # try/except cannot catch.  All provider paths have been extracted
        # above, so the catalog handle is no longer needed.  Null it out
        # immediately so that EpCatalog.close() becomes a no-op for the
        # remainder of the process lifetime, whether invoked from a background
        # thread or interpreter shutdown.  Native resources are reclaimed by
        # the OS when the process exits.
        # TODO: Remove once windowsml fixes WinMLEpCatalogRelease to be safe
        # during process teardown on all QNN NPU driver configurations.
        if hasattr(self._catalog, "_handle"):
            self._catalog._handle = None

    def __del__(self) -> None:
        """Clean up WinML resources."""
        self._providers = None
        self._catalog = None

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
                    except Exception:
                        logger.exception(
                            "Failed to register %s for module %s",
                            name,
                            module.__name__,
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


@functools.lru_cache(maxsize=1)
def get_registered_ep_devices() -> tuple[Any, ...]:
    """Return ORT EP devices after ensuring WinML EPs are registered.

    This helper centralizes the common sequence used by callers that need the
    authoritative autoEP device list from ``onnxruntime.get_ep_devices()``.

    Returns a tuple (not a list) because the result is cached via lru_cache —
    a mutable container would let callers silently poison the cache for
    every subsequent caller in the process.
    """
    import onnxruntime as ort

    register_execution_providers(ort=True)
    return tuple(ort.get_ep_devices())


def add_ep_for_device(
    session_options: Any,
    ep_name: EPName,
    device_type: Any,
    ep_options: dict | None = None,
) -> bool:
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
            logger.info("Adding %s for %s", ep_name, device_type)
            session_options.add_provider_for_devices(
                [ep_device], {} if ep_options is None else ep_options
            )
            return True
    return False
