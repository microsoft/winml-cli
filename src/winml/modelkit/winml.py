# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .utils.constants import EPName


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

        self._fix_winrt_runtime()
        import winui3.microsoft.windows.ai.machinelearning as winml
        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
            InitializeOptions,
            initialize,
        )

        self._win_app_sdk_handle = initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI)
        self._win_app_sdk_handle.__enter__()
        catalog = winml.ExecutionProviderCatalog.get_default()
        self._providers = catalog.find_all_providers()
        self._ep_paths: dict[str, str] = {}
        for provider in self._providers:
            provider.ensure_ready_async().get()
            if provider.library_path == "":
                continue
            self._ep_paths[provider.name] = provider.library_path
        self._registered_eps: dict[str, list[str]] = {
            "onnxruntime": [],
            "onnxruntime_genai": [],
        }

    def __del__(self) -> None:
        """Clean up WinML resources."""
        self._providers = None
        self._win_app_sdk_handle.__exit__(None, None, None)

    def _fix_winrt_runtime(self) -> None:
        """This function removes the msvcp140.dll from the winrt-runtime package.

        So it does not cause issues with other libraries.
        """
        from importlib import metadata

        site_packages_path = Path(str(metadata.distribution("winrt-runtime").locate_file("")))
        dll_path = site_packages_path / "winrt" / "msvcp140.dll"
        if dll_path.exists():
            dll_path.unlink()

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
                        traceback.print_exc()
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
    ep_name: EPName,
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
