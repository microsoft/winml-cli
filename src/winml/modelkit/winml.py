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


def register_execution_providers(ort: bool = True, ort_genai: bool = False) -> dict[str, list[str]]:
    """Register WinML execution providers for ONNX Runtime and ONNX Runtime GenAI.

    Args:
        ort (bool): Whether to register for ONNX Runtime.
        ort_genai (bool): Whether to register for ONNX Runtime GenAI.

    Returns:
        dict[str, list[str]]: Dictionary of registered execution provider names
        by module.
    """
    from .session import WinMLEPRegistry

    result = {}
    if ort:
        registry = WinMLEPRegistry.get_instance()
        registered_eps = registry.register_to_ort()
        result["onnxruntime"] = registered_eps
    if ort_genai:
        raise NotImplementedError("ONNX Runtime GenAI support is not yet implemented.")
    return result


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
