# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Device detection, prioritization, and EP-aware resolution for ModelKit."""

from __future__ import annotations

import functools
import logging

from ..session import VALID_DEVICES as _VALID_DEVICES
from ..session import eps_for_device


logger = logging.getLogger(__name__)


def _get_available_devices() -> list[str]:
    """Return prioritized list of available devices.

    Priority: NPU > GPU > CPU.
    Always includes "cpu" as fallback.
    Uses SysInfo hardware classes for detection.

    This is an internal helper for :func:`resolve_device_category` and should not
    be called directly by external code.

    Returns:
        List like ["npu", "gpu", "cpu"] with only available devices.
    """
    devices: list[str] = []

    try:
        from .hardware import NPU

        if NPU.get_all():
            devices.append("npu")
    except Exception:
        logger.debug("NPU detection failed or unavailable")

    try:
        from .hardware import GPU

        if GPU.get_all():
            devices.append("gpu")
    except Exception:
        logger.debug("GPU detection failed or unavailable")

    devices.append("cpu")  # CPU always available
    return devices


@functools.lru_cache(maxsize=1)
def _get_available_eps() -> frozenset[str]:
    """Collect available EP names from WinML and ORT (cached).

    Hardware and EPs do not change during a process lifetime,
    so this result is cached via lru_cache.

    Returns:
        Frozenset of available EP name strings.
    """
    available_eps: set[str] = set()

    try:
        from ..session.ep_registry import WinMLEPRegistry

        registry = WinMLEPRegistry.get_instance()
        available_eps.update(registry.get_available_eps().keys())
    except (ImportError, RuntimeError):
        pass  # WinML not available
    except Exception:
        logger.warning("Unexpected error during WinML EP discovery", exc_info=True)

    try:
        import onnxruntime as ort

        available_eps.update(ort.get_available_providers())
    except (ImportError, RuntimeError):
        pass  # ORT not installed
    except Exception:
        logger.warning("Unexpected error during ORT EP discovery", exc_info=True)

    return frozenset(available_eps)


def resolve_device_category(device: str = "auto") -> tuple[str, list[str]]:
    """Resolve a device hint to (category, candidate EP names).

    Args:
        device: "auto", "npu", "gpu", or "cpu".

    Returns:
        (chosen_device, available_devices_list)

    Raises:
        ValueError: If device is not recognized.
    """
    device = device.lower()

    if device != "auto" and device not in _VALID_DEVICES:
        raise ValueError(f"Unknown device '{device}'. Expected 'auto', 'npu', 'gpu', or 'cpu'.")

    available_devices = _get_available_devices()
    available_eps = _get_available_eps()

    if not available_eps:
        logger.warning(
            "No execution providers detected. Falling back to CPU. "
            "Install onnxruntime or Windows App SDK for EP discovery."
        )

    if device == "auto":
        # Walk priority list, pick first device with a matching EP.
        # eps_for_device returns canonical EP names from the catalog —
        # includes OpenVINO for npu/gpu/cpu (the old _DEVICE_EP_MAP excluded it).
        for dev in available_devices:
            if any(ep in available_eps for ep in eps_for_device(dev)):
                return dev, available_devices
        # Fallback: CPU is always valid
        return "cpu", available_devices

    # Explicit device requested -- warn if no compatible EP
    compatible_eps = eps_for_device(device)
    if not any(ep in available_eps for ep in compatible_eps):
        logger.warning(
            "Device '%s' requested but no compatible EP found. "
            "Compatible EPs: %s. Available EPs: %s",
            device,
            sorted(compatible_eps),
            sorted(available_eps),
        )
    return device, available_devices
