"""Device detection, prioritization, and EP-aware resolution for ModelKit."""

from __future__ import annotations

import functools
import logging


logger = logging.getLogger(__name__)

# --- EP-to-Device mapping constants ---

# FIXME: This mapping must be hardcoded because the standard ``onnxruntime`` PyPI
# package does not expose an API to query the target device type of an EP.
#
# ORT *does* define ``OrtHardwareDeviceType`` (CPU/GPU/NPU) in the C API and a
# ``get_ep_devices()`` Python helper, but these are currently available **only**
# in the Windows ML build of ORT (Windows 11 25H2+), not the cross-platform
# ``pip install onnxruntime`` package.  The standard Python API offers
# ``get_available_providers()`` / ``get_device()`` which return EP *names* and a
# coarse device string ("CPU"/"GPU") — neither provides a structured
# EP-to-device-category mapping.
#
# Until ``get_ep_devices()`` (or equivalent) lands in the mainline PyPI package,
# we maintain the mapping manually.
#
# Refs:
#   - C API enums: https://onnxruntime.ai/docs/api/c/group___global.html
#   - Windows ML EP selection: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers
#   - Feature request (closed, not planned): https://github.com/microsoft/onnxruntime/issues/20725
#   - EP list: https://onnxruntime.ai/docs/execution-providers/

# EP name -> target device type (all lowercase values)
_EP_DEVICE_MAP: dict[str, str] = {
    # NVIDIA
    "CUDAExecutionProvider": "gpu",
    "TensorrtExecutionProvider": "gpu",
    # AMD
    "MIGraphXExecutionProvider": "gpu",
    "VitisAIExecutionProvider": "npu",
    # Qualcomm
    "QNNExecutionProvider": "npu",
    # Microsoft
    "DmlExecutionProvider": "gpu",
    # Intel
    "OpenVINOExecutionProvider": "npu/gpu/cpu",
    # Always available
    "CPUExecutionProvider": "cpu",
}

# Derived inverse mapping (excludes multi-device EPs like OpenVINO)
_DEVICE_EP_MAP: dict[str, list[str]] = {}
for _ep, _device in _EP_DEVICE_MAP.items():
    if "/" not in _device:
        _DEVICE_EP_MAP.setdefault(_device, []).append(_ep)

# Valid explicit device values
_VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})


def get_ep_device_map() -> dict[str, str]:
    """Return a copy of the EP-to-device mapping.

    Public accessor for the internal ``_EP_DEVICE_MAP``. Use this instead
    of importing the private dict directly.

    Returns:
        Dict mapping EP names to device types (e.g.
        ``{"QNNExecutionProvider": "npu", ...}``).
    """
    return dict(_EP_DEVICE_MAP)


def _get_available_devices() -> list[str]:
    """Return prioritized list of available devices.

    Priority: NPU > GPU > CPU.
    Always includes "cpu" as fallback.
    Uses SysInfo hardware classes for detection.

    This is an internal helper for :func:`resolve_device` and should not
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


def resolve_device(device: str = "auto") -> tuple[str, list[str]]:
    """Resolve target device with EP availability cross-check.

    Args:
        device: "auto", "npu", "gpu", or "cpu".

    Returns:
        (chosen_device, available_devices_list)

    Raises:
        ValueError: If device is not recognized.
    """
    device = device.lower()

    if device != "auto" and device not in _VALID_DEVICES:
        raise ValueError(
            f"Unknown device '{device}'. "
            f"Expected 'auto', 'npu', 'gpu', or 'cpu'."
        )

    available_devices = _get_available_devices()
    available_eps = _get_available_eps()

    if not available_eps:
        logger.warning(
            "No execution providers detected. Falling back to CPU. "
            "Install onnxruntime or Windows App SDK for EP discovery."
        )

    if device == "auto":
        # Walk priority list, pick first device with a matching EP
        for dev in available_devices:
            compatible_eps = _DEVICE_EP_MAP.get(dev, [])
            if any(ep in available_eps for ep in compatible_eps):
                return dev, available_devices
        # Fallback: CPU is always valid
        return "cpu", available_devices

    # Explicit device requested -- warn if no compatible EP
    compatible_eps = _DEVICE_EP_MAP.get(device, [])
    if not any(ep in available_eps for ep in compatible_eps):
        logger.warning(
            "Device '%s' requested but no compatible EP found. "
            "Compatible EPs: %s. Available EPs: %s",
            device,
            compatible_eps,
            sorted(available_eps),
        )
    return device, available_devices
