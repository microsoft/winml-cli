# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Device detection, prioritization, and EP-aware resolution for WinML CLI."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

from ..utils.constants import DEVICE_TYPE_TO_DEVICE, EP_SUPPORTED_DEVICES, EPName, normalize_ep_name
from ..winml import get_registered_ep_devices


if TYPE_CHECKING:
    from ..utils.constants import EPNameOrAlias

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

# Back-compat shim: EP name -> ``/``-joined device string. This format is
# the legacy public contract returned by :func:`get_ep_device_map`; new code
# should consume :data:`~winml.modelkit.utils.constants.EP_SUPPORTED_DEVICES`
# (tuple form) directly.
_EP_DEVICE_MAP: dict[EPName, str] = {
    ep: "/".join(devices) for ep, devices in EP_SUPPORTED_DEVICES.items()
}

# Derived inverse mapping (multi-device EPs are listed under each device)
_DEVICE_EP_MAP: dict[str, list[EPName]] = {}
for _ep, _devices in EP_SUPPORTED_DEVICES.items():
    for _d in _devices:
        _DEVICE_EP_MAP.setdefault(_d, []).append(_ep)

# Valid explicit device values - sort by preference
_VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})


def get_ep_device_map() -> dict[EPName, str]:
    """Return a copy of the EP-to-device mapping in legacy string form.

    Each value is a ``/``-joined string of supported device names (e.g.
    ``"npu/gpu"``). New code should prefer
    :data:`~winml.modelkit.utils.constants.EP_SUPPORTED_DEVICES` directly.

    Returns:
        Dict mapping EP names to device types (e.g.
        ``{"QNNExecutionProvider": "npu/gpu", ...}``).
    """
    return dict(_EP_DEVICE_MAP)


def get_device_ep_map() -> dict[str, list[EPName]]:
    """Return a copy of the device-to-EP mapping.

    Public accessor for the internal ``_DEVICE_EP_MAP``. Each device key
    maps to the EPs that target it, in priority order (most powerful EP
    first), derived from ``_EP_DEVICE_MAP``'s declaration order.

    Returns:
        Dict mapping device types to ordered EP-name lists (e.g.
        ``{"gpu": ["NvTensorRTRTXExecutionProvider", ...], ...}``).
    """
    return {device: list(eps) for device, eps in _DEVICE_EP_MAP.items()}


@functools.lru_cache(maxsize=1)
def _get_available_devices() -> tuple[str, ...]:
    """Return prioritized tuple of available devices (cached).

    Aggregates device types advertised by registered ORT EP devices and
    filters against the supported set in priority order: NPU > GPU > CPU.

    Returns:
        Tuple like ("npu", "gpu", "cpu") with only available devices.
    """
    from ..utils.constants import DEVICE_TYPE_TO_DEVICE

    available: set[str] = {}

    try:
        for ep_device in get_registered_ep_devices():
            device_name = DEVICE_TYPE_TO_DEVICE.get(ep_device.device.type)
            if device_name is not None:
                available.add(device_name.lower())
    except Exception:
        logger.debug("Failed to enumerate registered EP devices", exc_info=True)

    return tuple(d for d in ("npu", "gpu", "cpu") if d in available)


@functools.lru_cache(maxsize=1)
def _get_available_eps() -> frozenset[EPName]:
    """Collect available EP names from WinML and ORT (cached).

    Hardware and EPs do not change during a process lifetime,
    so this result is cached via lru_cache.

    Returns:
        Frozenset of available EP name strings.
    """
    available_eps: set[EPName] = set()

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


def resolve_device(
    device: str = "auto",
    *,
    ep: EPNameOrAlias | None = None,
) -> tuple[str, list[str]]:
    """Resolve target device with EP availability cross-check.

    Args:
        device: "auto", "npu", "gpu", or "cpu".
        ep: Optional EP short name (e.g., "qnn", "dml"). When set,
            ``available_devices`` is filtered to only those device types the
            EP can target, and ``available_eps`` is filtered to just this EP
            (intersected with what is actually available on the system).

    Returns:
        (chosen_device, available_devices_list)

    Raises:
        ValueError: If ``device`` or ``ep`` is not recognized, or if an
            explicit ``device`` (non-``auto``) is requested but no EP
            compatible with it is currently available.
    """
    device = device.lower()

    if device != "auto" and device not in _VALID_DEVICES:
        raise ValueError(f"Unknown device '{device}'. Expected 'auto', 'npu', 'gpu', or 'cpu'.")

    device_ep_map: dict[str, list[EPName]] = {}
    for ep_device in get_registered_ep_devices():
        device_name = DEVICE_TYPE_TO_DEVICE.get(ep_device.device.type)
        if device_name is not None:
            device_ep_map.setdefault(device_name.lower(), []).append(ep_device.ep_name)

    if ep is not None:
        ep_full = normalize_ep_name(ep)
        if ep_full not in EP_SUPPORTED_DEVICES:
            raise ValueError(f"Unknown EP '{ep}'. Expected one of: {sorted(EP_SUPPORTED_DEVICES)}")
        device_ep_map = {dev: [ep_full] for dev, eps in device_ep_map.items() if ep_full in eps}
        if not device_ep_map:
            raise ValueError(
                f"Requested EP '{ep}' is not available on this system. "
                f"Available EPs: {sorted(_get_available_eps())}."
            )

    if device == "auto":
        # Walk priority list, pick first device with a matching EP
        for dev in _VALID_DEVICES:
            if dev in device_ep_map:
                logger.info(
                    "Auto-selected device '%s' with compatible EPs: %s for auto device",
                    dev,
                    sorted(device_ep_map[dev]),
                )
                return dev, [dev for dev in _VALID_DEVICES if dev in device_ep_map]

    # Explicit device requested -- raise if no compatible EP is available.
    if device not in device_ep_map:
        raise ValueError(
            f"Device '{device}' requested but no EP is available. "
            f"Available EPs: {sorted(_get_available_eps())}."
        )
    return device, [dev for dev in _VALID_DEVICES if dev in device_ep_map]


def resolve_eps(resolved_device: str) -> list[EPName]:
    """Return list of available EPs compatible with the given device.

    Args:
        resolved_device: Concrete device name (``"npu"``, ``"gpu"``, or
            ``"cpu"``). Case-insensitive; ``"NPU"`` is accepted. An unknown
            value returns an empty list rather than raising.

    Returns:
        EPs from ``_DEVICE_EP_MAP[device]`` that are also currently
        advertised by ORT/WinML, in ``_DEVICE_EP_MAP`` priority order.
    """
    available_eps = set()
    for ep_device in get_registered_ep_devices():
        device_name = DEVICE_TYPE_TO_DEVICE.get(ep_device.device.type)
        if device_name is not None and device_name.lower() == resolved_device.lower():
            available_eps.add(ep_device.ep_name)
    return [ep for ep in _DEVICE_EP_MAP.get(resolved_device.lower(), []) if ep in available_eps]
