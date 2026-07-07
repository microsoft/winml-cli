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

# Device priority for auto-selection. Order is significant — NPU is the
# preferred accelerator, CPU is the safe fallback. Use this whenever an
# ordered device list is needed; ``_VALID_DEVICES`` (below) is only for
# fast membership checks.
_DEVICE_PRIORITY: tuple[str, ...] = ("npu", "gpu", "cpu")
_VALID_DEVICES = frozenset(_DEVICE_PRIORITY)


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


# EPs that exist in ``onnxruntime.get_available_providers()`` but are not yet
# exposed via the new ``get_ep_devices()``/AutoEP machinery. Mapped to the
# canonical device they target so they can still be selected via the legacy
# ``SessionOptions.add_provider`` code path.
_LEGACY_EP_DEVICE_FALLBACK: dict[EPName, str] = {
    "VitisAIExecutionProvider": "npu",  # AMD Phoenix/Strix XDNA NPU
}


@functools.lru_cache(maxsize=1)
def _get_device_ep_map_from_ort() -> dict[str, tuple[EPName, ...]]:
    """Return device -> EPs targeting it, derived from registered ORT EP devices.

    Built from :func:`get_registered_ep_devices` (the authoritative ORT API
    available in the Windows ML build). Single source of truth consumed by
    :func:`_get_available_devices`, :func:`resolve_device`, and
    :func:`resolve_eps`. Cached for the process lifetime since hardware/EPs
    do not change at runtime.

    Also merges in EPs from :data:`_LEGACY_EP_DEVICE_FALLBACK` that are
    advertised by ``onnxruntime.get_available_providers()`` but not yet
    registered as ``OrtEpDevice`` instances (e.g. ``VitisAIExecutionProvider``
    in ``onnxruntime-vitisai`` 1.23.x).
    """
    result: dict[str, list[EPName]] = {}
    try:
        for ep_device in get_registered_ep_devices():
            device_name = DEVICE_TYPE_TO_DEVICE.get(ep_device.device.type)
            if device_name is not None:
                result.setdefault(device_name.lower(), []).append(ep_device.ep_name)
    except Exception:
        # WARNING (not DEBUG): if ORT is installed but enumeration fails
        # (driver bug, version mismatch, etc.) downstream code sees an empty
        # map and raises "No execution providers detected" — the user needs
        # the root cause visible at default verbosity to act on it.
        logger.warning("Failed to enumerate registered EP devices", exc_info=True)

    # Legacy-API fallback: some EPs (e.g. VitisAI) only register via
    # ``get_available_providers()``, not via ``get_ep_devices()``.
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        for ep_name, device_name in _LEGACY_EP_DEVICE_FALLBACK.items():
            if ep_name in available and ep_name not in result.get(device_name, ()):
                result.setdefault(device_name, []).append(ep_name)
    except Exception:
        logger.debug("Legacy EP fallback enumeration failed", exc_info=True)

    return {dev: tuple(eps) for dev, eps in result.items()}


@functools.lru_cache(maxsize=1)
def _get_available_devices() -> tuple[str, ...]:
    """Return prioritized tuple of available devices (cached).

    Derived from :func:`_get_device_ep_map`; only device types with at least
    one registered EP appear. Priority order: NPU > GPU > CPU.

    Returns:
        Tuple like ("npu", "gpu", "cpu") with only available devices.
    """
    device_ep_map = _get_device_ep_map_from_ort()
    return tuple(d for d in _DEVICE_PRIORITY if d in device_ep_map)


@functools.lru_cache(maxsize=1)
def _get_available_eps() -> frozenset[EPName]:
    """Return all EPs registered with ORT EP devices (cached).

    Derived from :func:`_get_device_ep_map_from_ort` so EP-availability checks
    and error messages stay consistent — see the comment block at the top of
    this module. Earlier implementations also queried WinMLEPRegistry and
    ``ort.get_available_providers()``, but those can disagree with
    ``ort.get_ep_devices()`` and produced contradictory diagnostics
    ("EP X not available" while X appeared in the listed set).
    """
    return frozenset(ep for eps in _get_device_ep_map_from_ort().values() for ep in eps)


def resolve_device(
    device: str,
    *,
    ep: EPNameOrAlias | None,
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

    device_ep_map = dict(_get_device_ep_map_from_ort())

    if ep is not None:
        ep_full = normalize_ep_name(ep)
        if ep_full not in EP_SUPPORTED_DEVICES:
            raise ValueError(f"Unknown EP '{ep}'. Expected one of: {sorted(EP_SUPPORTED_DEVICES)}")
        # Static policy gate (see issue #860): reject EP/device combos the EP
        # architecture cannot support, before consulting runtime ORT availability.
        if device != "auto" and device not in EP_SUPPORTED_DEVICES[ep_full]:
            raise ValueError(
                f"EP '{ep}' does not support device '{device}'. "
                f"Supported devices: {', '.join(EP_SUPPORTED_DEVICES[ep_full])}."
            )
        device_ep_map = {dev: (ep_full,) for dev, eps in device_ep_map.items() if ep_full in eps}
        if not device_ep_map:
            raise ValueError(
                f"Requested EP '{ep}' is not available on this system. "
                f"Available EPs: {sorted(_get_available_eps())}."
            )

    if not device_ep_map:
        raise RuntimeError("No execution providers detected.")

    available_devices = [d for d in _DEVICE_PRIORITY if d in device_ep_map]

    if device == "auto":
        chosen = available_devices[0]
        logger.info(
            "Auto-selected device '%s' with compatible EPs: %s for auto device",
            chosen,
            sorted(device_ep_map[chosen]),
        )
        return chosen, available_devices

    # Explicit device requested -- raise if no compatible EP is available.
    if device not in device_ep_map:
        raise ValueError(
            f"Device '{device}' requested but no compatible EP is available. "
            f"Compatible EPs: {_DEVICE_EP_MAP[device]}. "
            f"Available EPs: {sorted(_get_available_eps())}."
        )
    return device, available_devices


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
    device = resolved_device.lower()
    available_eps = set(_get_device_ep_map_from_ort().get(device, ()))
    return [ep for ep in _DEVICE_EP_MAP.get(device, []) if ep in available_eps]


def resolve_check_device_ep(
    *, device: str, ep: EPNameOrAlias | None
) -> tuple[str, list[str], list[EPName]]:
    """Resolve or check that the requested device and/or EP combination is valid, raising if not.

    Ideal for commands that do not need the device + ep actually exists on the system.

    Args:
        device: "auto", "npu", "gpu", or "cpu".
        ep: Optional EP short name (e.g., "qnn", "dml"). When set,
            availability is checked and an error is raised if no compatible EP
            is found.

    Raises:
        ValueError: If the requested device or EP combination is not valid.

    Returns:
    Tuple of (resolved_device, available_devices, available_eps) where:
    - resolved_device: The device that should be used based on the input parameters.
    - available_devices: List of devices that are compatible with the first in available_eps
    - available_eps: List of EPs that are compatible with the resolved device.
    """
    ep_name = normalize_ep_name(ep)
    if device == "auto" or ep_name is None:
        resolved_device, _ = resolve_device(device=device, ep=ep_name)
        available_eps: list[EPName] = resolve_eps(resolved_device) if ep_name is None else [ep_name]
        supported_devices = EP_SUPPORTED_DEVICES[available_eps[0]]
        return resolved_device, list(supported_devices), available_eps

    if ep_name not in EP_SUPPORTED_DEVICES:
        raise ValueError(f"Unknown EP '{ep}'. Expected one of: {sorted(EP_SUPPORTED_DEVICES)}")
    supported_devices = EP_SUPPORTED_DEVICES[ep_name]
    if device.lower() not in supported_devices:
        raise ValueError(
            f"EP '{ep}' does not support device '{device}'. "
            f"Supported devices for {ep_name}: {', '.join(supported_devices)}."
        )
    return device.lower(), list(supported_devices), [ep_name]
