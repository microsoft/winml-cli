# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Device detection, prioritization, and EP-aware resolution for WinML CLI."""

from __future__ import annotations

import functools
import logging

from ..session import DEVICE_TYPE_TO_DEVICE
from ..utils.constants import (
    EP_SUPPORTED_DEVICES,
    SUPPORTED_EPS,
    EPName,
)
from ..winml import get_registered_ep_devices


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
            if device_name is None:
                continue
            # ORT reports device-qualified aliases (e.g. "OpenVINOExecutionProvider.AUTO")
            # as distinct ep_name values. ORT's ep_name is always a canonical full
            # name (never a short alias), so a plain SUPPORTED_EPS membership check
            # drops the aliases and keeps downstream availability sets / error
            # messages limited to real providers.
            ep_name = ep_device.ep_name
            if ep_name not in SUPPORTED_EPS:
                continue
            result.setdefault(device_name.lower(), []).append(ep_name)
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


