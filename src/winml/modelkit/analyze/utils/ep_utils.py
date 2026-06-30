# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Execution Provider utility functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

    from ...utils.constants import EPName
    from ..models.ihv_type import IHVType


logger = logging.getLogger(__name__)


def infer_ihv_from_ep_name(ep_name: EPName) -> IHVType:
    """Infer IHVType from a canonical Execution Provider name.

    ``EPName`` is a closed set of canonical EP names, so this is a direct,
    exact lookup covering every member of that set.

    Args:
        ep_name: Canonical Execution Provider name (see ``utils.constants.EPName``).

    Returns:
        IHVType: Inferred IHV type (QC, INTEL, AMD, NVIDIA, or MICROSOFT).

    Raises:
        ValueError: If ``ep_name`` is not a known canonical EP name.

    Examples:
        >>> infer_ihv_from_ep_name("QNNExecutionProvider")
        <IHVType.QC: 'QC'>
        >>> infer_ihv_from_ep_name("OpenVINOExecutionProvider")
        <IHVType.INTEL: 'INTEL'>
        >>> infer_ihv_from_ep_name("VitisAIExecutionProvider")
        <IHVType.AMD: 'AMD'>
        >>> infer_ihv_from_ep_name("NvTensorRTRTXExecutionProvider")
        <IHVType.NVIDIA: 'NVIDIA'>
        >>> infer_ihv_from_ep_name("CPUExecutionProvider")
        <IHVType.MICROSOFT: 'Microsoft'>
    """
    from ..models.ihv_type import IHVType

    ep_name_to_ihv: dict[EPName, IHVType] = {
        "QNNExecutionProvider": IHVType.QC,
        "OpenVINOExecutionProvider": IHVType.INTEL,
        "VitisAIExecutionProvider": IHVType.AMD,
        "MIGraphXExecutionProvider": IHVType.AMD,
        "NvTensorRTRTXExecutionProvider": IHVType.NVIDIA,
        "CUDAExecutionProvider": IHVType.NVIDIA,
        "CPUExecutionProvider": IHVType.MICROSOFT,
        "DmlExecutionProvider": IHVType.MICROSOFT,
    }

    try:
        return ep_name_to_ihv[ep_name]
    except KeyError:
        raise ValueError(f"Cannot infer IHV for unknown EP name: {ep_name!r}") from None


def get_devices_with_rule_data(ep_name: EPName) -> list[str]:
    """Return all devices supported by an EP.

    First probes runtime-rule directories for parquet artifacts for each
    ``EP + device`` pair. If no rule data is found, falls
    back to the EP→device mapping from :func:`sysinfo.get_ep_device_map`.

    Args:
        ep_name: Full execution provider name (e.g., ``"QNNExecutionProvider"``).

    Returns:
        List of device strings (e.g., ``["NPU", "GPU"]``), empty if
        the EP is completely unknown.
    """
    from ...sysinfo.device import get_ep_device_map

    # Priority order: NPU > GPU > CPU (first match used as default device)
    known_devices = {d.upper() for v in get_ep_device_map().values() for d in v.split("/") if d}
    priority = ["NPU", "GPU", "CPU"]
    probe_order = [d for d in priority if d in known_devices]
    # Append any devices not in the priority list
    probe_order.extend(d for d in sorted(known_devices) if d not in priority)

    devices = [d for d in probe_order if has_rule_data_for_ep(ep_name, d)]
    if devices:
        return devices
    # Fallback: derive from the authoritative EP→device mapping
    device_str = get_ep_device_map().get(ep_name, "")
    return [d.upper() for d in device_str.split("/") if d]


def has_any_rule_data() -> bool:
    """Return True if any parquet exists under one-level subdirectories.

    Used to distinguish "no data at all" (needs setup) from "data exists
    but not for this specific EP/device combination".

    This intentionally uses a minimal filesystem probe pattern.
    """
    from .rule_loader import get_runtime_rules_search_dirs

    for search_dir in get_runtime_rules_search_dirs():
        if not search_dir.is_dir():
            continue

        if any(search_dir.glob("*/*.parquet")):
            return True

    return False


def has_rule_data_for_ep(ep_name: EPName, device: str) -> bool:
    """Check whether runtime check rule data exists for a given EP and device.

        Probes runtime-rule search directories for provider subdirectory layout only:
            ``<search_dir>/{ep_name}_{device}/*.parquet``

        This is a fast filesystem check and does not parse parquet contents.

    Args:
        ep_name: Full execution provider name (e.g., ``"QNNExecutionProvider"``).
        device: Device type (e.g., ``"NPU"``, ``"GPU"``, ``"CPU"``).

    Returns:
        ``True`` if at least one matching parquet rule file exists for
        this EP + device pair.
    """
    from .rule_loader import get_runtime_rules_search_dirs

    def _has_parquet_in_search_dir(search_dir: Path, ep: EPName, device_upper: str) -> bool:
        provider_dir = search_dir / f"{ep}_{device_upper}"
        return provider_dir.is_dir() and any(provider_dir.glob("*.parquet"))

    device_upper = device.upper()
    for search_dir in get_runtime_rules_search_dirs():
        if not search_dir.is_dir():
            continue
        if _has_parquet_in_search_dir(search_dir, ep_name, device_upper):
            return True
    return False
