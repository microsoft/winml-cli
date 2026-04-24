# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Execution Provider utility functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..models.ihv_type import IHVType


logger = logging.getLogger(__name__)


def infer_ihv_from_ep_name(ep_name: str) -> IHVType:
    """Infer IHVType from Execution Provider name.

    Maps an execution provider name to its corresponding IHV type.
    Supports multiple name variations for each provider.

    Args:
        ep_name: Execution Provider name (e.g., QNNExecutionProvider, OpenVINOExecutionProvider)

    Returns:
        IHVType: Inferred IHV type (QC, INTEL, or AMD)

    Raises:
        ValueError: If EP name is not recognized

    Examples:
        >>> infer_ihv_from_ep_name("QNNExecutionProvider")
        <IHVType.QC: 'QC'>
        >>> infer_ihv_from_ep_name("OpenVINOExecutionProvider")
        <IHVType.INTEL: 'INTEL'>
        >>> infer_ihv_from_ep_name("VitisAIExecutionProvider")
        <IHVType.AMD: 'AMD'>
        >>> infer_ihv_from_ep_name("unknown")
        ValueError: Unknown execution provider...
    """
    from ..models.ihv_type import IHVType

    ep_lower = ep_name.lower()

    # QNN / Qualcomm
    if "qnn" in ep_lower or "qualcomm" in ep_lower:
        return IHVType.QC

    # OpenVINO / Intel
    if "openvino" in ep_lower or "intel" in ep_lower:
        return IHVType.INTEL

    # VitisAI / AMD / ACE (AMD)
    if "amd" in ep_lower or "quark" in ep_lower or "vitis" in ep_lower or "ace" in ep_lower:
        return IHVType.AMD

    raise ValueError(
        f"Unknown execution provider: {ep_name}. "
        "Supported: QNNExecutionProvider, OpenVINOExecutionProvider, VitisAIExecutionProvider"
    )


def get_devices_with_rule_data(ep_name: str) -> list[str]:
    """Return all devices supported by an EP.

    First probes rule zip search directories for files matching
    ``{ep_name}_{device}_*.zip``.  If no rule data is found, falls
    back to the EP→device mapping from :func:`sysinfo.get_ep_device_map`.

    Args:
        ep_name: Full execution provider name (e.g., ``"QNNExecutionProvider"``).

    Returns:
        List of device strings (e.g., ``["NPU", "GPU"]``), empty if
        the EP is completely unknown.
    """
    from winml.modelkit.sysinfo.device import get_ep_device_map

    devices = [d for d in ("NPU", "GPU", "CPU") if has_rule_data_for_ep(ep_name, d)]
    if devices:
        return devices
    # Fallback: derive from the authoritative EP→device mapping
    device_str = get_ep_device_map().get(ep_name, "")
    return [d.upper() for d in device_str.split("/") if d]


def has_rule_data_for_ep(ep_name: str, device: str) -> bool:
    """Check whether runtime check rule data exists for a given EP and device.

    Probes the rule zip search directories for any zip file matching the
    naming convention ``{ep_name}_{device}_*.zip``.  This is a fast
    filesystem check — no zip contents are read.

    Args:
        ep_name: Full execution provider name (e.g., ``"QNNExecutionProvider"``).
        device: Device type (e.g., ``"NPU"``, ``"GPU"``, ``"CPU"``).

    Returns:
        ``True`` if at least one rule zip exists for this EP + device pair.
    """
    from .rule_loader import get_runtime_rules_search_dirs

    prefix = f"{ep_name}_{device.upper()}_"
    for search_dir in get_runtime_rules_search_dirs():
        if not search_dir.is_dir():
            continue
        if any(search_dir.glob(f"{prefix}*.zip")):
            return True
    return False
