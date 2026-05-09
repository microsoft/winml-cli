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

    from ..models.ihv_type import IHVType


logger = logging.getLogger(__name__)


def infer_ihv_from_ep_name(ep_name: str) -> IHVType:
    """Infer IHVType from Execution Provider name.

    Maps an execution provider name to its corresponding IHV type.
    Supports multiple name variations for each provider.

    Args:
        ep_name: Execution Provider name (e.g., QNNExecutionProvider, OpenVINOExecutionProvider)

    Returns:
        IHVType: Inferred IHV type (QC, INTEL, AMD, or NVIDIA)

    Raises:
        ValueError: If EP name is not recognized

    Examples:
        >>> infer_ihv_from_ep_name("QNNExecutionProvider")
        <IHVType.QC: 'QC'>
        >>> infer_ihv_from_ep_name("OpenVINOExecutionProvider")
        <IHVType.INTEL: 'INTEL'>
        >>> infer_ihv_from_ep_name("VitisAIExecutionProvider")
        <IHVType.AMD: 'AMD'>
        >>> infer_ihv_from_ep_name("NvTensorRTRTXExecutionProvider")
        <IHVType.NVIDIA: 'NVIDIA'>
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

    # VitisAI / MIGraphX / AMD / ACE (AMD)
    amd_keywords = ("amd", "quark", "vitis", "ace", "migraphx")
    if any(kw in ep_lower for kw in amd_keywords):
        return IHVType.AMD

    # NVIDIA / TensorRT RTX
    nvidia_keywords = ("nvidia", "nvtensorrt", "tensorrt", "rtx")
    if any(kw in ep_lower for kw in nvidia_keywords):
        return IHVType.NVIDIA

    raise ValueError(
        f"Unknown execution provider: {ep_name}. "
        "Supported: QNNExecutionProvider, OpenVINOExecutionProvider, "
        "VitisAIExecutionProvider, NvTensorRTRTXExecutionProvider"
    )


def get_devices_with_rule_data(ep_name: str) -> list[str]:
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


def has_rule_data_for_ep(ep_name: str, device: str) -> bool:
    """Check whether runtime check rule data exists for a given EP and device.

        Probes runtime-rule search directories for parquet files in either layout:
        - flat files under search dir:
            ``*_{ep_name}_{device}_*.parquet``
        - provider subdirectory layout:
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

    def _has_parquet_in_search_dir(search_dir: Path, ep: str, device_upper: str) -> bool:
        provider_dir = search_dir / f"{ep}_{device_upper}"
        if provider_dir.is_dir() and any(provider_dir.glob("*.parquet")):
            return True

        if any(search_dir.glob(f"*_{ep}_{device_upper}_*.parquet")):
            return True

        return any(search_dir.glob(f"{ep}_{device_upper}_*.parquet"))

    device_upper = device.upper()
    for search_dir in get_runtime_rules_search_dirs():
        if not search_dir.is_dir():
            continue
        if _has_parquet_in_search_dir(search_dir, ep_name, device_upper):
            return True
    return False
