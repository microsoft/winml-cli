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
