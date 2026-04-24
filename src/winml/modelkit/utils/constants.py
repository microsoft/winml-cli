# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared constants for ModelKit."""

import onnxruntime as ort


# Supported execution providers — derived from sysinfo's authoritative EP→device map.
def _get_supported_eps() -> list[str]:
    from winml.modelkit.sysinfo.device import get_ep_device_map

    return list(get_ep_device_map().keys())


SUPPORTED_EPS = _get_supported_eps()

# EP shorthand aliases (case-insensitive)
EP_ALIASES = {
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "ov": "OpenVINOExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "vitis": "VitisAIExecutionProvider",
    "cpu": "CPUExecutionProvider",
    "dml": "DmlExecutionProvider",
    "tensorrt": "NvTensorRTRTXExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
}

# All accepted EP names (full names + aliases)
ALL_EP_NAMES = list(SUPPORTED_EPS) + list(EP_ALIASES.keys())


def normalize_ep_name(ep: str | None) -> str | None:
    """Normalize EP name from shorthand to full name.

    Converts EP aliases to their full names (case-insensitive).
    If the input is already a full name, returns it unchanged.

    Args:
        ep: Execution provider name (can be full name or alias)

    Returns:
        Full execution provider name, or None if input is None

    Examples:
        >>> normalize_ep_name("qnn")
        'QNNExecutionProvider'
        >>> normalize_ep_name("ov")
        'OpenVINOExecutionProvider'
        >>> normalize_ep_name("QNNExecutionProvider")
        'QNNExecutionProvider'
    """
    if ep is None:
        return None

    # Check if it's already a full name
    if ep in SUPPORTED_EPS:
        return ep

    # Try to find in aliases (case-insensitive)
    ep_lower = ep.lower()
    if ep_lower in EP_ALIASES:
        return EP_ALIASES[ep_lower]

    # Return as-is if not found (let validation catch invalid names)
    return ep


def extract_ep_options(kwargs: dict) -> dict[str, str]:
    """Extract EP-specific options from CLI parameters.

    Collects parameters that start with an EP alias prefix (e.g., 'qnn_', 'ov_')
    and extracts the option name by removing the prefix.

    Args:
        kwargs: Dictionary of CLI parameters

    Returns:
        Dictionary of EP-specific options with prefix removed

    Examples:
        >>> extract_ep_options({'qnn_qairt': '/path', 'other': 'value'})
        {'qairt': '/path'}
        >>> extract_ep_options({'qnn_qairt': '/path', 'qnn_backend': 'htp'})
        {'qairt': '/path', 'backend': 'htp'}
    """
    ep_aliases = list(EP_ALIASES.keys())
    ep_options = {}
    for param_name, param_value in kwargs.items():
        parts = param_name.split("_", 1)
        if param_value is not None and len(parts) == 2 and parts[0] in ep_aliases:
            ep_options[parts[1]] = str(param_value)
    return ep_options


# Supported device types
SUPPORTED_DEVICES = [
    "CPU",
    "GPU",
    "NPU",
]

# TODO: unify casing with SUPPORTED_DEVICES (uppercase) and DEVICE_TO_DEVICE_TYPE keys
SUPPORTED_DEVICES_WITH_AUTO = ["auto", "cpu", "gpu", "npu"]

# Device string to ORT device type mapping
DEVICE_TO_DEVICE_TYPE = {
    "CPU": ort.OrtHardwareDeviceType.CPU,
    "GPU": ort.OrtHardwareDeviceType.GPU,
    "NPU": ort.OrtHardwareDeviceType.NPU,
}

DEVICE_TYPE_TO_DEVICE = {
    ort.OrtHardwareDeviceType.CPU: "CPU",
    ort.OrtHardwareDeviceType.GPU: "GPU",
    ort.OrtHardwareDeviceType.NPU: "NPU",
}
