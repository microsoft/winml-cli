# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared constants for ModelKit."""

import onnxruntime as ort

from ..session import expand_ep_name


# EP alias prefixes used by extract_ep_options for CLI parameter parsing.
# Kept as a local tuple — not exported; does not duplicate the session taxonomy.
_EP_CLI_PREFIXES = ("qnn", "openvino", "ov", "vitisai", "vitis")


def normalize_ep_name(ep: str | None) -> str | None:
    """Normalize EP name from shorthand or alias to full canonical name.

    Delegates to ``expand_ep_name`` from the session facade, which covers
    all registered short names.  The legacy aliases ``ov`` and ``vitis``
    are mapped here before forwarding so existing callers keep working.

    Args:
        ep: Execution provider name (can be full name, short name, or alias)

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

    # Map legacy two-letter aliases not in the session facade.
    _legacy = {"ov": "openvino", "vitis": "vitisai"}
    ep_lower = ep.lower()
    if ep_lower in _legacy:
        ep = _legacy[ep_lower]

    return expand_ep_name(ep)


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
    ep_options = {}
    for param_name, param_value in kwargs.items():
        parts = param_name.split("_", 1)
        if param_value is not None and len(parts) == 2 and parts[0] in _EP_CLI_PREFIXES:
            ep_options[parts[1]] = str(param_value)
    return ep_options


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
