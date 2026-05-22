# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared constants for WinML CLI."""

from __future__ import annotations

import logging
import os
import re
import sys
from contextlib import contextmanager
from typing import Literal, TypeAlias, get_args


logger = logging.getLogger(__name__)

# Matches ANSI SGR escape sequences (e.g. the colour codes ORT emits).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# ORT messages captured before any logging handler is configured are buffered
# here so callers can replay them once the logging infrastructure is ready.
_ort_startup_logs: list[str] = []


@contextmanager
def _suppress_ep_registration_stderr():
    """Capture native stderr during ORT initialization and re-emit as debug logs.

    ORT native code writes "Init provider bridge failed." directly to native
    stderr (fd 2 / Win32 STD_ERROR_HANDLE), bypassing Python's logging system.
    Captures the output via a pipe and re-emits each line as
    ``logger.debug("[ORT] <line>")``.

    Restore order matters on Windows: Win32 STD_ERROR_HANDLE is restored
    *before* os.dup2 so that the pipe write HANDLE is not already closed by
    CRT when SetStdHandle releases its reference.
    """
    read_fd, write_fd = os.pipe()
    old_fd = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    old_w32 = None
    if sys.platform == "win32":
        import ctypes
        import msvcrt

        k32 = ctypes.WinDLL("kernel32")
        _std_error_handle = ctypes.c_uint32(0xFFFFFFF4)
        old_w32 = k32.GetStdHandle(_std_error_handle)
        k32.SetStdHandle(_std_error_handle, msvcrt.get_osfhandle(2))
    try:
        yield
    finally:
        # 1. Restore Win32 handle first so SetStdHandle releases the pipe
        #    write HANDLE before CRT closes it via dup2.
        if old_w32 is not None:
            k32.SetStdHandle(_std_error_handle, old_w32)
        # 2. Restore CRT fd 2; this closes the last reference to the pipe
        #    write end, so the subsequent read will reach EOF without blocking.
        os.dup2(old_fd, 2)
        os.close(old_fd)
        # 3. Read all captured output and log each line at DEBUG level.
        chunks: list[bytes] = []
        try:
            while chunk := os.read(read_fd, 4096):
                chunks.append(chunk)
        finally:
            os.close(read_fd)
        captured = b"".join(chunks).decode("utf-8", errors="replace")
        for line in captured.splitlines():
            line = _ANSI_RE.sub("", line).strip()
            if line:
                _ort_startup_logs.append(line)
                logger.debug("[ORT] %s", line)


with _suppress_ep_registration_stderr():
    import onnxruntime as ort


# Canonical ORT execution provider full names (the `*ExecutionProvider` symbols).
# Source of truth: docs/naming-convention.md.
EPName = Literal[
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "MIGraphXExecutionProvider",
    "NvTensorRTRTXExecutionProvider",
    "OpenVINOExecutionProvider",
    "QNNExecutionProvider",
    "VitisAIExecutionProvider",
]

# Shorthand aliases users can pass on the CLI (case-insensitive at the parser layer).
EPAlias = Literal[
    "qnn",
    "openvino",
    "vitisai",
    "cpu",
    "cuda",
    "dml",
    "nvtensorrtrtx",
    "nv_tensorrt_rtx",
    "migraphx",
]

# Either an alias or a full name — what user-facing entry points accept before normalization.
EPNameOrAlias: TypeAlias = EPName | EPAlias


# Supported execution providers — derived from the ``EPName`` Literal above so
# that ``utils.constants`` stays leaf-level (no import dependency on sysinfo).
# Membership parity with ``sysinfo.device._EP_DEVICE_MAP`` is enforced by
# ``tests/unit/utils/test_ep_constants.py::test_matches_ep_device_map_keys``.
SUPPORTED_EPS: list[EPName] = list(get_args(EPName))

# EP shorthand aliases (case-insensitive)
EP_ALIASES: dict[EPAlias, EPName] = {
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "nvtensorrtrtx": "NvTensorRTRTXExecutionProvider",
    "nv_tensorrt_rtx": "NvTensorRTRTXExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
}

# Reverse mapping: canonical EP name -> primary shorthand alias.
# Every canonical name has exactly one primary alias (the "preferred" one when
# multiple aliases share a canonical, e.g. ``openvino``/``ov`` -> ``openvino``).
# Use this to convert a canonical name back to the alias domain without `cast`.
EP_NAME_TO_ALIAS: dict[EPName, EPAlias] = {
    "QNNExecutionProvider": "qnn",
    "OpenVINOExecutionProvider": "openvino",
    "VitisAIExecutionProvider": "vitisai",
    "CPUExecutionProvider": "cpu",
    "CUDAExecutionProvider": "cuda",
    "DmlExecutionProvider": "dml",
    "NvTensorRTRTXExecutionProvider": "nv_tensorrt_rtx",
    "MIGraphXExecutionProvider": "migraphx",
}

# Runtime-iterable forms of the Literal types above (for membership checks, choice lists).
EP_NAMES: tuple[EPName, ...] = get_args(EPName)
EP_ALIAS_NAMES: tuple[EPAlias, ...] = get_args(EPAlias)

# All accepted EP names (full names + aliases)
ALL_EP_NAMES = list(SUPPORTED_EPS) + list(EP_ALIASES.keys())


def normalize_ep_name(ep: EPNameOrAlias | None) -> EPName | None:
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
        >>> normalize_ep_name("QNNExecutionProvider")
        'QNNExecutionProvider'
    """
    if ep is None:
        return None

    # Check if it's already a full name.
    # ``EP_NAMES`` is the runtime tuple of canonical names from the EPName Literal,
    # so membership narrowing here gives the type checker an EPName directly.
    if ep in EP_NAMES:
        return ep

    # Try to find in aliases (case-insensitive). ``.get()`` returns Optional, but
    # the prior membership check narrowed ``ep_lower`` so the alias mapping is
    # total in this branch.
    ep_lower = ep.lower()
    canonical = EP_ALIASES.get(ep_lower)  # type: ignore[arg-type]
    if canonical is not None:
        return canonical

    # Return as-is if not found (let validation catch invalid names).
    # The value isn't in ``EPName`` at runtime; the annotation is a best-effort
    # promise for downstream consumers, who handle the unknown case explicitly.
    return ep  # type: ignore[return-value]


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

# EP -> ordered tuple of supported devices (lowercase). The FIRST element is
# the canonical default device when only ``--ep`` is provided. Single source
# of truth for both compatibility checks and default-device inference.
# ``sysinfo.device._EP_DEVICE_MAP`` is derived from this table.
#
# Iteration order also feeds ``sysinfo.device._DEVICE_EP_MAP`` (and therefore
# ``resolve_eps``): the per-device priority is **IHV-first, native-last**
# (Nvidia -> AMD -> Qualcomm -> Intel -> Microsoft -> CPU), so the keys are
# listed in that order rather than alphabetically.
EP_SUPPORTED_DEVICES: dict[EPName, tuple[str, ...]] = {
    "NvTensorRTRTXExecutionProvider": ("gpu",),
    "CUDAExecutionProvider": ("gpu",),
    "MIGraphXExecutionProvider": ("gpu",),
    "VitisAIExecutionProvider": ("npu",),
    "QNNExecutionProvider": ("npu", "gpu"),
    "OpenVINOExecutionProvider": ("npu", "gpu", "cpu"),
    "DmlExecutionProvider": ("gpu",),
    "CPUExecutionProvider": ("cpu",),
}

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
