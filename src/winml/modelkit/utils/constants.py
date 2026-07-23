# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared constants for WinML CLI."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast, get_args, overload


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
    "TensorrtExecutionProvider",
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
    "tensorrt",
]

# Either an alias or a full name — what user-facing entry points accept before normalization.
EPNameOrAlias: TypeAlias = EPName | EPAlias


# Compile backends selectable via ``--compiler`` (see commands/compile.py):
#   "ort"          -> ort.ModelCompiler (default)
#   "ort_session"  -> ort.InferenceSession (ep.context_enable)
#   "qairt"        -> QAIRT SDK compiler
CompilerName = Literal["ort", "ort_session", "qairt"]

# The ``--compiler`` choice that selects the ort.InferenceSession backend (the others
# go through ort.ModelCompiler / the QAIRT SDK). Referenced wherever the backend is
# branched on, so the magic string lives in exactly one place.
ORT_SESSION_COMPILER: CompilerName = "ort_session"

# Runtime-iterable form of ``CompilerName`` (e.g. for the CLI choice list).
COMPILER_NAMES: tuple[CompilerName, ...] = get_args(CompilerName)


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
    "tensorrt": "TensorrtExecutionProvider",
}

# Runtime-iterable forms of the Literal types above (for membership checks, choice lists).
EP_NAMES: tuple[EPName, ...] = get_args(EPName)
EP_ALIAS_NAMES: tuple[EPAlias, ...] = get_args(EPAlias)

# All accepted EP names (full names + aliases)
ALL_EP_NAMES = list(SUPPORTED_EPS) + list(EP_ALIASES.keys())


@overload
def normalize_ep_name(ep: None) -> None: ...


@overload
def normalize_ep_name(ep: EPNameOrAlias) -> EPName: ...


@overload
def normalize_ep_name(ep: str) -> str: ...


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
        >>> normalize_ep_name("QNNExecutionProvider")
        'QNNExecutionProvider'
    """
    if ep is None:
        return None

    ep_folded = ep.casefold()
    for canonical_name in EP_NAMES:
        if canonical_name.casefold() == ep_folded:
            return canonical_name

    # Try to find in aliases (case-insensitive). ``.get()`` returns Optional, but
    # the prior membership check narrowed ``ep_lower`` so the alias mapping is
    # total in this branch.
    # ep_folded is an arbitrary folded string; cast to the key type for the
    # lookup (.get tolerates non-alias keys, returning None).
    canonical = EP_ALIASES.get(cast("EPAlias", ep_folded))
    if canonical is not None:
        return canonical

    # Return as-is if not found so downstream consumers can report their
    # context-specific validation error.
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


# Device priority is shared by auto-selection and hardware monitoring.
DeviceType = Literal["npu", "gpu", "cpu"]
DEVICE_PRIORITY = cast("tuple[DeviceType, ...]", get_args(DeviceType))
ACCELERATOR_DEVICE_TYPES = DEVICE_PRIORITY[:-1]

# Legacy uppercase form used by model metadata and CLI output.
SUPPORTED_DEVICES = [device.upper() for device in reversed(DEVICE_PRIORITY)]

# EP -> ordered tuple of supported devices (lowercase). The FIRST element is
# the canonical default device when only ``--ep`` is provided. Single source
# of truth for both compatibility checks and default-device inference.
# ``sysinfo.device._EP_DEVICE_MAP`` is derived from this table.
#
# Iteration order also feeds ``sysinfo.device._DEVICE_EP_MAP`` (and therefore
# ``resolve_eps``): the per-device priority is **IHV-first, native-last**
# (Nvidia -> AMD -> Qualcomm -> Intel -> Microsoft -> CPU -> Vitis), so the
# keys are listed in that order rather than alphabetically.
# VitisAI is placed last because it is not yet fully supported.
EP_SUPPORTED_DEVICES: dict[EPName, tuple[DeviceType, ...]] = {
    "NvTensorRTRTXExecutionProvider": ("gpu",),
    "CUDAExecutionProvider": ("gpu",),
    "MIGraphXExecutionProvider": ("gpu",),
    "QNNExecutionProvider": ("npu", "gpu"),
    "OpenVINOExecutionProvider": ("npu", "gpu", "cpu"),
    "TensorrtExecutionProvider": ("gpu",),
    "DmlExecutionProvider": ("gpu",),
    "CPUExecutionProvider": ("cpu",),
    "VitisAIExecutionProvider": ("npu",),
}
