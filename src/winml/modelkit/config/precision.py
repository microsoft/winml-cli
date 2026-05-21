# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Precision resolution for WinML CLI.

Pure decision logic: given a device, precision, and available devices,
produce a PrecisionPolicy. No I/O, no config mutation, no sysinfo dependency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..sysinfo import resolve_eps
from ..utils.constants import EP_SUPPORTED_DEVICES, EPName, EPNameOrAlias, normalize_ep_name


logger = logging.getLogger(__name__)

# Tasks where GPU auto-precision may differ (LLM = w4a16 recommendation)
_LLM_TASKS = frozenset(
    {
        "text-generation",
        "text2text-generation",
    }
)

# Default auto-precision mapping: device -> precision
_AUTO_PRECISION: dict[str, str] = {
    "npu": "w8a16",
    "gpu": "fp16",
    "cpu": "fp16",
}

# Precision -> weight/activation type mapping (named presets)
_WEIGHT_TYPE: dict[str, str | None] = {
    "int8": "uint8",
    "int16": "int16",
    "fp16": None,
    "fp32": None,
}

_ACTIVATION_TYPE: dict[str, str | None] = {
    "int8": "uint8",
    "int16": "uint16",
    "fp16": None,
    "fp32": None,
}

# Bit-width -> default quantization type.
# Uses unsigned types by default (works for QNN EP).
# TODO: If a future EP (e.g., OpenVINO) requires signed types (int8/int16),
# add an EP-specific override layer keyed by compile_provider.
_BITS_TO_WEIGHT_TYPE: dict[int, str] = {
    8: "uint8",
    16: "int16",
}

_BITS_TO_ACTIVATION_TYPE: dict[int, str] = {
    8: "uint8",
    16: "uint16",
}


_VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})

# Named precision presets (non-mixed)
_NAMED_PRECISIONS = frozenset({"auto", "fp32", "fp16", "int8", "int16"})

# Regex for mixed precision: w{weight_bits}a{activation_bits}
_MIXED_RE = re.compile(r"^w(\d+)a(\d+)$")


def resolve_quant_types(precision: str) -> tuple[str, str]:
    """Resolve a precision string to (weight_type, activation_type).

    Handles both named presets ("int8", "int16") and mixed format ("w8a16").
    Float precisions ("fp16", "fp32") raise ValueError — they have no quant types.

    Args:
        precision: Precision string (e.g., "int8", "w8a16").

    Returns:
        (weight_type, activation_type) tuple (e.g., ("uint8", "uint16")).

    Raises:
        ValueError: If precision is float, "auto", or uses unsupported bit widths.
    """
    p = precision.lower()

    # Named preset
    if p in _WEIGHT_TYPE:
        w, a = _WEIGHT_TYPE[p], _ACTIVATION_TYPE[p]
        if w is None:
            raise ValueError(f"Precision '{precision}' is a float type — no quantization types.")
        return w, a

    # Mixed w{x}a{y} format
    m = _MIXED_RE.match(p)
    if m:
        w_bits, a_bits = int(m.group(1)), int(m.group(2))
        if w_bits not in _BITS_TO_WEIGHT_TYPE:
            raise ValueError(
                f"Unsupported weight bit-width {w_bits} in '{precision}'. "
                f"Supported: {sorted(_BITS_TO_WEIGHT_TYPE.keys())}"
            )
        if a_bits not in _BITS_TO_ACTIVATION_TYPE:
            raise ValueError(
                f"Unsupported activation bit-width {a_bits} in '{precision}'. "
                f"Supported: {sorted(_BITS_TO_ACTIVATION_TYPE.keys())}"
            )
        return _BITS_TO_WEIGHT_TYPE[w_bits], _BITS_TO_ACTIVATION_TYPE[a_bits]

    raise ValueError(
        f"Unknown precision '{precision}'. "
        f"Expected one of {sorted(_NAMED_PRECISIONS)} or w{{x}}a{{y}} format (e.g., w8a16)."
    )


def is_quantized_precision(precision: str) -> bool:
    """Return True if precision implies quantization (not float).

    Only returns True for *supported* precisions — unknown w{x}a{y} bit
    widths (e.g., w4a16) return False rather than claiming to be quantized.
    """
    p = precision.lower()
    if p in ("fp16", "fp32", "auto"):
        return False
    if p in _WEIGHT_TYPE:
        return _WEIGHT_TYPE[p] is not None
    m = _MIXED_RE.match(p)
    if not m:
        return False
    w_bits, a_bits = int(m.group(1)), int(m.group(2))
    return w_bits in _BITS_TO_WEIGHT_TYPE and a_bits in _BITS_TO_ACTIVATION_TYPE


def _is_valid_precision(precision: str) -> bool:
    """Check if a precision string is valid (named preset or w{x}a{y})."""
    if precision in _NAMED_PRECISIONS:
        return True
    m = _MIXED_RE.match(precision)
    if not m:
        return False
    w_bits, a_bits = int(m.group(1)), int(m.group(2))
    return w_bits in _BITS_TO_WEIGHT_TYPE and a_bits in _BITS_TO_ACTIVATION_TYPE


@dataclass
class PrecisionPolicy:
    """Resolved precision policy for a build.

    Attributes:
        device: Concrete device: "npu", "gpu", or "cpu".
        precision: Resolved precision string (e.g., "int8", "w8a16", "fp16").
        weight_type: Quantization weight type, or None for fp32/fp16.
        activation_type: Quantization activation type, or None for fp32/fp16.
        compile_provider: Canonical EP name (e.g., "QNNExecutionProvider"), or None.
    """

    device: str
    precision: str
    weight_type: str | None
    activation_type: str | None
    compile_provider: EPName | None


def resolve_precision(
    *,
    device: str = "auto",
    precision: str = "auto",
    ep: EPNameOrAlias | None = None,
    available_devices: list[str] | None = None,
    task: str | None = None,
) -> PrecisionPolicy:
    """Resolve precision into a concrete PrecisionPolicy.

    Pure function, no I/O.

    When device is "auto" and precision is "auto", returns a no-op policy
    (device="auto") signaling the caller should keep config defaults.

    When device is "auto" but precision is explicit, walks available_devices
    to find a suitable device for the requested precision.

    Args:
        device: Target device ("npu", "gpu", "cpu", or "auto").
        precision: Target precision ("fp32", "fp16", "int8", "int16", "w8a16", or "auto").
            "w8a16" = mixed precision: uint8 weights + uint16 activations.
        ep: Explicit EP override (e.g., "migraphx", "nv_tensorrt_rtx"). When set,
            overrides the default device→provider mapping. If device is
            "auto", the device is inferred from the EP.
        available_devices: Prioritized device list from resolve_device().
            Used when device="auto" + precision is explicit.
        task: Optional task name for LLM-specific warnings.

    Returns:
        PrecisionPolicy with all fields resolved.

    Raises:
        ValueError: If device or precision is not recognized.
    """
    # Normalize inputs
    device = device.lower()
    resolved_precision = precision.lower() if precision != "auto" else "auto"

    # Validate: must be a named preset, w{x}a{y} format, or "auto"
    if resolved_precision != "auto" and not _is_valid_precision(resolved_precision):
        raise ValueError(
            f"Unknown precision '{precision}'. "
            f"Expected one of {sorted(_NAMED_PRECISIONS)} or w{{x}}a{{y}} format (e.g., w8a16)."
        )

    # Validate EP override (normalize aliases → canonical name before lookup).
    ep_canonical: EPName | None = None
    if ep is not None:
        ep_canonical = normalize_ep_name(ep)
        if ep_canonical not in EP_SUPPORTED_DEVICES:
            raise ValueError(f"Unknown EP '{ep}'. Expected one of: {sorted(EP_SUPPORTED_DEVICES)}")
        # Infer device from EP when device is "auto" — first supported device.
        if device == "auto":
            device = EP_SUPPORTED_DEVICES[ep_canonical][0]
            logger.info("Inferred device '%s' from EP '%s'", device, ep_canonical)

    # --- Both auto: no-op, keep config defaults ---
    if device == "auto" and resolved_precision == "auto":
        return PrecisionPolicy(
            device="auto",
            precision="auto",
            weight_type=None,
            activation_type=None,
            compile_provider=None,
        )

    # --- Device is explicit ---
    if device != "auto":
        if device not in _VALID_DEVICES:
            raise ValueError(
                f"Unknown device '{device}'. Expected one of: {sorted(_VALID_DEVICES)}"
            )
        resolved_device = device
    else:
        # Device is "auto" but precision is explicit — pick best device
        # FIXME: improve device-precision compatibility lookup table later
        resolved_device = _pick_device_for_precision(
            resolved_precision,
            available_devices or ["cpu"],
        )

    # Resolve "auto" precision for the resolved device
    if resolved_precision == "auto":
        resolved_precision = _AUTO_PRECISION[resolved_device]

        # GPU + LLM: warn about w4a16 recommendation
        if resolved_device == "gpu" and task in _LLM_TASKS:
            logger.warning(
                "GPU + LLM task '%s': auto-precision is fp16 (no quantization). "
                "For better performance, consider w4a16 quantization manually.",
                task,
            )

    # ep=CPUExecutionProvider means no EPContext compilation needed.
    # For all other explicit EPs (canonical names), use ep as the provider.
    compile_provider: EPName | None = ep_canonical
    if not compile_provider:
        eps = resolve_eps(resolved_device)
        compile_provider = eps[0] if eps else None
    if compile_provider == "CPUExecutionProvider":
        compile_provider = None

    # Resolve weight/activation types — supports named presets and w{x}a{y}
    if is_quantized_precision(resolved_precision):
        weight_type, activation_type = resolve_quant_types(resolved_precision)
    else:
        weight_type, activation_type = None, None

    return PrecisionPolicy(
        device=resolved_device,
        precision=resolved_precision,
        weight_type=weight_type,
        activation_type=activation_type,
        compile_provider=compile_provider,
    )


def _pick_device_for_precision(
    precision: str,
    available_devices: list[str],
) -> str:
    """Pick the best available device for an explicit precision.

    FIXME: This is a simple first-match heuristic. Will be improved with
    a proper device-precision compatibility matrix later.

    Current logic:
        quantized (int8/int16/w{x}a{y}) → prefer NPU, fall back to first available
        float (fp16/fp32)               → prefer GPU, fall back to first available
    """
    if is_quantized_precision(precision):
        # Prefer NPU for quantized models
        for d in available_devices:
            if d == "npu":
                return d
    elif precision in ("fp16", "fp32"):
        # Prefer GPU for float models
        for d in available_devices:
            if d == "gpu":
                return d

    # Fallback: first available device
    return available_devices[0] if available_devices else "cpu"
