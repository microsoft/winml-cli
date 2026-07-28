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
from typing import TYPE_CHECKING, Literal, cast

from ..session import (
    VALID_DEVICES,
    default_ep_for_device,
    ep_short_or_none,
)

# New session selection stays on EP_DEVICE_SPECS; this legacy build-policy
# surface intentionally keeps using EP_SUPPORTED_DEVICES for compatibility.
from ..utils.constants import (
    EP_SUPPORTED_DEVICES,
    EPS_WITH_INTERNAL_QUANT,
    normalize_ep_name,
)


if TYPE_CHECKING:
    # Referenced only from the quoted ``cast()`` below, so importing it at
    # runtime would leave an unused import behind.
    from ..utils.constants import EPNameOrAlias


logger = logging.getLogger(__name__)

QuantType = Literal["uint8", "int8", "uint16", "int16"]

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
    "gpu": "fp32",
    "cpu": "fp32",
}

# Precision -> weight/activation type mapping (named presets)
_WEIGHT_TYPE: dict[str, QuantType | None] = {
    "int8": "uint8",
    "int16": "int16",
    "fp16": None,
    "fp32": None,
}

_ACTIVATION_TYPE: dict[str, QuantType | None] = {
    "int8": "uint8",
    "int16": "uint16",
    "fp16": None,
    "fp32": None,
}

# Bit-width -> default quantization type.
# Uses unsigned types by default (works for QNN EP).
# TODO: If a future EP (e.g., OpenVINO) requires signed types (int8/int16),
# add an EP-specific override layer keyed by compile_provider.
_BITS_TO_WEIGHT_TYPE: dict[int, QuantType] = {
    8: "uint8",
    16: "int16",
}

_BITS_TO_ACTIVATION_TYPE: dict[int, QuantType] = {
    8: "uint8",
    16: "uint16",
}

# Named precision presets (non-mixed)
_NAMED_PRECISIONS = frozenset({"auto", "fp32", "fp16", "int4", "int8", "int16"})

# Regex for mixed precision: w{weight_bits}a{activation_bits}
_MIXED_RE = re.compile(r"^w(\d+)a(\d+)$")

# Valid bit widths for w{x}a{y} validation.
# Weight supports 4-bit (RTN weight-only) plus 8/16-bit (QDQ).
# Activation supports 8/16-bit for QDQ, plus 32-bit (meaning "keep FP32, no
# activation quantization") which is only valid with weight-only (4-bit RTN).
_VALID_WEIGHT_BITS = frozenset({4, 8, 16})
_VALID_ACTIVATION_BITS = frozenset({8, 16, 32})


def resolve_quant_types(precision: str) -> tuple[QuantType, QuantType]:
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

    # Weight-only precisions use RTN, not QDQ — caller should use
    # is_weight_only_precision() to detect these before calling here.
    if is_weight_only_precision(p):
        raise ValueError(
            f"Precision '{precision}' is weight-only (RTN) — no QDQ quant types. "
            "Use is_weight_only_precision() to detect and create RTN config instead."
        )

    # Named preset
    if p in _WEIGHT_TYPE:
        w, a = _WEIGHT_TYPE[p], _ACTIVATION_TYPE[p]
        if w is None or a is None:
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

    Includes both QDQ precisions (int8, int16, w8a8) and weight-only
    precisions (int4, w4a16, w4a32) that use RTN.
    """
    p = precision.lower()
    if p in ("fp16", "fp32", "auto"):
        return False
    if p == "int4":
        return True
    if p in _WEIGHT_TYPE:
        return _WEIGHT_TYPE[p] is not None
    m = _MIXED_RE.match(p)
    if not m:
        return False
    w_bits, a_bits = int(m.group(1)), int(m.group(2))
    if w_bits not in _VALID_WEIGHT_BITS or a_bits not in _VALID_ACTIVATION_BITS:
        return False
    # a_bits=32 (keep FP32) only valid with weight-only (4-bit) RTN
    return not (a_bits == 32 and w_bits in _BITS_TO_WEIGHT_TYPE)


def _is_valid_precision(precision: str) -> bool:
    """Check if a precision string is valid (named preset or w{x}a{y})."""
    if precision in _NAMED_PRECISIONS:
        return True
    m = _MIXED_RE.match(precision)
    if not m:
        return False
    w_bits, a_bits = int(m.group(1)), int(m.group(2))
    if w_bits not in _VALID_WEIGHT_BITS or a_bits not in _VALID_ACTIVATION_BITS:
        return False
    # a_bits=32 (keep FP32) only valid with weight-only (4-bit) RTN
    return not (a_bits == 32 and w_bits in _BITS_TO_WEIGHT_TYPE)


def is_weight_only_precision(precision: str) -> bool:
    """Return True if precision implies weight-only quantization (RTN).

    Weight-only precisions use the RTN (Round-To-Nearest) algorithm with
    MatMulNBits ops instead of QDQ (QuantizeLinear/DequantizeLinear).

    Rules:
        - ``int4`` → weight-only 4-bit RTN (equivalent to ``w4a32``)
        - ``w4a32`` → weight 4-bit RTN, activation stays FP32
        - ``w4a16`` → weight 4-bit RTN + FP16 post-processing on activations
        - ``w4a8`` → weight 4-bit RTN + 8-bit activation (reserved)
        - All other precisions → False (use QDQ or FP16)

    Only returns True for valid precisions — ``w4a4`` returns False because
    4-bit activation is not supported.
    """
    p = precision.lower()
    if p == "int4":
        return True
    m = _MIXED_RE.match(p)
    if not m:
        return False
    w_bits, a_bits = int(m.group(1)), int(m.group(2))
    # Must be a valid precision AND have weight bits that are not QDQ-supported
    return (
        w_bits not in _BITS_TO_WEIGHT_TYPE
        and w_bits in _VALID_WEIGHT_BITS
        and a_bits in _VALID_ACTIVATION_BITS
    )


def extract_weight_bits(precision: str) -> int:
    """Extract weight bit-width from a precision string.

    Used to derive ``rtn_bits`` from the precision (e.g., ``int4`` → 4).
    Validates the precision format before extracting.

    Args:
        precision: A valid precision string (e.g., ``int4``, ``w4a16``, ``int8``).

    Returns:
        Weight bit-width as integer.

    Raises:
        ValueError: If precision is invalid or bit-width cannot be extracted.
    """
    p = precision.lower()
    preset_bits = {"int4": 4, "int8": 8, "int16": 16}
    if p in preset_bits:
        return preset_bits[p]
    m = _MIXED_RE.match(p)
    if m:
        w_bits, a_bits = int(m.group(1)), int(m.group(2))
        if w_bits not in _VALID_WEIGHT_BITS or a_bits not in _VALID_ACTIVATION_BITS:
            raise ValueError(
                f"'{precision}' has unsupported bit-widths (weight={w_bits}, activation={a_bits})"
            )
        # a_bits=32 only valid with weight-only (4-bit) — reject w8a32, w16a32
        if a_bits == 32 and w_bits in _BITS_TO_WEIGHT_TYPE:
            raise ValueError(
                f"'{precision}' is invalid: a32 (keep FP32) is only valid with "
                "weight-only precisions (4-bit RTN)"
            )
        return w_bits
    raise ValueError(f"Cannot extract weight bits from '{precision}'")


def extract_activation_bits(precision: str) -> int:
    """Extract activation bit-width from a precision string.

    For named presets: ``int4`` → 32 (activation stays FP32).
    For mixed format: ``w4a16`` → 16, ``w4a32`` → 32.

    Args:
        precision: A valid precision string.

    Returns:
        Activation bit-width as integer (8, 16, or 32).

    Raises:
        ValueError: If activation bits cannot be extracted.
    """
    p = precision.lower()
    # Named presets: int4 means activation stays FP32
    if p == "int4":
        return 32
    m = _MIXED_RE.match(p)
    if m:
        a_bits = int(m.group(2))
        if a_bits not in _VALID_ACTIVATION_BITS:
            raise ValueError(f"'{precision}' has unsupported activation bit-width: {a_bits}")
        return a_bits
    raise ValueError(f"Cannot extract activation bits from '{precision}'")


def expand_precision(precision: str) -> list[str]:
    """Expand a composite precision into an ordered list of single-operation passes.

    Only weight-only precisions with FP16 activation (w4a16) expand into
    multiple passes. QDQ precisions like w8a16 are a single QDQ operation
    (activation=uint16), NOT "int8 then FP16".

    Args:
        precision: A precision string (e.g., "w4a16", "int4", "fp16", "int8").

    Returns:
        List of single-pass precision strings in execution order.

    Examples:
        >>> expand_precision("w4a16")
        ['int4', 'fp16']
        >>> expand_precision("int4")
        ['int4']
        >>> expand_precision("fp16")
        ['fp16']
        >>> expand_precision("w8a16")
        ['w8a16']
    """
    p = precision.lower()
    if p == "w4a16":
        return ["int4", "fp16"]
    return [p]


@dataclass
class PrecisionPolicy:
    """Resolved precision policy for a build.

    Attributes:
        device: Concrete device: "npu", "gpu", or "cpu".
        precision: Resolved precision string (e.g., "int8", "w8a16", "fp16").
            Stays ``"auto"`` when ``skip_quantization`` is set — winml made no
            precision choice, the EP decides.
        weight_type: Quantization weight type, or None for fp32/fp16.
        activation_type: Quantization activation type, or None for fp32/fp16.
        compile_provider: Short EP name (e.g. "qnn", "dml") or None for CPU.
        skip_quantization: True when the build must run no quantization stage at
            all, i.e. apply exactly what ``--no-quant`` does (``quant = None``).
            Set for EPs that quantize internally (:data:`EPS_WITH_INTERNAL_QUANT`).
            Callers must honor this BEFORE inspecting ``precision``.
    """

    device: str
    precision: str
    weight_type: QuantType | None
    activation_type: QuantType | None
    compile_provider: str | None
    skip_quantization: bool = False


def resolve_precision(
    *,
    device: str = "auto",
    precision: str = "auto",
    ep: str | None = None,
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
        available_devices: Prioritized device list from sysinfo.get_available_devices().
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

    # Normalize and validate the EP override against the shared catalog.
    if ep is not None:
        ep_canonical = normalize_ep_name(cast("EPNameOrAlias", ep))
        if ep_canonical not in EP_SUPPORTED_DEVICES:
            raise ValueError(f"Unknown EP '{ep}'.")
        ep = ep_canonical
        supported_devices = EP_SUPPORTED_DEVICES[ep_canonical]
        # Infer device from EP when device is "auto"
        if device == "auto":
            device = _pick_available_device_for_ep(
                ep=ep_canonical,
                supported_devices=supported_devices,
                available_devices=available_devices,
            )
            logger.info("Inferred device '%s' from EP '%s'", device, ep)
        elif device not in supported_devices:
            raise ValueError(f"EP '{ep}' does not support device '{device}'.")

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
        if device not in VALID_DEVICES:
            raise ValueError(f"Unknown device '{device}'. Expected one of: {sorted(VALID_DEVICES)}")
        resolved_device = device
    else:
        # Device is "auto" but precision is explicit — pick best device
        # FIXME: improve device-precision compatibility lookup table later
        resolved_device = _pick_device_for_precision(
            resolved_precision,
            available_devices or ["cpu"],
        )

    # Resolve the EP this build will actually target, BEFORE any policy decision
    # that depends on it. An explicit --ep wins; otherwise deduce the registered
    # default for the device. Callers routinely pin only the device
    # (``--device npu`` with no ``--ep``), and on an AMD-only host that device
    # still resolves to VitisAI — so the auto-precision decision below must see
    # the deduced EP, not ``None``. ``compile_provider`` is derived from the same
    # value so the two can never disagree.
    effective_ep = ep if ep else default_ep_for_device(resolved_device)

    # Resolve "auto" precision for the resolved device
    skip_quantization = False
    if resolved_precision == "auto":
        if effective_ep in EPS_WITH_INTERNAL_QUANT:
            # This EP applies its own quantization scheme and cannot consume a
            # winml-produced QDQ graph, so make no precision choice at all and
            # tell the caller to skip the quantization stage (what --no-quant
            # does). Leaving precision as "auto" keeps the config honest: winml
            # picked nothing, the EP decides.
            skip_quantization = True
            logger.warning(
                "EP '%s' quantizes internally, so winml is skipping its own "
                "quantization stage (equivalent to --no-quant) instead of applying "
                "the '%s' default for device '%s'. This EP therefore behaves "
                "differently from the others: the artifact stays unquantized and "
                "the accelerator applies its own scheme at load time. "
                "Pass --precision explicitly to force a winml quantization pass.",
                effective_ep,
                _AUTO_PRECISION[resolved_device],
                resolved_device,
            )
        else:
            resolved_precision = _AUTO_PRECISION[resolved_device]

            # GPU + LLM: warn about w4a16 recommendation
            if resolved_device == "gpu" and task in _LLM_TASKS:
                logger.warning(
                    "GPU + LLM task '%s': auto-precision is fp32 (no conversion). "
                    "For better performance, consider w4a16 quantization manually.",
                    task,
                )

    # The policy contract uses short aliases, with CPU represented as no
    # offline compiler.
    compile_provider = ep_short_or_none(effective_ep) if effective_ep is not None else None

    # Resolve weight/activation types — supports named presets and w{x}a{y}.
    # Weight-only precisions (int4, w4a16) use RTN, not QDQ — they have no
    # traditional weight_type/activation_type.  The caller (resolve_quant_compile_config)
    # inspects PrecisionPolicy.precision to create RTN config.
    if is_weight_only_precision(resolved_precision):
        weight_type, activation_type = None, None
    elif is_quantized_precision(resolved_precision):
        weight_type, activation_type = resolve_quant_types(resolved_precision)
    else:
        weight_type, activation_type = None, None

    return PrecisionPolicy(
        device=resolved_device,
        precision=resolved_precision,
        weight_type=weight_type,
        activation_type=activation_type,
        compile_provider=compile_provider,
        skip_quantization=skip_quantization,
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


def _pick_available_device_for_ep(
    *,
    ep: str,
    supported_devices: tuple[str, ...],
    available_devices: list[str] | None,
) -> str:
    """Pick the first available device that the EP can actually target.

    ``resolve_precision`` is intentionally pure/offline: it must not query the
    runtime registry. When the caller provides ``available_devices``, that list
    is the only host signal available, so auto-device inference for an explicit
    EP must stay within its intersection with the EP's static device support.
    """
    if available_devices is None:
        return supported_devices[0]

    available_order = [candidate.lower() for candidate in available_devices]
    for candidate in available_order:
        if candidate in supported_devices:
            return candidate

    raise ValueError(
        f"EP '{ep}' does not support any available devices. "
        f"Supported devices: {', '.join(supported_devices)}. "
        f"Available devices: {', '.join(available_order) or '<none>'}."
    )
