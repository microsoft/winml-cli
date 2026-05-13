# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# src/winml/modelkit/session/ep_device.py
"""EPDevice descriptor + resolution helpers + exception taxonomy.

EPDevice is a pure-data identifier for one (EP, hardware-device) target.
It is frozen, JSON-serializable, and has no runtime dependency on ORT.
Construction is performed by resolve_device(...) or rehydrated via
from_dict(...). The OrtEpDevice handle is re-derived inside session.py
at session-build time and never stored on EPDevice itself.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import asdict, dataclass
from typing import Any, Final


logger = logging.getLogger(__name__)


# --- exceptions ------------------------------------------------------------


class EPNotDiscovered(Exception):  # noqa: N818
    """EP plugin is not in the catalog or MODELKIT_EP_PATH."""


class EPRegistrationFailed(Exception):  # noqa: N818
    """ort.register_execution_provider_library raised."""


class DeviceNotFound(Exception):  # noqa: N818
    """EP registered, but no OrtEpDevice matches the descriptor."""


class AmbiguousMatch(Exception):  # noqa: N818
    """Multiple OrtEpDevices match the descriptor after dedup (bug signal)."""


class EPMonitorMismatch(Exception):  # noqa: N818
    """Monitor.ep_name does not agree with EPDevice.ep."""


# --- dataclass -------------------------------------------------------------


@dataclass(frozen=True)
class EPDevice:
    """Pure-data identifier of one (EP, hardware-device) binding target."""

    ep: str
    device: str
    vendor_id: int
    device_id: int
    vendor: str = ""

    def __post_init__(self) -> None:
        # Frozen dataclass — must use object.__setattr__ to mutate.
        if self.device != self.device.lower():
            object.__setattr__(self, "device", self.device.lower())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON round-trip."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EPDevice:
        """Rehydrate from a dict produced by to_dict."""
        return cls(
            ep=d["ep"],
            device=d["device"],
            vendor_id=d["vendor_id"],
            device_id=d["device_id"],
            vendor=d.get("vendor", ""),
        )


# --- canonicalization ------------------------------------------------------

# MIGRATION: After feat/update-pkg-deps merges, replace this stub with
#     from .ep_path import canonicalize_ep_name
# and delete _EP_NAME_ALIASES below. This stub is only the casing-fix
# slice required to keep this PR self-contained.
_EP_NAME_ALIASES: Final[dict[str, str]] = {
    "nvtensorrtrtxexecutionprovider": "NvTensorRtRtxExecutionProvider",
}


def canonicalize_ep_name(name: str) -> str:
    """Normalize a canonical EP name's casing via the alias table."""
    return _EP_NAME_ALIASES.get(name.lower(), name)


_SHORT_TO_FULL: Final[dict[str, str]] = {
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "nv_tensorrt_rtx": "NvTensorRtRtxExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def expand_ep_name(name: str) -> str:
    """Expand a short EP name to its full form; passthrough if already full.

    "xxx" is the short form of "xxxExecutionProvider" (case-folded for
    lookup). Already-full names flow through canonicalize_ep_name()
    for casing fixes (e.g. NvTensorRTRTX -> NvTensorRtRtx).
    """
    full = _SHORT_TO_FULL.get(name.lower())
    if full is not None:
        return full
    return canonicalize_ep_name(name)


# Inverse of _SHORT_TO_FULL — built lazily so any future additions to
# _SHORT_TO_FULL are picked up automatically.
_FULL_TO_SHORT: Final[dict[str, str]] = {v: k for k, v in _SHORT_TO_FULL.items()}


def short_ep_name(full: str) -> str:
    """Inverse of expand_ep_name: full EP name -> short form.

    Returns the short alias if known (e.g. ``"QNNExecutionProvider"`` -> ``"qnn"``).
    Falls back to ``full.removesuffix("ExecutionProvider").lower()`` for
    unknown full names so the function never raises — the caller can
    then validate against their own short-name allowlist.
    """
    if full in _FULL_TO_SHORT:
        return _FULL_TO_SHORT[full]
    return full.removesuffix("ExecutionProvider").lower()


# --- EP / device taxonomy --------------------------------------------------
# Single authoritative source.  config/precision.py imports from here
# (via the session facade) so there is no duplication.

# EP short name  ->  device category
_EP_TO_DEVICE: Final[dict[str, str]] = {
    "qnn": "npu",
    "vitisai": "npu",
    "dml": "gpu",
    "migraphx": "gpu",
    "tensorrt": "gpu",
    "cuda": "gpu",
    "openvino": "gpu",
    "cpu": "cpu",
}

# Device category -> default compile provider (None = built-in CPU EP)
_DEVICE_TO_PROVIDER: Final[dict[str, str | None]] = {
    "npu": "qnn",
    "gpu": "dml",
    "cpu": None,
}

# Public frozensets for callers that only need membership tests
VALID_EPS: Final[frozenset[str]] = frozenset(_EP_TO_DEVICE.keys())
_VALID_DEVICES: Final[frozenset[str]] = frozenset({"npu", "gpu", "cpu"})


def get_provider_for_device(device: str) -> str | None:
    """Get the default compile provider for a resolved device.

    Args:
        device: Resolved device name (``"npu"``, ``"gpu"``, ``"cpu"``).

    Returns:
        Provider name (e.g. ``"qnn"``, ``"dml"``) or ``None`` for CPU.
    """
    return _DEVICE_TO_PROVIDER.get(device)


def ep_to_device(ep: str) -> str:
    """Map an EP short name to its device category.

    Args:
        ep: EP short name (e.g. ``"qnn"``, ``"dml"``).

    Returns:
        Device category string: ``"npu"``, ``"gpu"``, or ``"cpu"``.

    Raises:
        ValueError: If *ep* is not a recognised EP short name.
    """
    ep_lower = ep.lower()
    if ep_lower not in _EP_TO_DEVICE:
        raise ValueError(f"Unknown EP '{ep}'. Known EPs: {sorted(_EP_TO_DEVICE.keys())}")
    return _EP_TO_DEVICE[ep_lower]


# --- resolution ------------------------------------------------------------

# Module-level sentinel — populated lazily on first call to resolve_device so
# that the circular-import at startup (ep_registry imports ep_device) is
# avoided. Tests patch this binding directly via:
#   patch("winml.modelkit.session.ep_device.WinMLEPRegistry")
# which replaces the name in this module's namespace before resolve_device runs,
# so the lazy-load branch is never taken during tests.
WinMLEPRegistry: Any = None


def _get_ep_registry() -> Any:
    """Return WinMLEPRegistry, importing ep_registry lazily on first real call."""
    global WinMLEPRegistry
    if WinMLEPRegistry is None:
        mod = importlib.import_module(".ep_registry", package=__name__.rsplit(".", 1)[0])
        WinMLEPRegistry = mod.WinMLEPRegistry
    return WinMLEPRegistry


def resolve_device(
    ep: str | None = None,
    device: str | None = None,
) -> EPDevice:
    """Resolve a (EP name, device kind) pair to an EPDevice.

    Deduction matrix:
        both given   -> validate + return
        ep only      -> _EP_TO_DEVICE[ep] gives device
        device only  -> _DEVICE_TO_PROVIDER[device] gives ep
        neither      -> sysinfo auto-detect: pick strongest device,
                        then fall through to the device-only path

    Args:
        ep: User-supplied EP name (short form e.g. ``"qnn"`` or full).
            ``None`` deduces from *device*.
        device: ``"cpu"`` | ``"gpu"`` | ``"npu"`` (case-insensitive).
            ``None`` deduces from *ep* or sysinfo.

    Raises:
        ValueError:           Unknown EP or device string.
        EPNotDiscovered:      EP plugin not in catalog or MODELKIT_EP_PATH.
        EPRegistrationFailed: ort.register_execution_provider_library raised.
        DeviceNotFound:       EP registered, but no matching OrtEpDevice.
        AmbiguousMatch:       multiple OrtEpDevice match after dedup.
    """
    # --- deduction phase ---------------------------------------------------
    if ep is None and device is None:
        # Auto-detect: pick strongest available device via sysinfo
        from ..sysinfo import resolve_device_category

        resolved_device_str, _ = resolve_device_category(device="auto")
        device = resolved_device_str

    if ep is not None and device is None:
        # ep given, device missing — infer from taxonomy
        ep_lower = ep.lower()
        # Normalise to short form for lookup (e.g. "QNNExecutionProvider" -> "qnn")
        ep_short = (
            ep_lower if ep_lower in _EP_TO_DEVICE else short_ep_name(expand_ep_name(ep_lower))
        )
        if ep_short not in _EP_TO_DEVICE:
            raise ValueError(
                f"Cannot deduce device for EP '{ep}'. Known EPs: {sorted(_EP_TO_DEVICE.keys())}"
            )
        device = _EP_TO_DEVICE[ep_short]
        logger.debug("Deduced device=%r from ep=%r", device, ep)

    if device is not None and ep is None:
        # device given, ep missing — infer default EP
        device_lower = device.lower()
        if device_lower not in _DEVICE_TO_PROVIDER:
            raise ValueError(
                f"Unknown device '{device}'. Expected one of: {sorted(_VALID_DEVICES)}"
            )
        default_ep = _DEVICE_TO_PROVIDER[device_lower]
        # cpu maps to None (built-in); fall back to "cpu" short name
        ep = default_ep if default_ep is not None else "cpu"
        device = device_lower
        logger.debug("Deduced ep=%r from device=%r", ep, device)

    # At this point both ep and device are non-None strings (type-checker aid)
    assert ep is not None
    assert device is not None

    # --- resolution phase --------------------------------------------------
    ep_full = expand_ep_name(ep)
    device_lower = device.lower()
    registry_cls = _get_ep_registry()
    devices = registry_cls.get_instance().register_ep(ep_full)

    matching = [d for d in devices if d.device.type.name.lower() == device_lower]

    # Dedup by (vendor_id, device_id) — handles QNN's duplicate-GPU rows.
    seen: set[tuple[int, int]] = set()
    deduped: list[Any] = []
    for d in matching:
        key = (d.device.vendor_id, d.device.device_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)

    if not deduped:
        available = [
            (d.device.type.name, hex(d.device.vendor_id), hex(d.device.device_id)) for d in devices
        ]
        raise DeviceNotFound(
            f"No OrtEpDevice for {ep_full} matches device={device_lower!r}. Available: {available}"
        )
    if len(deduped) > 1:
        conflicting = [
            (d.device.type.name, hex(d.device.vendor_id), hex(d.device.device_id)) for d in deduped
        ]
        raise AmbiguousMatch(
            f"Multiple OrtEpDevice match {ep_full}+{device_lower} after "
            f"dedup: {conflicting}. This is a registry bug; not a user error."
        )

    chosen = deduped[0]
    return EPDevice(
        ep=ep_full,
        device=device_lower,
        vendor_id=chosen.device.vendor_id,
        device_id=chosen.device.device_id,
        vendor=getattr(chosen.device, "vendor", "") or "",
    )
