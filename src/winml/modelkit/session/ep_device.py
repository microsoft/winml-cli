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
from dataclasses import asdict, dataclass
from typing import Any, Final


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


_SHORT_TO_CANONICAL: Final[dict[str, str]] = {
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "nv_tensorrt_rtx": "NvTensorRtRtxExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def expand_ep_name(name: str) -> str:
    """Expand a short EP name to canonical; passthrough if already canonical.

    "xxx" is the short form of "xxxExecutionProvider" (case-folded for
    lookup). Already-canonical names flow through canonicalize_ep_name()
    for casing fixes (e.g. NvTensorRTRTX -> NvTensorRtRtx).
    """
    canonical = _SHORT_TO_CANONICAL.get(name.lower())
    if canonical is not None:
        return canonical
    return canonicalize_ep_name(name)


# Inverse of _SHORT_TO_CANONICAL — built lazily so any future additions to
# _SHORT_TO_CANONICAL are picked up automatically.
_CANONICAL_TO_SHORT: Final[dict[str, str]] = {v: k for k, v in _SHORT_TO_CANONICAL.items()}


def short_ep_name(canonical: str) -> str:
    """Inverse of expand_ep_name: canonical EP name -> short form.

    Returns the short alias if known (e.g. ``"QNNExecutionProvider"`` -> ``"qnn"``).
    Falls back to ``canonical.removesuffix("ExecutionProvider").lower()`` for
    unknown canonical names so the function never raises — the caller can
    then validate against their own short-name allowlist.
    """
    if canonical in _CANONICAL_TO_SHORT:
        return _CANONICAL_TO_SHORT[canonical]
    return canonical.removesuffix("ExecutionProvider").lower()


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


def resolve_device(ep: str, device: str) -> EPDevice:
    """Resolve a (user-friendly EP name, device kind) pair to an EPDevice.

    Args:
        ep: User-supplied EP name. Short forms (e.g. "qnn") are expanded
            via expand_ep_name().
        device: "cpu" | "gpu" | "npu" (case-insensitive).

    Raises:
        EPNotDiscovered:      EP plugin not in catalog or MODELKIT_EP_PATH.
        EPRegistrationFailed: ort.register_execution_provider_library raised.
        DeviceNotFound:       EP registered, but no matching OrtEpDevice.
        AmbiguousMatch:       multiple OrtEpDevice match after dedup.
    """
    ep_canonical = expand_ep_name(ep)
    device_lower = device.lower()
    registry_cls = _get_ep_registry()
    devices = registry_cls.get_instance().register_ep(ep_canonical)

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
            f"No OrtEpDevice for {ep_canonical} matches device={device_lower!r}. "
            f"Available: {available}"
        )
    if len(deduped) > 1:
        conflicting = [
            (d.device.type.name, hex(d.device.vendor_id), hex(d.device.device_id)) for d in deduped
        ]
        raise AmbiguousMatch(
            f"Multiple OrtEpDevice match {ep_canonical}+{device_lower} after "
            f"dedup: {conflicting}. This is a registry bug; not a user error."
        )

    chosen = deduped[0]
    return EPDevice(
        ep=ep_canonical,
        device=device_lower,
        vendor_id=chosen.device.vendor_id,
        device_id=chosen.device.device_id,
        vendor=getattr(chosen.device, "vendor", "") or "",
    )
