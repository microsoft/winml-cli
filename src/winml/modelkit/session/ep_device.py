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
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
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
# Single authoritative source: the EPDeviceSpec catalog.
# config/precision.py imports helpers from here (via the session facade).


@dataclass(frozen=True, kw_only=True, slots=True)
class EPDeviceSpec:
    """One supported (EP, device) target in the catalog.

    Distinct from EPDevice:
      - EPDeviceSpec is the *kind-of-target* (machine-independent).
      - EPDevice is the *runtime instance* (machine-specific, carries
        vendor_id / device_id from the OrtEpDevice handle).
    Many EPDevices map to one EPDeviceSpec.
    """

    ep: str
    device: str
    default_provider_options: Mapping[str, str] = field(default_factory=dict)


EP_DEVICE_SPECS: Final[tuple[EPDeviceSpec, ...]] = (
    # Order encodes first-match deduction preference per device:
    #   npu-first:  QNNExecutionProvider   (Snapdragon HTP — highest-throughput)
    #   gpu-first:  DmlExecutionProvider   (Windows-native; compile-path default)
    #   cpu-first:  CPUExecutionProvider   (bundled with ORT — always available)
    # Secondary entries follow their primary within each device group.
    # ---- Primary per-device (positions 0-2) ----
    EPDeviceSpec(
        ep="QNNExecutionProvider",
        device="npu",
        default_provider_options={
            # Verified 2026-05-13: +3x throughput on ResNet-50 vs default mode
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        },
    ),  # primary NPU
    EPDeviceSpec(ep="DmlExecutionProvider", device="gpu"),  # primary GPU
    EPDeviceSpec(ep="CPUExecutionProvider", device="cpu"),  # primary CPU
    # ---- QNN secondary ----
    EPDeviceSpec(ep="QNNExecutionProvider", device="gpu"),  # TODO: measure
    EPDeviceSpec(ep="QNNExecutionProvider", device="cpu"),
    # ---- OpenVINO family ----
    # TODO: verify whether `device_type` is needed under add_provider_for_devices,
    # or auto-derived from OrtEpDevice handle (like QNN's backend_type).
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="npu"),
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="cpu"),
    # ---- Other single-device EPs ----
    EPDeviceSpec(ep="VitisAIExecutionProvider", device="npu"),
    EPDeviceSpec(ep="MIGraphXExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="TensorrtExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="CUDAExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="NvTensorRtRtxExecutionProvider", device="gpu"),
)

# O(1) lookup cache built from the ordered catalog.
_BY_KEY: Final[dict[tuple[str, str], EPDeviceSpec]] = {(s.ep, s.device): s for s in EP_DEVICE_SPECS}

# Public frozensets for callers that only need membership tests.
# VALID_EPS uses short names for backward compatibility with callers that pass
# short forms (e.g. "qnn", "dml").  The catalog uses full names internally.
VALID_EPS: Final[frozenset[str]] = frozenset({short_ep_name(s.ep) for s in EP_DEVICE_SPECS})
VALID_DEVICES: Final[frozenset[str]] = frozenset({s.device for s in EP_DEVICE_SPECS})


def lookup_device_spec(ep: str, device: str) -> EPDeviceSpec | None:
    """O(1) lookup by exact (ep, device) match using full EP names.

    Args:
        ep: Full EP name (e.g. ``"QNNExecutionProvider"``).
        device: Device category (e.g. ``"npu"``).

    Returns:
        The matching :class:`EPDeviceSpec`, or ``None`` if not found.
    """
    return _BY_KEY.get((ep, device))


def default_device_for_ep(ep: str) -> str | None:
    """First catalog variant whose ep matches. Replaces ``_EP_TO_DEVICE``.

    Order in ``EP_DEVICE_SPECS`` encodes preference:
      QNN's first variant is npu  → ``default_device_for_ep("QNNExecutionProvider") == "npu"``
      DML only has gpu            → ``default_device_for_ep("DmlExecutionProvider") == "gpu"``

    Args:
        ep: Full EP name (e.g. ``"QNNExecutionProvider"``).

    Returns:
        Device category string, or ``None`` if *ep* is not in the catalog.
    """
    return next((s.device for s in EP_DEVICE_SPECS if s.ep == ep), None)


def default_ep_for_device(device: str) -> str | None:
    """First catalog variant whose device matches. Replaces ``_DEVICE_TO_PROVIDER``.

    Order in ``EP_DEVICE_SPECS`` encodes preference:

    * Among npu variants, QNN comes first.
    * Among gpu variants, OpenVINO comes first.

    NOTE: unlike the old ``_DEVICE_TO_PROVIDER``, this returns the full canonical
    EP name.  Callers that need a short name must call
    ``short_ep_name(default_ep_for_device(device))``.

    Args:
        device: Device category (e.g. ``"npu"``, ``"gpu"``, ``"cpu"``).

    Returns:
        Full EP name, or ``None`` if *device* is not in the catalog.
    """
    return next((s.ep for s in EP_DEVICE_SPECS if s.device == device), None)


def eps_for_device(device: str) -> frozenset[str]:
    """All canonical EP names in the catalog that target the given device.

    Replaces inline hardcoded EP lists (e.g. ``candidate_eps`` in
    ``commands/build.py``).  Returns canonical (full) names — callers
    needing short names use :func:`short_ep_name` per element.

    Args:
        device: Device category (``"npu"``, ``"gpu"``, ``"cpu"``).
            Case-insensitive.

    Returns:
        Frozenset of canonical EP names for that device.  Returns an empty
        frozenset for unknown devices (no raise — callers can check membership).
    """
    d = device.lower()
    return frozenset(s.ep for s in EP_DEVICE_SPECS if s.device == d)


def ep_to_device(ep: str) -> str:
    """Map an EP short name to its device category.

    Args:
        ep: EP short name (e.g. ``"qnn"``, ``"dml"``).

    Returns:
        Device category string: ``"npu"``, ``"gpu"``, or ``"cpu"``.

    Raises:
        ValueError: If *ep* is not a recognised EP short name.
    """
    ep_full = expand_ep_name(ep)
    device = default_device_for_ep(ep_full)
    if device is None:
        raise ValueError(f"Unknown EP '{ep}'. Known EPs: {sorted(VALID_EPS)}")
    return device


# --- resolve_device_category -----------------------------------------------


def resolve_device_category(device: str = "auto") -> tuple[str, list[str]]:
    """Resolve a device hint to (category, candidate EP names).

    Args:
        device: "auto", "npu", "gpu", or "cpu".

    Returns:
        (chosen_device, available_devices_list)

    Raises:
        ValueError: If device is not recognized.
    """
    device = device.lower()

    if device != "auto" and device not in VALID_DEVICES:
        raise ValueError(f"Unknown device '{device}'. Expected 'auto', 'npu', 'gpu', or 'cpu'.")

    from ..sysinfo.hardware import get_available_devices
    from .ep_registry import available_eps as _available_eps

    available_devices = get_available_devices()
    _eps = _available_eps()

    if not _eps:
        logger.warning(
            "No execution providers detected. Falling back to CPU. "
            "Install onnxruntime or Windows App SDK for EP discovery."
        )

    if device == "auto":
        # Walk priority list, pick first device with a matching EP.
        # eps_for_device returns canonical EP names from the catalog —
        # includes OpenVINO for npu/gpu/cpu (the old _DEVICE_EP_MAP excluded it).
        for dev in available_devices:
            if any(ep in _eps for ep in eps_for_device(dev)):
                return dev, available_devices
        # Fallback: CPU is always valid
        return "cpu", available_devices

    # Explicit device requested -- warn if no compatible EP
    compatible_eps = eps_for_device(device)
    if not any(ep in _eps for ep in compatible_eps):
        logger.warning(
            "Device '%s' requested but no compatible EP found. "
            "Compatible EPs: %s. Available EPs: %s",
            device,
            sorted(compatible_eps),
            sorted(_eps),
        )
    return device, available_devices


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
        ep only      -> default_device_for_ep(ep) gives device
        device only  -> default_ep_for_device(device) gives ep
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
        # Auto-detect: pick strongest available device via local resolver
        resolved_device_str, _ = resolve_device_category(device="auto")
        device = resolved_device_str

    if ep is not None and device is None:
        # ep given, device missing — infer from catalog
        ep_full = expand_ep_name(ep)
        deduced = default_device_for_ep(ep_full)
        if deduced is None:
            raise ValueError(f"Cannot deduce device for EP '{ep}'. Known EPs: {sorted(VALID_EPS)}")
        device = deduced
        logger.debug("Deduced device=%r from ep=%r", device, ep)

    if device is not None and ep is None:
        # device given, ep missing — infer default EP from catalog
        device_lower = device.lower()
        if device_lower not in VALID_DEVICES:
            raise ValueError(f"Unknown device '{device}'. Expected one of: {sorted(VALID_DEVICES)}")
        default_ep_full = default_ep_for_device(device_lower)
        # cpu maps to CPUExecutionProvider; use its short name for consistency
        ep = short_ep_name(default_ep_full) if default_ep_full is not None else "cpu"
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
