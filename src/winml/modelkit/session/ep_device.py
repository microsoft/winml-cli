# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# src/winml/modelkit/session/ep_device.py
"""EPDeviceTarget descriptor + resolution helpers + exception taxonomy.

EPDeviceTarget is a pure-data identifier for one (EP, hardware-device) target.
It is frozen, JSON-serializable, and has no runtime dependency on ORT.
Construction is performed by the CLI parser / JSON loader / tests, then
fed through resolve_device(target) which fills in any ``"auto"`` axes.
The OrtEpDevice handle is re-derived inside session.py at session-build
time and never stored on EPDeviceTarget itself.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Final


logger = logging.getLogger(__name__)


# --- exceptions ------------------------------------------------------------


class WinMLEPNotDiscovered(Exception):  # noqa: N818
    """EP plugin is not in the catalog or WINMLCLI_EP_PATH."""


class WinMLEPRegistrationFailed(Exception):  # noqa: N818
    """ort.register_execution_provider_library raised."""


class DeviceNotFound(Exception):  # noqa: N818
    """EP registered, but no OrtEpDevice matches the descriptor."""


class AmbiguousMatch(Exception):  # noqa: N818
    """Multiple OrtEpDevices match the descriptor after dedup (bug signal)."""


class WinMLEPMonitorMismatch(Exception):  # noqa: N818
    """Monitor.ep_name does not agree with EPDeviceTarget.ep."""


class UnknownListingPick(Exception):  # noqa: N818
    """Raised when target.source tag doesn't match any discovered EPEntry for target.ep.

    Args:
        ep_name: The EP short or full name the user requested.
        source_tag: The source tag string from target.source (or '@<tag>' in CLI).
    """

    def __init__(self, ep_name: str, source_tag: str) -> None:
        self.ep_name = ep_name
        self.source_tag = source_tag
        super().__init__(
            f"No discovered EPEntry for ep={ep_name!r} with source tag {source_tag!r}. "
            f"Run 'winml sys --list-ep' to see available sources."
        )


class IncompatibleListingPick(Exception):  # noqa: N818
    """Raised when target.source matched an EPEntry that is incompatible with the host.

    Args:
        ep_name: The EP short or full name.
        source_tag: The source tag string.
        reason: Human-readable reason for incompatibility.
    """

    def __init__(self, ep_name: str, source_tag: str, reason: str) -> None:
        self.ep_name = ep_name
        self.source_tag = source_tag
        self.reason = reason
        super().__init__(
            f"EP {ep_name!r} with source tag {source_tag!r} is not compatible "
            f"with this host: {reason}"
        )


class AmbiguousListingPick(Exception):  # noqa: N818
    """Raised when target.source matches multiple EPEntries (defensive bug signal).

    Args:
        ep_name: The EP short or full name.
        source_tag: The source tag string.
        candidate_count: How many EPEntries matched.
    """

    def __init__(self, ep_name: str, source_tag: str, candidate_count: int) -> None:
        self.ep_name = ep_name
        self.source_tag = source_tag
        self.candidate_count = candidate_count
        super().__init__(
            f"Ambiguous source pick for ep={ep_name!r} with tag {source_tag!r}: "
            f"{candidate_count} candidates matched (expected 1)"
        )


# --- EP-name short<->full helpers -----------------------------------------
# These live above EPDeviceTarget so its __post_init__ can validate ep names
# against the known catalog without forward references.


_SHORT_TO_FULL: Final[dict[str, str]] = {
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "nvtensorrtrtx": "NvTensorRtRtxExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def expand_ep_name(name: str) -> str:
    """Expand a short EP name to its full form; passthrough if already full.

    "xxx" is the short form of "xxxExecutionProvider" (case-folded for
    lookup). Names that don't match a short alias are passed through
    unchanged — downstream registration will fail loudly if the spelling
    doesn't match ORT's canonical name.
    """
    full = _SHORT_TO_FULL.get(name.lower())
    if full is not None:
        return full
    return name


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


# --- validation closed-sets -----------------------------------------------
# These three closed sets are the canonical authority used by
# EPDeviceTarget.__post_init__ for construction-time validation.
# - VALID_DEVICES: the 3 device categories ORT enumerates.
# - VALID_SOURCE_TAGS: the 7 canonical EPSource origin tags (see
#   docs/design/session/3_design_classes.md §4).
# - known_ep_short_names(): derived from _SHORT_TO_FULL (no hardcoded list,
#   per CLAUDE.md cardinal rule #1).

VALID_DEVICES: Final[frozenset[str]] = frozenset({"npu", "gpu", "cpu"})

VALID_SOURCE_TAGS: Final[frozenset[str]] = frozenset(
    {
        "bundled",
        "pypi",
        "nuget",
        "msix-microsoft",
        "msix-workload",
        "winml-catalog",
        "directory",
    }
)


def known_ep_short_names() -> frozenset[str]:
    """Set of EP short names registered in ``_SHORT_TO_FULL``.

    Derived (not hardcoded) per CLAUDE.md cardinal rule #1 — adding a new
    EP to ``_SHORT_TO_FULL`` automatically expands the validation set.
    """
    return frozenset(_SHORT_TO_FULL.keys())


# --- dataclass -------------------------------------------------------------


@dataclass(frozen=True)
class EPDeviceTarget:
    """Pure-data identifier of one (EP, hardware-device) binding target.

    No runtime hardware fingerprints (vendor_id/device_id/vendor) — those
    belong on :class:`WinMLDevice`, the ``OrtEpDevice`` adapter. This is
    the user-craftable intent type: "I want EP X on device class Y, optionally
    via source tag Z." The OrtEpDevice handle is resolved at session-build
    time by :meth:`WinMLEPRegistry.auto_device`.

    Construction-time validation (see ``__post_init__``):
      - ``device``: must be ``"auto"`` or in :data:`VALID_DEVICES`
      - ``ep``:     must be ``"auto"`` or a known short/full name from
                    :data:`_SHORT_TO_FULL`
      - ``source``: must be ``None`` or in :data:`VALID_SOURCE_TAGS`
    """

    ep: str
    device: str
    source: str | None = None

    def __post_init__(self) -> None:
        # Frozen dataclass — must use object.__setattr__ to mutate.
        # Normalize device casing (existing behavior — preserve).
        if self.device != self.device.lower():
            object.__setattr__(self, "device", self.device.lower())

        # Validate device class.
        if self.device != "auto" and self.device not in VALID_DEVICES:
            raise ValueError(
                f"Unknown device {self.device!r}; "
                f"expected one of {sorted(VALID_DEVICES)} or 'auto'"
            )

        # Validate EP name (short OR full).
        if (
            self.ep != "auto"
            and self.ep.lower() not in known_ep_short_names()
            and self.ep not in _FULL_TO_SHORT
        ):
            raise ValueError(
                f"Unknown EP {self.ep!r}; "
                f"expected one of {sorted(known_ep_short_names())} or 'auto' "
                f"(also accepts full names like 'OpenVINOExecutionProvider')"
            )

        # Validate source tag.
        if self.source is not None and self.source not in VALID_SOURCE_TAGS:
            raise ValueError(
                f"Unknown source tag {self.source!r}; "
                f"expected one of {sorted(VALID_SOURCE_TAGS)} or None"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON round-trip."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EPDeviceTarget:
        """Rehydrate from a dict produced by to_dict.

        Legacy keys ``vendor_id``/``device_id``/``vendor`` are silently
        ignored (forward-compat for persisted JSON written before the
        Batch C strip).
        """
        return cls(
            ep=d["ep"],
            device=d["device"],
            source=d.get("source"),
        )


# --- EP / device taxonomy --------------------------------------------------
# Single authoritative source: the EPDeviceSpec catalog.
# config/precision.py imports helpers from here (via the session facade).


@dataclass(frozen=True, kw_only=True, slots=True)
class EPDeviceSpec:
    """One supported (EP, device) target in the catalog.

    Distinct from EPDeviceTarget:
      - EPDeviceSpec is the *kind-of-target* (machine-independent).
      - EPDeviceTarget is the *user intent* (machine-independent pair, plus
        optional source tag). Runtime hardware fingerprints live on
        :class:`WinMLDevice`.
    Many EPDeviceTargets map to one EPDeviceSpec.
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
    EPDeviceSpec(ep="NvTensorRtRtxExecutionProvider", device="gpu"),
)

# O(1) lookup cache built from the ordered catalog.
_BY_KEY: Final[dict[tuple[str, str], EPDeviceSpec]] = {(s.ep, s.device): s for s in EP_DEVICE_SPECS}

# Public frozenset for callers that only need EP membership tests.
# VALID_EPS uses short names for backward compatibility with callers that pass
# short forms (e.g. "qnn", "dml").  The catalog uses full names internally.
# VALID_DEVICES is defined above (alongside EPDeviceTarget validation) and the
# closed set is invariant against this catalog by construction — every
# EP_DEVICE_SPECS row's device must be in VALID_DEVICES.
VALID_EPS: Final[frozenset[str]] = frozenset({short_ep_name(s.ep) for s in EP_DEVICE_SPECS})


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
    """First catalog variant whose device matches AND whose EP is registered on this host.

    Walks ``EP_DEVICE_SPECS`` in order and returns the first ``spec.ep`` that is
    also in ``available_eps()`` (from :mod:`session.ep_registry`). Returns
    ``None`` when no catalog entry for the requested device has a registered EP
    — the caller decides whether to raise, fall back, or treat as a no-op.

    The static catalog order encodes *preference among installed EPs*, not
    unconditional defaults. Without the registration filter, this returns the
    catalog primary even on hosts where it isn't installed (e.g. QNN on an
    OpenVINO-only box). See ``docs/design/session/3_design_ep.md`` §6.4 for the
    rationale.

    Historical note: replaces ``_DEVICE_TO_PROVIDER`` and returns the full
    canonical EP name; callers that need a short name must call
    ``short_ep_name(default_ep_for_device(device))``.

    Args:
        device: Device category (e.g. ``"npu"``, ``"gpu"``, ``"cpu"``).

    Returns:
        Full EP name of the first registered catalog match, or ``None``.
    """
    # Lazy import: ep_registry imports from this module at top level, so
    # importing it here avoids the circular-import cycle.
    from .ep_registry import available_eps

    eps = available_eps()
    return next(
        (s.ep for s in EP_DEVICE_SPECS if s.device == device and s.ep in eps),
        None,
    )


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


# --- auto-detect helper ----------------------------------------------------


def auto_detect_device() -> str:
    """Pick the strongest hardware-and-EP-backed device on this host.

    Walks sysinfo's available-devices priority list, returning the first
    entry whose catalog EPs are actually registered. Falls back to "cpu"
    when no plugin EPs are discovered.
    """
    from ..sysinfo.hardware import get_available_devices
    from .ep_registry import available_eps as _available_eps

    available_devices = get_available_devices()
    _eps = _available_eps()

    if not _eps:
        logger.warning(
            "No execution providers detected. Falling back to CPU. "
            "Install onnxruntime or Windows App SDK for EP discovery."
        )

    for dev in available_devices:
        if any(ep in _eps for ep in eps_for_device(dev)):
            return dev
    return "cpu"


# --- resolution ------------------------------------------------------------


def resolve_device(target: EPDeviceTarget) -> EPDeviceTarget:
    """Pure-deduction resolver: EPDeviceTarget -> EPDeviceTarget.

    Takes a typed :class:`EPDeviceTarget` intent (possibly carrying
    ``"auto"`` on either axis and ``source=None``) and returns a
    self-describing :class:`EPDeviceTarget` whose ``ep`` and ``device``
    are concrete. ``source`` is passed through unchanged in Scenario A
    (``target.source is None``); in Scenario B (``target.source is not
    None``) it is validated against the discovered EPEntry set.

    Validation against ``available_eps()`` is cheap (no DLL load); the
    DLL-load + runtime-handle binding lives in
    :meth:`WinMLEPRegistry.auto_device` — see
    ``docs/design/session/2_coreloop.md`` §4.3.

    Deduction matrix (matches the prior loose-string ``resolve_device``):
        both concrete -> validate + return
        ep only       -> ``default_device_for_ep(ep)`` gives device
        device only   -> ``default_ep_for_device(device)`` gives ep
                         (filtered by ``available_eps()``)
        neither       -> ``auto_detect_device()`` picks device,
                         then fall through to the device-only branch

    Args:
        target: User intent. ``target.ep`` and ``target.device`` may be
            the literal ``"auto"``; ``target.source`` may be ``None`` or
            a canonical source tag.

    Returns:
        Resolved :class:`EPDeviceTarget` with no ``"auto"`` values and a
        validated ``source`` (when set by the caller).

    Raises:
        ValueError: Unknown EP or device after deduction, or no
            registered EP backs the requested device.
        UnknownListingPick: ``target.source`` is set but no discovered
            EPEntry for ``target.ep`` matches that source tag (Scenario B).
    """
    # --- Step 1: normalize "auto" sentinels into None for the matrix ----
    ep: str | None = None if target.ep == "auto" else target.ep
    device: str | None = (
        None if target.device == "auto" or target.device is None else target.device.lower()
    )
    source = target.source  # passthrough; validated in Step 4

    # --- Step 2: deduction matrix (same body as the prior signature) ----
    if ep is None and device is None:
        # Auto-detect: pick strongest available device on this host.
        device = auto_detect_device()

    if ep is not None and device is None:
        # ep given, device missing — infer from catalog.
        ep_full = expand_ep_name(ep)
        deduced = default_device_for_ep(ep_full)
        if deduced is None:
            raise ValueError(f"Cannot deduce device for EP '{ep}'. Known EPs: {sorted(VALID_EPS)}")
        device = deduced
        logger.debug("Deduced device=%r from ep=%r", device, ep)

    if device is not None and ep is None:
        # device given, ep missing — infer default EP from catalog.
        device_lower = device.lower()
        if device_lower not in VALID_DEVICES:
            raise ValueError(f"Unknown device '{device}'. Expected one of: {sorted(VALID_DEVICES)}")
        default_ep_full = default_ep_for_device(device_lower)
        if default_ep_full is None:
            raise ValueError(
                f"No registered EP for device {device_lower!r}. "
                f"Install a plugin EP that targets this device, or pass --ep explicitly."
            )
        ep = short_ep_name(default_ep_full)
        logger.debug("Deduced ep=%r from device=%r", ep, device_lower)

    # At this point both ep and device are non-None strings (type-checker aid).
    assert ep is not None
    assert device is not None

    ep_full = expand_ep_name(ep)
    device_lower = device.lower()

    # --- Step 3: source validation (Scenario B only) --------------------
    # When the caller pinned a source tag, validate it against the
    # discovery layer. Defer IncompatibleListingPick to a later batch.
    if source is not None:
        from .ep_registry import WinMLEPRegistry, _entry_source_tag

        registry = WinMLEPRegistry.instance()
        matches = [
            e
            for e in registry.entries_for(ep_full)
            if _entry_source_tag(e) == source
        ]
        if not matches:
            raise UnknownListingPick(ep_full, source)

    # --- Step 4: return self-describing resolved target -----------------
    return EPDeviceTarget(ep=ep_full, device=device_lower, source=source)
