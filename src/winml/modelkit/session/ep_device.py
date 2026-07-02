# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# src/winml/modelkit/session/ep_device.py
"""EP-device types: intent + catalog + runtime adapter + resolution.

This module hosts three layers of (EP, device) types:

- ``EPDeviceTarget`` — pure-data user intent. Frozen, JSON-serializable,
  no runtime dependency on ORT. Built by CLI parser / JSON loader / tests,
  then fed through :func:`resolve_device` which fills any ``"auto"`` axes.
- ``EPDeviceSpec`` — catalog row: machine-independent (EP, device) target
  plus default provider options. The ``EP_DEVICE_SPECS`` constant is the
  single authoritative catalog.
- ``WinMLDevice`` — vendor-normalized runtime adapter over
  ``ort.OrtEpDevice``. Single concrete class; per-EP metadata schemas
  handled by internal dispatch on ``self._ort.ep_name``. OS-bound:
  cannot be hand-constructed; instantiated via ``WinMLDevice(handle)``
  inside :meth:`WinMLEPRegistry.register_ep` after a successful ORT
  registration produced the underlying ``OrtEpDevice``. See
  docs/design/session/4_winml_device.md for the full dispatch design.

The ``OrtEpDevice`` handle is re-derived inside session.py at session-build
time and never stored on EPDeviceTarget itself.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import onnxruntime as ort


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# Bidirectional bridge between lowercase short device strings ("cpu", "gpu",
# "npu" — the project-wide convention used by ``VALID_DEVICES``,
# ``EPDeviceTarget.device``, ``EPDeviceSpec.device``) and ORT's
# :class:`onnxruntime.OrtHardwareDeviceType` enum used by the runtime checker
# pipeline (``EPChecker`` and friends).
#
# Previously these lived in ``utils/constants.py`` with **uppercase** keys
# (``"CPU"``, ``"GPU"``, ``"NPU"``) — a silent casing-mismatch footgun
# whenever a lowercase device string from the session taxonomy reached the
# lookup. Unified on lowercase by T-16 so the whole codebase agrees.
DEVICE_TO_DEVICE_TYPE: Final[dict[str, ort.OrtHardwareDeviceType]] = {
    "cpu": ort.OrtHardwareDeviceType.CPU,
    "gpu": ort.OrtHardwareDeviceType.GPU,
    "npu": ort.OrtHardwareDeviceType.NPU,
}

DEVICE_TYPE_TO_DEVICE: Final[dict[ort.OrtHardwareDeviceType, str]] = {
    ort.OrtHardwareDeviceType.CPU: "cpu",
    ort.OrtHardwareDeviceType.GPU: "gpu",
    ort.OrtHardwareDeviceType.NPU: "npu",
}


# --- exceptions ------------------------------------------------------------


class WinMLEPNotDiscovered(Exception):  # noqa: N818
    """EP plugin is not in the catalog or WINMLCLI_EP_PATH."""


class _LoadFailure(NamedTuple):
    code: int | None
    reason: str


# ORT emits three shapes for LoadLibrary failures — all carry a Win32 code:
#   "... with error code: 127"          (OpenVINO/VitisAI shim path)
#   '(Error 193: "...")'                (arch mismatch, missing dep)
#   '(Error 1114: init failed)'         (DllMain failure)
_ORT_ERROR_CODE_RE = re.compile(r"(?:error code:?\s*|\(Error\s+)(\d+)", re.IGNORECASE)


_LOAD_FAILURE_REASONS: dict[int, str] = {
    2: "file not found (Win32 2)",
    5: "access denied (Win32 5)",
    126: "dependency DLL not found on disk (Win32 126)",
    127: "symbol not resolved in a dependency DLL (Win32 127)",
    193: "wrong architecture — ARM64 DLL in an x64 process (Win32 193)",
    1114: "DllMain returned failure (Win32 1114)",
}


def _parse_ort_load_failure(text: str) -> _LoadFailure:
    """Extract ``(code, reason)`` from an ORT DLL-load error message."""
    if not text:
        return _LoadFailure(code=None, reason="(no error message)")
    m = _ORT_ERROR_CODE_RE.search(text)
    if m is None:
        first_line = text.splitlines()[0] if "\n" in text else text
        return _LoadFailure(code=None, reason=first_line[:200])
    code = int(m.group(1))
    return _LoadFailure(
        code=code,
        reason=_LOAD_FAILURE_REASONS.get(code, f"DLL load failed (Win32 {code})"),
    )


def _read_pe_file_version(dll_path: Path | str) -> str | None:
    """DLL's PE ``FileVersion`` — reads VS_VERSIONINFO off disk, no LoadLibrary."""
    if sys.platform != "win32":
        return None
    path = Path(dll_path)
    if not path.is_file():
        return None
    try:
        from win32api import GetFileVersionInfo, HIWORD, LOWORD

        info = GetFileVersionInfo(str(path), "\\")
        ms, ls = info["FileVersionMS"], info["FileVersionLS"]
        return f"{HIWORD(ms)}.{LOWORD(ms)}.{HIWORD(ls)}.{LOWORD(ls)}"
    except Exception:  # noqa: BLE001 — fallback is best-effort
        return None


class WinMLEPRegistrationFailed(Exception):  # noqa: N818
    """``ort.register_execution_provider_library`` raised.

    Carries structured attribution (``code``, ``reason``, ``dll_path``,
    ``fallback_version``) parsed at construction so callers render
    ``[failed]`` rows without re-parsing the ORT message.
    """

    def __init__(self, message: str, *, dll_path: Path | None = None) -> None:
        super().__init__(message)
        lf = _parse_ort_load_failure(message)
        self.code: int | None = lf.code
        self.reason: str = lf.reason
        self.dll_path: Path | None = dll_path
        self.fallback_version: str | None = (
            _read_pe_file_version(dll_path) if dll_path is not None else None
        )


class DeviceNotFound(Exception):  # noqa: N818
    """EP registered, but no OrtEpDevice matches the descriptor."""


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


# --- EP-name short<->full helpers -----------------------------------------
# These live above EPDeviceTarget so its __post_init__ can validate ep names
# against the known catalog without forward references.


_SHORT_TO_FULL: Final[dict[str, str]] = {
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "nvtensorrtrtx": "NvTensorRtRtxExecutionProvider",
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


def _ep_short_or_none(ep_full: str) -> str | None:
    """Map a full EP name to its short form, collapsing ``"cpu"`` to ``None``.

    ``CPUExecutionProvider`` has no compile step, so callers that wire
    a compile-stage EP off a resolved device want ``None`` (= "no compile
    stage"), not the short string ``"cpu"``. The two consumers
    (``config/build.py`` STEP 4 auto/auto branch and ``config/precision.py``
    ``resolve_precision``) previously inlined the same
    ``_short if _short != "cpu" else None`` collapse — centralized here
    so the rule lives in exactly one place.
    """
    short = short_ep_name(ep_full)
    return None if short == "cpu" else short


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
    # Order encodes first-match deduction preference per device. Plugin
    # (vendor-optimal) EPs come first; built-ins (CPU / DML / Azure) trail
    # as fallbacks. This matches the design intent stated in
    # ``ep_registry.py`` where synthetic ``BuiltinSource`` entries are
    # appended AFTER filesystem discovery so "built-ins are lowest priority
    # — only used when no plugin provided the EP".
    #
    # Within each device group ordering is:
    #   npu:  QNN → OpenVINO → VitisAI                          (no NPU built-in)
    #   gpu:  OpenVINO → MIGraphX → Tensorrt → NvTensorRtRtx → QNN(2ary) → DML
    #   cpu:  OpenVINO → QNN(2ary) → CPU
    # ---- Plugin EPs (vendor-optimal — preferred) ----
    EPDeviceSpec(
        ep="QNNExecutionProvider",
        device="npu",
        default_provider_options={
            # Verified 2026-05-13: +3x throughput on ResNet-50 vs default mode
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        },
    ),
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="npu"),
    EPDeviceSpec(ep="VitisAIExecutionProvider", device="npu"),
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="MIGraphXExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="TensorrtExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="NvTensorRtRtxExecutionProvider", device="gpu"),
    EPDeviceSpec(ep="OpenVINOExecutionProvider", device="cpu"),
    # ---- QNN secondary (Snapdragon boxes without vendor-optimal alternatives) ----
    EPDeviceSpec(ep="QNNExecutionProvider", device="gpu"),  # TODO: measure
    EPDeviceSpec(ep="QNNExecutionProvider", device="cpu"),
    # ---- Built-in fallbacks (last per registry design intent) ----
    EPDeviceSpec(ep="DmlExecutionProvider", device="gpu"),  # cross-vendor GPU fallback
    EPDeviceSpec(ep="CPUExecutionProvider", device="cpu"),  # always-available CPU fallback
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
    from ..ep_path import EP_CATALOG
    from .ep_registry import WinMLEPRegistry

    eps = WinMLEPRegistry.instance().available_eps()
    # is_compatible may raise RuntimeError on headless servers (transitively
    # via _get_detected_vendors); treat that as "no compatible EP" which is
    # what the None return already signals to callers.
    try:
        return next(
            (
                s.ep
                for s in EP_DEVICE_SPECS
                if s.device == device
                and s.ep in eps                          # L0: discovered
                and EP_CATALOG.is_compatible(s.ep)       # L2: vendor-compatible
            ),
            None,
        )
    except RuntimeError as e:
        logger.warning(
            "Hardware detection failed (%s); no compatible EP for device %r.",
            e,
            device,
        )
        return None


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
    from ..ep_path import EP_CATALOG
    from ..sysinfo import get_available_devices
    from .ep_registry import WinMLEPRegistry

    available_devices = get_available_devices()
    _eps = WinMLEPRegistry.instance().available_eps()

    if not _eps:
        logger.warning(
            "No execution providers detected. Falling back to CPU. "
            "Install onnxruntime or Windows App SDK for EP discovery."
        )

    # Pick the strongest device with at least one EP that is both
    # discovered (L0) AND vendor-compatible (L2 — EP_CATALOG.is_compatible).
    # is_compatible transitively calls _get_detected_vendors which raises
    # RuntimeError on headless servers where GPU/NPU WMI queries fail; treat
    # that as "no compatibility signal → CPU fallback" so the click command
    # layer never sees an unhandled traceback.
    try:
        for dev in available_devices:
            if any(
                ep in _eps and EP_CATALOG.is_compatible(ep)
                for ep in eps_for_device(dev)
            ):
                return dev
    except RuntimeError as e:
        logger.warning(
            "Hardware detection failed (%s); falling back to CPU. "
            "Pass --ep and --device explicitly to bypass auto-detect.",
            e,
        )
    return "cpu"


# --- resolution ------------------------------------------------------------
def resolve_device(target: EPDeviceTarget) -> EPDeviceTarget:
    """Pure-deduction resolver: EPDeviceTarget -> EPDeviceTarget.

    Takes a typed :class:`EPDeviceTarget` intent (possibly carrying
    ``"auto"`` on either axis) and returns a self-describing
    :class:`EPDeviceTarget` whose ``ep`` and ``device`` are concrete.
    ``source`` passes through unchanged; source-tag validation against
    discovered entries lives in :meth:`WinMLEPRegistry.auto_device`
    (Path A's registration step) — this function does no filesystem
    scan, no DLL load, and no registry I/O.

    Resolution order (resolve device first, then ep — device-only path
    consults ``default_ep_for_device`` which already filters by
    ``available_eps()``):

    - ``device == "auto"`` and ``ep == "auto"`` → ``auto_detect_device()``
      picks the device, then fall through to the device-only branch
    - ``device == "auto"`` and ``ep`` given → ``default_device_for_ep(ep)``
    - ``ep == "auto"`` and ``device`` given → ``default_ep_for_device(device)``
      (registration-aware filter)
    - both given → validate and return

    Args:
        target: User intent. ``target.ep`` and ``target.device`` may be
            the literal ``"auto"``; ``target.source`` may be ``None`` or
            a canonical source tag.

    Returns:
        Resolved :class:`EPDeviceTarget` with no ``"auto"`` values.
        ``source`` is passed through unchanged (validation deferred to
        ``auto_device``).

    Raises:
        ValueError: Unknown EP or device after deduction, or no
            registered EP backs the requested device.
    """
    ep = target.ep
    device = target.device

    # --- Resolve device axis first --------------------------------------
    if device == "auto":
        if ep == "auto":
            device = auto_detect_device()
        else:
            deduced = default_device_for_ep(expand_ep_name(ep))
            if deduced is None:
                raise ValueError(
                    f"Cannot deduce device for EP '{ep}'. Known EPs: {sorted(VALID_EPS)}"
                )
            device = deduced
            logger.debug("Deduced device=%r from ep=%r", device, ep)
    else:
        device = device.lower()
        if device not in VALID_DEVICES:
            raise ValueError(
                f"Unknown device '{device}'. Expected one of: {sorted(VALID_DEVICES)}"
            )

    # --- Resolve ep axis (device is concrete by this point) -------------
    if ep == "auto":
        default_ep_full = default_ep_for_device(device)
        if default_ep_full is None:
            raise ValueError(
                f"No registered EP for device {device!r}. "
                f"Install a plugin EP that targets this device, or pass --ep explicitly."
            )
        ep = short_ep_name(default_ep_full)
        logger.debug("Deduced ep=%r from device=%r", ep, device)

    # --- Final validation + return --------------------------------------
    ep_full = expand_ep_name(ep)
    resolved = EPDeviceTarget(ep=ep_full, device=device, source=target.source)
    logger.info(
        "resolve_device: %s/%s%s -> %s/%s%s",
        target.ep, target.device,
        f"@{target.source}" if target.source else "",
        resolved.ep, resolved.device,
        f"@{resolved.source}" if resolved.source else "",
    )
    return resolved


# --- runtime adapter: WinMLDevice -----------------------------------------


class WinMLDevice:
    """Vendor-normalized adapter over ort.OrtEpDevice."""

    def __init__(self, ort_device: ort.OrtEpDevice) -> None:
        self._ort = ort_device

    # ---- common properties (no per-EP dispatch needed) ------------------

    @property
    def ep_name(self) -> str:
        """Canonical EP name reported by ORT (e.g. ``"OpenVINOExecutionProvider"``)."""
        return self._ort.ep_name

    @property
    def device_type(self) -> str:
        """'NPU' | 'GPU' | 'CPU' - uppercased from device.type.name."""
        return self._ort.device.type.name.upper()

    @property
    def hardware_name(self) -> str:
        """Prefers ep_metadata['FULL_DEVICE_NAME']; falls back to device.metadata['Description']."""
        return (
            self._ort.ep_metadata.get("FULL_DEVICE_NAME")
            or self._ort.device.metadata.get("Description")
            or "<unknown>"
        )

    @property
    def vendor(self) -> str:
        """Hardware vendor string (e.g. ``"Intel"``) from the underlying OrtEpDevice."""
        return self._ort.device.vendor

    @property
    def ep_vendor(self) -> str:
        """EP vendor string (e.g. ``"Microsoft"``) from the underlying OrtEpDevice."""
        return self._ort.ep_vendor

    @property
    def library_path(self) -> str | None:
        """Plugin DLL path from ``ep_metadata['library_path']``, or ``None`` if unset."""
        return self._ort.ep_metadata.get("library_path") or None

    @property
    def ort_handle(self) -> ort.OrtEpDevice:
        """Public read-only accessor for the underlying ORT handle.

        For external callers (analyze/, future plugins) that need to pass
        the raw OrtEpDevice to APIs like SessionOptions.add_provider_for_devices
        or ort.ModelCompiler. Internal session/ code reads self._ort directly.
        """
        return self._ort

    # ---- vendor-specific properties - internal dispatch on ep_name ------

    @property
    def memory_bytes(self) -> int | None:
        """Total device memory in bytes, or None when not applicable / unknown."""
        ep = self._ort.ep_name
        device_type = self.device_type
        # OpenVINO uses NPU_DEVICE_TOTAL_MEM_SIZE / GPU_DEVICE_TOTAL_MEM_SIZE
        if "OpenVINO" in ep:
            key = {
                "NPU": "NPU_DEVICE_TOTAL_MEM_SIZE",
                "GPU": "GPU_DEVICE_TOTAL_MEM_SIZE",
            }.get(device_type)
            if key:
                raw = self._ort.ep_metadata.get(key)
                if raw:
                    try:
                        return int(raw)
                    except ValueError:
                        return None
        # DML uses device.metadata['DxgiVideoMemory'] (e.g., '128 MB')
        if ep == "DmlExecutionProvider":
            raw = self._ort.device.metadata.get("DxgiVideoMemory", "")
            # Parse '<N> MB' / '<N> GB' / '<N> B' to bytes (cheap, best-effort)
            parts = raw.split()
            if len(parts) == 2:
                try:
                    n = int(parts[0])
                    unit = parts[1].upper()
                    multiplier = {
                        "B": 1,
                        "KB": 1024,
                        "MB": 1024**2,
                        "GB": 1024**3,
                    }.get(unit, 0)
                    if multiplier:
                        return n * multiplier
                except ValueError:
                    pass
        return None

    @property
    def architecture(self) -> str | None:
        """Short architecture string, or None."""
        ep = self._ort.ep_name
        if "OpenVINO" in ep:
            raw = self._ort.ep_metadata.get("DEVICE_ARCHITECTURE")
            if not raw:
                return None
            # 'GPU: vendor=0x8086 arch=v20.4.4' -> 'v20.4.4'; 'intel64' passes through
            if "arch=" in raw:
                return raw.split("arch=", 1)[1].strip()
            return raw
        return None

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Normalized capability flags. Empty tuple when unknown."""
        ep = self._ort.ep_name
        if "OpenVINO" in ep:
            raw = self._ort.ep_metadata.get("OPTIMIZATION_CAPABILITIES", "")
            tokens = raw.split()
            rewrites = {
                "GPU_HW_MATMUL": "MatMul",
                "GPU_USM_MEMORY": "USM",
                "EXPORT_IMPORT": "",
            }
            return tuple(rewrites.get(t, t) for t in tokens if rewrites.get(t, t))
        return ()

    @property
    def driver_version(self) -> str | None:
        """NPU driver version string, or ``None`` when unknown / not applicable."""
        ep = self._ort.ep_name
        if "OpenVINO" in ep and self.device_type == "NPU":
            return self._ort.ep_metadata.get("NPU_DRIVER_VERSION")
        return None

    @property
    def compiler_version(self) -> str | None:
        """NPU compiler version string, or ``None`` when unknown / not applicable."""
        ep = self._ort.ep_name
        if "OpenVINO" in ep and self.device_type == "NPU":
            return self._ort.ep_metadata.get("NPU_COMPILER_VERSION")
        return None

    # ---- introspection + display ----------------------------------------

    def available_metadata(self) -> Mapping[str, str]:
        """Raw ep_metadata mapping - for --verbose / debug dumps."""
        return dict(self._ort.ep_metadata)

    def device_facts(self) -> tuple[str, ...]:
        """Device-intrinsic facts for the *Available Devices* section.

        Returns Architecture + Driver — values keyed off the underlying
        silicon and kernel driver, invariant across the EPs that bind to
        the device. Empty entries are skipped.

        Contrast with :meth:`ep_facts` (Memory + Capabilities), which
        reflect this specific EP runtime's view of the device and can
        differ between sources of the same EP and between different EPs
        on the same hardware. See ``docs/design/session/4_winml_device.md``
        §4 + §4.1 for the attribute-attribution table.
        """
        out: list[str] = []
        if (a := self.architecture) is not None:
            out.append(f"Architecture: {a}")
        if (d := self.driver_version) is not None:
            out.append(f"Driver: {d}")
        return tuple(out)

    def ep_facts(self) -> tuple[str, ...]:
        """EP-mediated facts for the *Available Execution Providers* section.

        Returns Memory + Capabilities — values keyed off this specific
        EP runtime's view of the device, which can differ between
        sources of the same EP (e.g. OpenVINO 1.4.1 vs 1.8.79.0 report
        different addressable memory) and between different EPs on the
        same hardware (OpenVINO vs DML on the iGPU). Empty entries are
        skipped.

        ``compiler_version`` stays as a public property but is deferred
        from default render to ``--verbose`` via :meth:`available_metadata`.
        """
        out: list[str] = []
        if (m := self.memory_bytes) is not None:
            out.append(f"Memory: {_format_bytes(m)}")
        if caps := self.capabilities:
            out.append(f"Capabilities: {', '.join(caps)}")
        return tuple(out)


# T-14: single source of truth for byte formatting lives in
# ``session.monitor.report`` (signature is the strict superset, accepting
# ``int | float | None``). Re-exported here so existing call sites in
# ``WinMLDevice.ep_facts()`` keep working through a private import.
from .monitor.report import _format_bytes  # noqa: E402, F401
