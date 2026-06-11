# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Execution Provider Registry for plugin-style ONNX Runtime EPs.

Discovers plugin EPs via the unified :mod:`winml.modelkit.ep_path`
discovery layer and registers them with ONNX Runtime via
``register_execution_provider_library()`` (ORT 1.24+).
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import onnxruntime as ort

from ..ep_path import EPEntry, discover_all_eps
from .ep_device import (
    DeviceNotFound,
    EPDeviceTarget,
    UnknownListingPick,
    WinMLDevice,
    WinMLEPNotDiscovered,
    WinMLEPRegistrationFailed,
    expand_ep_name,
    wrap_ort_device,
)


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


def _dedup_ort_devices(devices: list[ort.OrtEpDevice]) -> list[ort.OrtEpDevice]:
    """Collapse OrtEpDevices that share ``(vendor_id, device_id, type)``.

    Some hosts (dual-iGPU listings, OpenVINO on Intel) emit duplicate handles
    for the same physical device.
    """
    seen: set[tuple[int, int, str]] = set()
    out: list[ort.OrtEpDevice] = []
    for d in devices:
        try:
            key = (d.device.vendor_id, d.device.device_id, d.device.type.name)
        except AttributeError:
            out.append(d)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _entry_source_tag(entry: EPEntry) -> str:
    """Derive the canonical source tag for an :class:`EPEntry`.

    Mirrors :func:`commands.sys._describe_source` but lives here so
    :meth:`WinMLEPRegistry.auto_device` can match against ``EPDeviceTarget.source``
    without depending on a CLI module.
    """
    from ..ep_path import (
        DirectorySource,
        MSIXPackageSource,
        NuGetSource,
        PyPISource,
        WinMLCatalogSource,
    )

    s = entry.source
    if isinstance(s, PyPISource):
        return "pypi"
    if isinstance(s, NuGetSource):
        return "nuget"
    if isinstance(s, WinMLCatalogSource):
        return "winml-catalog"
    if isinstance(s, DirectorySource):
        return "directory"
    if isinstance(s, MSIXPackageSource):
        prefix = getattr(s, "family_name_prefix", "")
        if prefix.startswith("WindowsWorkload.EP."):
            return "msix-workload"
        return "msix-microsoft"
    return "unknown"


@dataclass(frozen=True)
class WinMLEP:
    """Per-source registration aggregate. Successful registration only.

    Invariant: len(devices) >= 1. The runtime aggregate produced by
    WinMLEPRegistry.register_ep - it pairs the source EPEntry (which
    DLL was loaded) with the WinMLDevices that ORT exposed after the
    register_execution_provider_library call.

    See docs/design/session/3_design_classes.md section 3.5.
    """

    source: EPEntry
    devices: tuple[WinMLDevice, ...]

    def __post_init__(self) -> None:
        if len(self.devices) == 0:
            raise ValueError(
                "WinMLEP invariant violated: devices tuple must be non-empty "
                f"(source.ep_name={self.source.ep_name!r})"
            )

    def ep_devices(self) -> tuple[WinMLEPDevice, ...]:
        """Flatten self into one WinMLEPDevice pair per device.

        Each returned WinMLEPDevice has .ep == self and .device == self.devices[i].
        """
        return tuple(WinMLEPDevice(ep=self, device=d) for d in self.devices)


@dataclass(frozen=True)
class WinMLEPDevice:
    """Flat (source, device) pair - project mirror of ort.OrtEpDevice.

    Invariant: .device is always one of .ep.devices (same object, not a copy).
    Constructed only by WinMLEPRegistry.auto_device() and by
    WinMLEP.ep_devices(); never by direct user code.

    See docs/design/session/3_design_classes.md section 3.6.
    """

    ep: WinMLEP
    device: WinMLDevice

    def __post_init__(self) -> None:
        if not any(d is self.device for d in self.ep.devices):
            raise ValueError(
                "WinMLEPDevice invariant violated: device must be one of ep.devices "
                f"(ep={self.ep.source.ep_name!r}, "
                f"device.device_type={self.device.device_type!r})"
            )


class WinMLEPRegistry:
    """Execution Provider Registry for plugin-style ONNX Runtime EPs.

    Discovers plugin EPs via :func:`winml.modelkit.ep_path.discover_all_eps`
    (which walks the default EP source list and the ``WINMLCLI_EP_PATH``
    env-var override) once at construction time, caches the result in
    ``self._entries``, and registers entries with ONNX Runtime on demand
    via :meth:`register_ep`.

    Usage:
        registry = WinMLEPRegistry.instance()
        target = resolve_device(EPDeviceTarget(ep="auto", device="auto"))
        ep_device = registry.auto_device(target)
    """

    _instance: ClassVar[WinMLEPRegistry | None] = None

    def __new__(cls) -> WinMLEPRegistry:
        """Singleton pattern. Tests may reset via ``WinMLEPRegistry._instance = None``."""
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Discover plugin EPs from the default EP source list."""
        if self._initialized:
            return
        self._initialized = True

        # Single canonical cache — one filesystem scan per process.
        self._entries: list[EPEntry] = list(discover_all_eps())
        # Cache of successful registrations keyed by EPEntry.dll_path so
        # repeated requests for the same plugin DLL short-circuit.
        self._registered: dict[Path, WinMLEP] = {}
        # ORT's built-in EPs (CPUExecutionProvider, DmlExecutionProvider,
        # bundled Azure, etc.) aren't discovered via EP_PATH — they're
        # baked into ORT itself. Snapshot them here so callers can ask
        # "is X available" through one canonical surface without reaching
        # past the registry to touch ORT directly.
        try:
            self._builtin_eps: frozenset[str] = frozenset(ort.get_available_providers())
        except Exception:
            logger.warning("Unexpected error querying ORT built-in EPs", exc_info=True)
            self._builtin_eps = frozenset()

    def _entries_for(self, ep_full_name: str) -> list[EPEntry]:
        """Return cached EPEntries for the given EP name (no fresh scan).

        Registry-internal. Public callers should not depend on cached
        discovery state; live filesystem walks go through
        :func:`discover_all_eps` directly.
        """
        return [e for e in self._entries if e.ep_name == ep_full_name]

    def builtin_eps(self) -> frozenset[str]:
        """ORT's built-in EPs (CPU, DML, bundled Azure) snapshotted at __init__.

        These EPs are baked into ORT itself — not discovered via EP_PATH.
        The registry is the canonical ORT wrapper; external callers query
        this surface instead of importing onnxruntime directly.
        """
        return self._builtin_eps

    def register_ep(self, entry: EPEntry) -> WinMLEP:
        """Atomic registration: load entry.dll_path, enumerate OrtEpDevices, wrap.

        Idempotent at the ``entry.dll_path`` level — re-registering returns
        the cached :class:`WinMLEP`. The aggregate wraps every
        :class:`WinMLDevice` ORT exposed after the
        ``register_execution_provider_library`` call.

        Raises:
            WinMLEPRegistrationFailed: DLL load failure or ORT registered the
                DLL but yielded zero devices.
        """
        # Idempotency — keyed on the DLL path because two EPEntries pointing
        # at the same DLL must collapse to one registration.
        cached = self._registered.get(entry.dll_path)
        if cached is not None:
            return cached

        # Defensive: another singleton (e.g. winml.py:WinML) may have already
        # called ort.register_execution_provider_library for this EP.  ORT's
        # C++ layer is NOT idempotent — a second registration of the same DLL
        # calls exit(127) with no Python traceback.  Check ORT's live device
        # list before attempting the DLL load.
        already_loaded = any(d.ep_name == entry.ep_name for d in ort.get_ep_devices())
        if not already_loaded:
            try:
                ort.register_execution_provider_library(entry.ep_name, str(entry.dll_path))
            except Exception as exc:
                raise WinMLEPRegistrationFailed(
                    f"ort.register_execution_provider_library({entry.ep_name!r}, "
                    f"{str(entry.dll_path)!r}) failed: {exc}"
                ) from exc
        else:
            logger.debug(
                "EP %s already loaded by another caller; skipping DLL register",
                entry.ep_name,
            )

        # Enumerate devices for this EP and dedup by (vendor_id, device_id).
        all_handles = ort.get_ep_devices()
        matching = [d for d in all_handles if d.ep_name == entry.ep_name]
        deduped = _dedup_ort_devices(matching)

        if not deduped:
            raise WinMLEPRegistrationFailed(
                f"Registered {entry.ep_name!r} from {entry.dll_path} but no "
                f"OrtEpDevices visible in ort.get_ep_devices()."
            )

        devices = tuple(wrap_ort_device(h) for h in deduped)
        winml_ep = WinMLEP(source=entry, devices=devices)
        self._registered[entry.dll_path] = winml_ep

        # Keep _entries consistent with what we just registered, so a later
        # auto_device call for this EP can find this entry via _entries_for.
        # Closes the inconsistency where Path B (--list-ep) used wider extras
        # (e.g. list_msix_eps()) than the registry's default discovery scan
        # saw, causing auto_device to raise WinMLEPNotDiscovered for an EP
        # the registry just registered.
        #
        # EPEntry is a frozen dataclass — structural equality, so `not in`
        # is the natural idempotency check (no path-keyed lookup needed).
        if entry not in self._entries:
            self._entries.append(entry)

        return winml_ep

    def auto_device(self, target: EPDeviceTarget) -> WinMLEPDevice:
        """Find the first source satisfying ``target`` (ep + device + optional source).

        ``target`` must be fully resolved (no ``"auto"`` values). Filters
        the cached :attr:`_entries` list by ``target.ep`` + optional
        ``target.source`` tag, then tries each candidate in precedence
        order. First registration that succeeds *and* exposes
        ``target.device`` wins.

        Raises:
            ValueError: when ``target`` still contains an ``"auto"`` axis.
            WinMLEPNotDiscovered: no candidate EPEntry for the requested ep.
            UnknownListingPick: ``target.source`` is set but doesn't match any
                discovered EPEntry for ``target.ep``.
            WinMLEPRegistrationFailed: every candidate either failed to
                register or exposed no matching device class.
            DeviceNotFound: candidates registered cleanly but none exposed
                ``target.device``.
        """
        if target.ep == "auto" or target.device == "auto":
            raise ValueError(
                "auto_device requires a resolved EPDeviceTarget; "
                "call resolve_device(target) first"
            )

        ep_full = expand_ep_name(target.ep)
        candidates = self._entries_for(ep_full)

        if not candidates:
            raise WinMLEPNotDiscovered(
                f"No EPEntry discovered for ep={target.ep!r}. "
                f"Hint: install the plugin or set WINMLCLI_EP_PATH."
            )

        if target.source is not None:
            tagged = [e for e in candidates if _entry_source_tag(e) == target.source]
            if not tagged:
                raise UnknownListingPick(target.ep, target.source)
            candidates = tagged

        target_device_upper = target.device.upper()
        last_error: Exception | None = None
        for entry in candidates:
            try:
                winml_ep = self.register_ep(entry)
            except WinMLEPRegistrationFailed as e:
                last_error = e
                continue
            for device in winml_ep.devices:
                if device.device_type == target_device_upper:
                    return WinMLEPDevice(ep=winml_ep, device=device)

        # All candidates exhausted without a match.
        if last_error is not None:
            raise WinMLEPRegistrationFailed(
                f"No compatible source for {target.ep}/{target.device}; "
                f"all {len(candidates)} candidates failed"
            ) from last_error
        raise DeviceNotFound(
            f"No source for {target.ep}/{target.device} exposed device "
            f"class {target.device.upper()!r}"
        )

    @classmethod
    def instance(cls) -> WinMLEPRegistry:
        """Get singleton instance."""
        return cls()


@functools.lru_cache(maxsize=1)
def available_eps() -> frozenset[str]:
    """Collect available EP names from the WinMLEPRegistry (cached).

    Includes both plugin EPs discovered via :func:`discover_all_eps` and
    ORT's built-in EPs (CPU, DML, bundled Azure) snapshotted at registry
    init. The registry is the canonical wrapper over ONNX Runtime — callers
    should not query ORT directly for available EPs.

    Hardware and EPs do not change during a process lifetime,
    so this result is cached via lru_cache.

    Returns:
        Frozenset of available EP name strings.
    """
    try:
        registry = WinMLEPRegistry.instance()
        plugin = frozenset(e.ep_name for e in registry._entries)
        return plugin | registry.builtin_eps()
    except (ImportError, RuntimeError):
        return frozenset()  # WinML not available
    except Exception:
        logger.warning("Unexpected error during WinML EP discovery", exc_info=True)
        return frozenset()


