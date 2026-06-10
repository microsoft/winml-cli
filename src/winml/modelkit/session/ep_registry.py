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
from typing import TYPE_CHECKING

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

# Singleton instance
_winml_ep_registry: WinMLEPRegistry | None = None


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

    def __new__(cls) -> WinMLEPRegistry:
        """Singleton pattern."""
        global _winml_ep_registry
        if _winml_ep_registry is None:
            instance = super().__new__(cls)
            instance._initialized = False
            _winml_ep_registry = instance
        return _winml_ep_registry

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

    def entries_for(self, ep_full_name: str) -> list[EPEntry]:
        """Return cached EPEntries for the given EP name (no fresh scan)."""
        return [e for e in self._entries if e.ep_name == ep_full_name]

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
        candidates = self.entries_for(ep_full)

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
    """Collect available EP names from WinML and ORT (cached).

    Hardware and EPs do not change during a process lifetime,
    so this result is cached via lru_cache.

    Returns:
        Frozenset of available EP name strings.
    """
    eps: set[str] = set()

    try:
        registry = WinMLEPRegistry.instance()
        eps.update(e.ep_name for e in registry._entries)
    except (ImportError, RuntimeError):
        pass  # WinML not available
    except Exception:
        logger.warning("Unexpected error during WinML EP discovery", exc_info=True)

    try:
        import onnxruntime as ort

        eps.update(ort.get_available_providers())
    except (ImportError, RuntimeError):
        pass  # ORT not installed
    except Exception:
        logger.warning("Unexpected error during ORT EP discovery", exc_info=True)

    return frozenset(eps)


def get_ort_available_providers(use_winml: bool = True) -> list[str]:
    """Get available execution providers from ONNX Runtime.

    First registers any discovered plugin EPs (if ``use_winml=True``), then
    returns the full list of available providers from ORT.

    Note:
        This function is for informational/debugging purposes only.
        WinMLSession uses policy-based device selection (PREFER_NPU, etc.)
        and does NOT use explicit EP provider names.

    Args:
        use_winml: Try plugin EP discovery first to register providers.

    Returns:
        List of available provider names from ORT.
    """
    import onnxruntime as ort

    if use_winml:
        try:
            registry = WinMLEPRegistry.instance()
            # Best-effort: drive the same inline-loop pattern as commands/sys.py.
            # Per-entry failures are logged at WARN and don't abort the walk.
            for entry in registry._entries:
                try:
                    registry.register_ep(entry)
                except WinMLEPRegistrationFailed as e:
                    logger.warning(
                        "Failed to register EP %s (%s: %s)",
                        entry.ep_name,
                        type(e).__name__,
                        e,
                    )
        except Exception as e:
            # NFR-2: surface real failures at WARNING so users can diagnose.
            logger.warning(
                "Plugin EP discovery skipped (%s: %s)", type(e).__name__, e
            )

    return ort.get_available_providers()


