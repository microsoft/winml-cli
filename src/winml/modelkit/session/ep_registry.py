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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import onnxruntime as ort

from ..ep_path import BuiltinSource, EPEntry, discover_all_eps
from .ep_device import (
    DeviceNotFound,
    EPDeviceTarget,
    UnknownListingPick,
    WinMLDevice,
    WinMLEPNotDiscovered,
    WinMLEPRegistrationFailed,
    expand_ep_name,
)


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


def _ort_get_ep_devices_or_fail(entry: EPEntry) -> list[ort.OrtEpDevice]:
    """Call :func:`ort.get_ep_devices` inside the WinMLEPRegistrationFailed contract.

    Both branches of :meth:`WinMLEPRegistry.register_ep` (BuiltinSource
    wrap + plugin post-registration filter) call ``ort.get_ep_devices()``;
    its failure should surface as :class:`WinMLEPRegistrationFailed` so
    ``auto_device``'s ``except WinMLEPRegistrationFailed`` retry loop
    can fall through to the next candidate instead of crashing the CLI
    with a raw ORT traceback (driver reset, native init failure, etc.).
    """
    try:
        return ort.get_ep_devices()
    except Exception as exc:
        raise WinMLEPRegistrationFailed(
            f"ort.get_ep_devices() failed while resolving {entry.ep_name!r}: {exc}"
        ) from exc


def _entry_source_tag(entry: EPEntry) -> str:
    """Derive the canonical source tag for an :class:`EPEntry`.

    Mirrors :func:`commands.sys._describe_source` but lives here so
    :meth:`WinMLEPRegistry.auto_device` can match against ``EPDeviceTarget.source``
    without depending on a CLI module.
    """
    # ``BuiltinSource`` is already imported at module top level (used by
    # __init__'s synthesis loop); the rest stay lazy to keep import cost
    # off the registry-construction path.
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
    if isinstance(s, BuiltinSource):
        return "bundled"
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
    ``self._discovered``, and registers entries with ONNX Runtime on demand
    via :meth:`register_ep`.

    Usage:
        registry = WinMLEPRegistry.instance()
        target = resolve_device(EPDeviceTarget(ep="auto", device="auto"))
        ep_device = registry.auto_device(target)
    """

    _instance: ClassVar[WinMLEPRegistry | None] = None

    def __init__(self) -> None:
        """Discover plugin EPs from the default EP source list.

        Construction is unguarded — call :meth:`instance` to get the
        process-wide singleton; that classmethod is the only thing that
        caches the result. Tests reset via
        ``WinMLEPRegistry._instance = None`` and then re-invoke
        :meth:`instance` to rebuild.
        """
        # Output cache: dll_path -> WinMLEP returned after a successful
        # register_ep call. Reject double-registration (ORT errors otherwise).
        # NOTE: presence here means "DLL loaded" only — L2 vendor
        # compatibility and device-class availability are evaluated by
        # callers (sys.py renderer / auto_device) after the fact.
        self._registered: dict[Path, WinMLEP] = {}
        # How many times each canonical ep_name has been registered with
        # ORT so far. First registration uses the canonical name; later
        # ones get a ``_<n>`` suffix on the ORT-side arg0 — the device's
        # self-reported ``ep_name`` stays canonical (empirically verified
        # via temp/probe_double_register.py), so this only affects ORT's
        # internal registration-tracking key, never the device routing.
        self._registration_count: dict[str, int] = {}
        # ORT's built-in EPs (CPU/Dml/bundled Azure, plus whatever ORT
        # bundles in future versions) aren't discovered on disk — they're
        # baked into ORT itself. Cross-check get_available_providers() with
        # get_ep_devices() so a misconfigured ORT (provider name listed but
        # zero matching OrtEpDevices) doesn't get synthesized — that would
        # leak through auto_detect_device and crash at session-build time
        # with a confusing "Built-in EP exposed no devices" error (F-07).
        try:
            provider_names = frozenset(ort.get_available_providers())
            ep_devices = ort.get_ep_devices()
        except Exception:
            logger.warning("Unexpected error querying ORT built-in EPs", exc_info=True)
            provider_names = frozenset()
            ep_devices = []
        builtin_names = provider_names & {d.ep_name for d in ep_devices}

        # Unified-source synthesis: filesystem discovery + a synthetic
        # EPEntry for every built-in EP name that filesystem discovery
        # didn't already cover AND that ORT actually exposes via
        # get_ep_devices(). Built-in entries flow through the same
        # precedence loop as plugin entries; register_ep dispatches on
        # isinstance(entry.source, BuiltinSource) to skip the DLL-load
        # step and wrap pre-registered ORT handles directly. Synthesis
        # runs AFTER discover_all_eps, so plugin entries with the same
        # ep_name keep their natural precedence (built-ins are lowest
        # priority — only used when no plugin provided the EP). Single
        # immutable assignment.
        plugin_entries = list(discover_all_eps())
        discovered_names = {e.ep_name for e in plugin_entries}
        self._discovered: list[EPEntry] = plugin_entries + [
            EPEntry(
                ep_name=builtin_name,
                dll_path=Path(),
                source=BuiltinSource(eps=(builtin_name,)),
            )
            for builtin_name in sorted(builtin_names - discovered_names)
        ]
        # F-08: cache for built-in WinMLEPs keyed by ep_name so register_ep
        # is object-identity idempotent (the _registered dict can't be
        # reused because BuiltinSource entries all share Path("")).
        self._builtin_registered: dict[str, WinMLEP] = {}
        # F-17: memoize available_eps()'s derived frozenset. _discovered
        # is frozen post-init, so the result never changes; rebuilding
        # on every call wastes work in hot paths (auto_device, --list-ep).
        self._available_eps_cache: frozenset[str] | None = None

    def _entries_for(self, ep_full_name: str) -> list[EPEntry]:
        """Return cached EPEntries for the given EP name (no fresh scan).

        Registry-internal. Public callers should not depend on cached
        discovery state; live filesystem walks go through
        :func:`discover_all_eps` directly.
        """
        return [e for e in self._discovered if e.ep_name == ep_full_name]

    def register_ep(self, entry: EPEntry) -> WinMLEP:
        """Return the WinMLEP for ``entry.dll_path``, loading the DLL if needed.

        Idempotent on ``entry.dll_path``: the first call loads the DLL via
        ``ort.register_execution_provider_library``, enumerates matching
        OrtEpDevices, wraps them in a :class:`WinMLEP`, and caches the
        result in ``self._registered``. Every subsequent call with the
        same ``entry.dll_path`` short-circuits to the cached :class:`WinMLEP`
        (identity-equal) WITHOUT re-invoking ORT.

        This lets the precedence-loop callers (``auto_device``,
        ``commands.sys._gather_ep_info``) walk over discovered entries
        across multiple invocations in the same process without the
        second call falsely raising ``WinMLEPRegistrationFailed`` for an
        EP that was already loaded by an earlier call.

        Multiple DLLs reporting the same canonical ``ep_name`` (e.g. a
        PyPI OpenVINO + an MSIX OpenVINO + a Catalog OpenVINO) all load
        independently — idempotency is keyed on ``dll_path``, not
        ``ep_name``. ORT's first registration uses the canonical
        ``entry.ep_name`` as its registration-tracking key; subsequent
        registrations for the same ``ep_name`` (different DLL) are
        suffixed ``_<n>`` so ORT's internal arg0 stays unique. The
        OrtEpDevice handles still self-report the canonical ``ep_name``
        (verified via temp/probe_double_register.py), so neither
        :meth:`add_provider_for_devices` nor session compilation are
        affected by the suffix.

        Raises:
            WinMLEPRegistrationFailed: DLL load failed, or ORT exposed
                zero matching devices.
        """
        # Built-in EPs (BuiltinSource entries) are already loaded by ORT
        # at process init — no DLL to register, no library_path to filter
        # by. Wrap the matching ORT handles directly and return. Built-ins
        # can't share ``self._registered`` (all BuiltinSource entries have
        # Path("") as dll_path so they'd collide), so they live in their
        # own ``_builtin_registered`` cache keyed by ep_name to preserve
        # object identity across repeat calls (F-08).
        if isinstance(entry.source, BuiltinSource):
            cached = self._builtin_registered.get(entry.ep_name)
            if cached is not None:
                return cached
            all_handles = _ort_get_ep_devices_or_fail(entry)
            matching = [d for d in all_handles if d.ep_name == entry.ep_name]
            deduped = _dedup_ort_devices(matching)
            if not deduped:
                raise WinMLEPRegistrationFailed(
                    f"Built-in EP {entry.ep_name!r} exposed no devices via "
                    f"ort.get_ep_devices()."
                )
            devices = tuple(WinMLDevice(h) for h in deduped)
            winml_ep = WinMLEP(source=entry, devices=devices)
            self._builtin_registered[entry.ep_name] = winml_ep
            return winml_ep

        # Idempotency: cache hit means this DLL was already loaded by an
        # earlier call. Return the cached WinMLEP without re-registering
        # with ORT (which would fail with "library already registered").
        if entry.dll_path in self._registered:
            return self._registered[entry.dll_path]

        # First registration of this ep_name uses the canonical name;
        # subsequent get a ``_<n>`` suffix to keep ORT's arg0 unique.
        n = self._registration_count.get(entry.ep_name, 0)
        arg0 = entry.ep_name if n == 0 else f"{entry.ep_name}_{n}"

        try:
            ort.register_execution_provider_library(arg0, str(entry.dll_path))
        except Exception as exc:
            raise WinMLEPRegistrationFailed(
                f"ort.register_execution_provider_library({arg0!r}, "
                f"{str(entry.dll_path)!r}) failed: {exc}"
            ) from exc
        self._registration_count[entry.ep_name] = n + 1

        # Filter ORT's device list by THIS DLL's library_path — the
        # device's self-reported ep_name is canonical (not suffixed), so
        # filtering on ep_name would collapse multiple registrations of
        # the same ep_name into one set.
        all_handles = _ort_get_ep_devices_or_fail(entry)
        matching = [
            d for d in all_handles
            if d.ep_metadata.get("library_path") == str(entry.dll_path)
        ]
        deduped = _dedup_ort_devices(matching)

        if not deduped:
            raise WinMLEPRegistrationFailed(
                f"Registered {arg0!r} from {entry.dll_path} but no "
                f"OrtEpDevices visible in ort.get_ep_devices()."
            )

        devices = tuple(WinMLDevice(h) for h in deduped)
        winml_ep = WinMLEP(source=entry, devices=devices)
        self._registered[entry.dll_path] = winml_ep
        return winml_ep

    def auto_device(self, target: EPDeviceTarget) -> WinMLEPDevice:
        """Find the first source satisfying ``target`` (ep + device + optional source).

        ``target`` must be fully resolved (no ``"auto"`` values). Filters
        the cached :attr:`_discovered` list by ``target.ep`` + optional
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
            # Registration succeeded but no device-class match — this
            # candidate is NOT a registration failure, so don't let a
            # prior candidate's stale traceback survive into the
            # post-loop `last_error is not None` branch (T-04).
            last_error = None

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

    def all_discovered(self) -> tuple[EPEntry, ...]:
        """Snapshot of the discovery cache, in walk order, NO filtering.

        Returns every :class:`EPEntry` produced by ``discover_all_eps`` at
        registry-init time. No L1 (registration) or L2 (vendor-compatibility)
        filtering happens here — the inventory renderer in
        ``commands/sys.py:_gather_ep_info`` is the canonical consumer and
        owns per-row status derivation (§7.1).

        Discovery is cached at __init__; this method just exposes the cache.
        """
        return tuple(self._discovered)

    def available_eps(self) -> frozenset[str]:
        """EP names discovered on this host (L0 only): plugins + ORT built-ins.

        Returns the L0 discovery set — every EP with an EPEntry in
        :attr:`_discovered`. Built-in EPs (CPU/Dml/Azure) are synthesized
        into ``_discovered`` at ``__init__`` via :class:`BuiltinSource`,
        so they appear alongside filesystem-discovered plugin EPs in a
        single set.

        Three filters are deliberately NOT applied here:

        - **L1 (registration success)** — too expensive to gate a frequently
          called helper on DLL loads. :meth:`auto_device`'s retry loop owns
          L1 failure handling at session-build time.
        - **L2 (vendor-compatibility)** — kept out per
          ``docs/design/session/2_coreloop.md`` §7.1.1-§7.1.2: the
          ``--list-ep`` renderer needs an L0 set so it can distinguish a
          ``[failed]`` row (L1) from an ``[incompatible]`` row (L2). Baking
          L2 here would collapse those layers. Deduction helpers that need
          L2 (:func:`default_ep_for_device`, :func:`auto_detect_device`)
          compose ``EPCatalog.is_compatible`` on top — see
          ``docs/design/session/3_design_ep.md`` §6.4 for the composition
          pattern.
        - **L3 (validation)** — not implemented (per §3.3 PROPOSED).

        Returns:
            Frozenset of EP name strings; empty on import / RuntimeError
            (WinML unavailable) or any unexpected error (logged WARN).

        Memoized: ``self._discovered`` is frozen post-init, so the
        derived set is built once and cached on the instance for every
        subsequent call (read on every CLI command, twice per session
        startup, plus in tight loops inside ``auto_device``).
        """
        if self._available_eps_cache is not None:
            return self._available_eps_cache
        try:
            result = frozenset(e.ep_name for e in self._discovered)
        except (ImportError, RuntimeError):
            result = frozenset()  # WinML / sysinfo not available
        except Exception:
            logger.warning("Unexpected error during WinML EP discovery", exc_info=True)
            result = frozenset()
        self._available_eps_cache = result
        return result

    @classmethod
    def instance(cls) -> WinMLEPRegistry:
        """Return the process-wide singleton (built on first call).

        Single owner of the singleton invariant: all production call
        sites enter through here; direct ``WinMLEPRegistry()`` calls
        bypass the cache and build a fresh instance (used by tests).
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


