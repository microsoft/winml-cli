# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unified execution-provider discovery.

This module replaces the legacy ``EP_PLUGIN_REGISTRY`` dict (which only
modeled PyPI-installed plugin EPs) with an ordered list of typed
``EPSource`` entries, analogous to the OS ``PATH`` environment variable.
Each entry knows how to resolve itself for the current machine and yields
:class:`EPEntry` records (``ep_name``, ``dll_path``, ``source``,
``status``, ``version``).

See ``docs/ep-path-design.md`` for the full design rationale, including
the per-origin x per-EP map and the migration plan.

Public API:

* :data:`EP_CATALOG`: canonical EP metadata registry (:class:`EPCatalog`).
* :class:`EPCatalog.Row`: frozen dataclass for one EP's metadata (name, DLL, vendor).
* :class:`EPCatalog`: registry with forward/inverse lookups and vendor compat.
* :class:`PyPISource`: pip-installed plugin EP wheels.
* :class:`NuGetSource`: NuGet-cached plugin EP packages
  (``~/.nuget/packages/<id>/<version>/runtimes/<rid>/native/...``).
* :class:`DirectorySource`: directory drops (installer, unzipped archive,
  custom build).
* :class:`WinMLCatalogSource`: WinAppSDK ``ExecutionProviderCatalog``
  MSIX-delivered EPs. Lazily imports the WinAppSDK ML Python binding;
  yields nothing silently when the binding is not installed.
* :class:`MSIXPackageSource`: WinRT ``PackageManager`` MSIX EP discovery
  by family-name prefix (handles non-current versions and the
  ``WindowsWorkload.EP.*`` OEM channel).
* :class:`EPSource`: abstract base for the five concrete sources.
* :func:`discover_all_eps`: walk the default EP source list (plus any
  extras) and return a flat ``list[EPEntry]`` containing the primary
  winner per EP followed by any shadowed entries. Single canonical
  discovery entry point — used by ``WinMLEPRegistry``, the inventory CLI
  (``winml sys --list-ep``), and the legacy ``winml.py`` shim.
"""

from __future__ import annotations

import atexit
import dataclasses
import functools
import logging
import os
import platform
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

from packaging.version import InvalidVersion, Version


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical EP metadata registry.
# ---------------------------------------------------------------------------


class EPCatalog:
    """Canonical EP metadata registry: forward + inverse lookups + vendor compat.

    Replaces the three legacy module-level dicts (``EP_DLL_NAMES``,
    ``_DLL_TO_EP_NAME``, ``_EP_VENDOR_REQUIREMENT``). Constructed once at
    module load; ``EP_CATALOG`` is the project-wide instance.

    Immutable after construction: the internal lookup dicts are wrapped
    in ``MappingProxyType`` (no in-place mutation) and ``__setattr__`` is
    locked after ``__init__`` (no attribute rebinding). Tests swap by
    constructing a fresh ``EPCatalog`` and patching the module-level
    ``EP_CATALOG`` binding.
    """

    @dataclass(frozen=True)
    class Row:
        """One canonical EP's metadata: name, DLL filename, vendor requirement.

        The ``dll_name`` is empty for bundled EPs (CPU, DML, Azure) — they
        ship with ORT itself and never need plugin DLL loading.
        """

        name: str
        dll_name: str
        vendor_requirements: frozenset[str]

    __slots__ = ("_by_dll", "_by_name", "_initialized")

    # Annotation-only declarations (attributes are set via object.__setattr__
    # in __init__ because the instance is immutable). No assigned value, so
    # these do not conflict with __slots__.
    _by_name: MappingProxyType[str, EPCatalog.Row]
    _by_dll: MappingProxyType[str, str]
    _initialized: bool

    def __init__(self, entries: Iterable[EPCatalog.Row]) -> None:
        by_name: dict[str, EPCatalog.Row] = {e.name: e for e in entries}
        by_dll: dict[str, str] = {e.dll_name: e.name for e in by_name.values() if e.dll_name}
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))
        object.__setattr__(self, "_by_dll", MappingProxyType(by_dll))
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError(f"EPCatalog is immutable; cannot set {name!r}")
        object.__setattr__(self, name, value)

    def dll_name_for(self, ep: str) -> str | None:
        """Return the DLL filename for ``ep``, or ``None`` for bundled / unknown EPs."""
        entry = self._by_name.get(ep)
        return entry.dll_name if entry and entry.dll_name else None

    def ep_for_dll(self, dll: str) -> str | None:
        """Reverse lookup: DLL filename -> canonical EP name. ``None`` if unknown."""
        return self._by_dll.get(dll)

    def is_compatible(self, ep: str) -> bool:
        """Return True iff ``ep`` has compatible hardware on this machine.

        Empty / missing vendor requirement -> always compatible.
        Otherwise compatible iff at least one required vendor substring
        appears (case-insensitively) in any detected vendor string.
        """
        entry = self._by_name.get(ep)
        if entry is None or not entry.vendor_requirements:
            return True
        detected = _get_detected_vendors()
        return any(req.lower() in v.lower() for req in entry.vendor_requirements for v in detected)

    def all_eps(self) -> tuple[str, ...]:
        """Return all canonical EP names in catalog order."""
        return tuple(self._by_name)


EP_CATALOG = EPCatalog(
    [
        EPCatalog.Row(
            name="OpenVINOExecutionProvider",
            dll_name="onnxruntime_providers_openvino_plugin.dll",
            vendor_requirements=frozenset({"Intel"}),
        ),
        EPCatalog.Row(
            name="QNNExecutionProvider",
            dll_name="onnxruntime_providers_qnn.dll",
            vendor_requirements=frozenset({"Qualcomm"}),
        ),
        EPCatalog.Row(
            name="VitisAIExecutionProvider",
            dll_name="onnxruntime_providers_vitisai.dll",
            vendor_requirements=frozenset({"AMD"}),
        ),
        # TODO(ep_path): MIGraphX DLL leaf is unverified; mirrors the VitisAI
        # naming convention. Confirm by inspecting an installed MSIX. See
        # docs/ep-path-design.md TODO #4.
        EPCatalog.Row(
            name="MIGraphXExecutionProvider",
            dll_name="onnxruntime_providers_migraphx.dll",
            vendor_requirements=frozenset({"AMD"}),
        ),
        EPCatalog.Row(
            name="NvTensorRTRTXExecutionProvider",
            dll_name="onnxruntime_providers_nv_tensorrt_rtx.dll",
            vendor_requirements=frozenset({"NVIDIA"}),
        ),
        EPCatalog.Row(name="DmlExecutionProvider", dll_name="", vendor_requirements=frozenset()),
        EPCatalog.Row(name="CPUExecutionProvider", dll_name="", vendor_requirements=frozenset()),
        EPCatalog.Row(name="AzureExecutionProvider", dll_name="", vendor_requirements=frozenset()),
    ]
)


@functools.cache
def _get_detected_vendors() -> frozenset[str]:
    """Return the union of vendor identification strings from sysinfo.

    Aggregates ``manufacturer`` and ``name`` across detected GPUs and
    NPUs. Both fields are included because Windows reports vendor
    inconsistently — sometimes the manufacturer is the IHV
    (``"Qualcomm Incorporated"``), sometimes a parent company
    (``"Microsoft Corporation"`` for OEM-rebranded devices).

    Cached process-wide; tests reset via ``_get_detected_vendors.cache_clear()``.
    Raises ``RuntimeError`` if hardware detection fails — preventing
    ``functools.cache`` from pinning an empty-set fallback that would
    silently make every hardware-gated EP appear incompatible.
    """
    try:
        from .sysinfo.hardware import GPU, NPU
    except ImportError as e:
        raise RuntimeError(f"Hardware detection unavailable: {e}") from e

    strings: set[str] = set()
    for cls in (GPU, NPU):
        try:
            for hw in cls.get_all():
                for attr in ("manufacturer", "name"):
                    value = getattr(hw, attr, None)
                    if value:
                        strings.add(str(value))
        except Exception as e:  # noqa: PERF203
            raise RuntimeError(f"{cls.__name__}.get_all() failed: {e}") from e

    return frozenset(strings)


# ---------------------------------------------------------------------------
# Architecture resolver helpers.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _resolve_arch_key() -> str:
    """One of the 3 physically-reachable (process arch, host arch) combinations.

    The 4th combination — a native-ARM64 process on genuine x64 hardware —
    isn't reachable: there is no Windows emulation path that runs an
    ARM64-only executable on x64 silicon, so it's intentionally omitted.

    Returns:
        ``"x64_native"``: x64 process, x64 host (no emulation involved).
        ``"x64_on_arm64"``: x64 process, ARM64 host (Windows-on-ARM x64
            emulation — the scenario this module's arch bug lived in).
        ``"arm64_native"``: native ARM64 process, ARM64 host.

    Host architecture (the piece that actually matters for this bug) comes
    from :class:`~winml.modelkit.sysinfo.CPU`'s ``architecture`` (WMI's
    ``Win32_Processor.Architecture``): it queries the physical processor
    inventory directly, not the calling process, so it's correct even
    under x64-on-ARM64 emulation — unlike ``platform.machine()``, which
    reports "AMD64" for an x64 Python venv on Windows-on-ARM64 even though
    the machine is ARM64. That's a documented ``onnxruntime-qnn`` footgun
    ("WoS AMD64 — Python 3.11 installer issue" in the v2.3.0 release
    notes): picking the amd64 (x64-emulated) QNN DLL over the native
    arm64ec one causes the QNN HTP EP to hang indefinitely on some SDK
    versions and run ~1000x slower on others, even though session creation
    succeeds either way.

    Process-own architecture uses ``platform.machine()`` directly — unlike
    host detection, it's correct for this: it only misreports the *host*
    under emulation, not itself.

    Cached (``lru_cache(maxsize=1)``): the host CPU cannot change within a
    process's lifetime, and this is called on every ``PyPISource.resolve()``
    with an ``arch_folder_map``; ``CPU.get_all()`` shells out to PowerShell
    per call, so repeating it per resolve would be wasteful. Tests that
    monkeypatch ``CPU.get_all``/``platform.machine`` to simulate different
    hosts must call ``_resolve_arch_key.cache_clear()`` first, or they'll
    observe a stale result from an earlier test.
    """
    host_is_arm64 = False
    if os.name == "nt":
        try:
            from .sysinfo import CPU

            host_is_arm64 = any(cpu.architecture == CPU.Architecture.ARM64 for cpu in CPU.get_all())
        except Exception:
            host_is_arm64 = False

    if not host_is_arm64:
        return "x64_native"
    is_process_arm64 = platform.machine().lower() in ("arm64", "aarch64")
    return "arm64_native" if is_process_arm64 else "x64_on_arm64"


# ---------------------------------------------------------------------------
# NuGet packages root helper (used by NuGetSource).
# ---------------------------------------------------------------------------


def _nuget_packages_root() -> Path:
    """Return the NuGet global-packages cache root for this user.

    Honors ``USERPROFILE`` on Windows (matching the .NET SDK default) and
    ``HOME`` on POSIX. Does not consult the ``NUGET_PACKAGES`` env var or
    the user's ``NuGet.Config`` ``globalPackagesFolder`` override — those
    are advanced .NET CLI tunings rare for ML users; revisit if it becomes
    a need. Returns the path even if it does not exist; the caller is
    responsible for the existence check (a missing cache is "no packages
    installed", not an error).
    """
    if os.name == "nt":
        userprofile = os.environ.get("USERPROFILE")
        base = Path(userprofile) if userprofile else Path.home()
    else:
        base = Path.home()
    return base / ".nuget" / "packages"


# ---------------------------------------------------------------------------
# EPEntry: the canonical (ep_name, dll_path, source, status, version) record.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EPEntry:
    """One ``(ep_name, dll_path, source, status, version)`` hit.

    Yielded by every :meth:`EPSource.resolve` and returned (flattened)
    by :func:`discover_all_eps`. The ``status`` field distinguishes the
    precedence-winner ("primary") from later sources for the same EP
    that were skipped under first-hit-wins ("shadowed"). Sources always
    yield ``status="primary"``; :func:`discover_all_eps` overrides to
    ``"shadowed"`` for non-precedence-winners via ``dataclasses.replace``.

    The optional ``version`` field carries per-subclass version metadata
    (PyPI distribution version, NuGet package version, MSIX
    ``Package.Id.Version``). ``None`` for sources with no version concept
    (DirectorySource, WinMLCatalogSource).
    """

    ep_name: str
    dll_path: Path
    source: EPSource
    status: str = "primary"  # "primary" | "shadowed"
    version: str | None = None

    def is_built_in(self) -> bool:
        """True iff this entry was synthesized for a :class:`BuiltinSource`.

        Built-in EPs (CPU, DML, Azure) are pre-loaded by ORT itself — no
        DLL to discover, register, or stat. Their :attr:`dll_path` is the
        sentinel ``Path("")``; callers that ``Path.is_file()``-check
        (:func:`discover_all_eps`) or spawn a subprocess to load the DLL
        (``winml sys --list-ep`` isolation) must branch on this.
        """
        return isinstance(self.source, BuiltinSource)


# ---------------------------------------------------------------------------
# EPSource ABC + concrete dataclass implementations.
# ---------------------------------------------------------------------------


class EPSource(ABC):
    """Abstract base for any source that can yield :class:`EPEntry` records.

    Five concrete subclasses cover the origins documented in
    ``docs/ep-path-design.md``: :class:`PyPISource`, :class:`NuGetSource`,
    :class:`DirectorySource`, :class:`WinMLCatalogSource`, and
    :class:`MSIXPackageSource`. Subclasses are frozen dataclasses; this
    base provides the shared :meth:`is_compatible` body and documents
    the :meth:`resolve` / :meth:`iter_eps` contract.
    """

    @abstractmethod
    def resolve(self) -> Iterator[EPEntry]:
        """Yield :class:`EPEntry` records zero or more times.

        Each yielded entry should have ``status="primary"`` (the default);
        :func:`discover_all_eps` overrides to ``"shadowed"`` for entries
        that lose precedence against an earlier source.

        Errors during resolution should be logged and swallowed (yield
        nothing) — :func:`discover_all_eps` tolerates source-level failures
        but cannot recover from a raised exception.
        """

    @abstractmethod
    def iter_eps(self) -> Iterable[str]:
        """Return the canonical EP names this source declares to provide.

        Used by :meth:`is_compatible` and by the CLI inventory layer.
        Should match the ``ep_name`` values that :meth:`resolve` would
        yield, but is statically declarable (no I/O required).
        """

    def is_compatible(self) -> bool:
        """True iff every EP this source provides has compatible hardware."""
        return all(EP_CATALOG.is_compatible(ep) for ep in self.iter_eps())


@dataclass(frozen=True)
class BuiltinSource(EPSource):
    """Marker for EPs bundled into ORT itself (CPU/Dml/Azure today).

    NOT walked by :func:`discover_all_eps` — ``resolve()`` returns nothing.
    Instances are synthesized directly into ``WinMLEPRegistry._discovered``
    at registry ``__init__`` for every name in ``ort.get_available_providers()``
    that isn't already present from filesystem discovery. The corresponding
    EPEntry carries a sentinel ``dll_path=Path("")`` (no on-disk DLL).

    ``WinMLEPRegistry.register_ep`` dispatches on ``isinstance(entry.source,
    BuiltinSource)`` to wrap the already-loaded handles via
    ``ort.get_ep_devices()`` rather than calling
    ``ort.register_execution_provider_library`` — built-ins are
    pre-registered at ORT init.
    """

    eps: tuple[str, ...] = ()

    def resolve(self) -> Iterator[EPEntry]:
        """No-op — built-in entries are synthesized at registry init."""
        return iter(())

    def iter_eps(self) -> Iterable[str]:
        """Declared EP names (used by ``is_compatible`` and CLI inventory)."""
        return self.eps


@dataclass(frozen=True)
class PyPISource(EPSource):
    """A pip-installed plugin EP wheel.

    The DLL path is computed lazily via
    ``importlib.metadata.distribution(name).locate_file(rel)`` so it
    follows whichever venv is currently active.

    Args:
        distribution: PyPI distribution name, e.g. ``"onnxruntime-ep-openvino"``.
        relative_dll: Path inside the wheel, POSIX-style. May contain an
            ``{arch}`` placeholder that ``arch_folder_map`` substitutes.
        eps: Canonical EP names this source provides (typically a single name).
        arch_folder_map: Optional ``Mapping[str, str]`` keyed by
            :func:`_resolve_arch_key` (``"x64_native"``, ``"x64_on_arm64"``,
            ``"arm64_native"``) whose values fill the ``{arch}`` placeholder
            in ``relative_dll``. ``None`` means ``relative_dll`` is used
            as-is (the common case for wheels with only one DLL layout —
            e.g. OpenVINO's PyPI package, which ships no ``{arch}`` split).
            A key missing from the map for the current combination means
            "no known build for this machine" — logged at DEBUG, no entry
            yielded, not an error (mirrors "distribution not installed").
    """

    distribution: str
    relative_dll: str
    eps: tuple[str, ...]
    arch_folder_map: Mapping[str, str] | None = None

    def resolve(self) -> Iterator[EPEntry]:
        """Yield one :class:`EPEntry` per EP this source provides.

        Yields nothing (silently) when the distribution is not installed —
        that is the common case for optional EPs and is not an error.
        Logs a warning if the distribution is installed but the file is
        missing. The yielded entries carry ``version`` populated from
        ``importlib.metadata.version(self.distribution)``; if metadata
        lookup raises (rare), ``version=None`` and the failure is logged
        at DEBUG.
        """
        try:
            dist = metadata.distribution(self.distribution)
        except metadata.PackageNotFoundError:
            logger.debug("PyPISource: distribution %r not installed; skipping", self.distribution)
            return

        rel = self.relative_dll
        if self.arch_folder_map is not None:
            arch_key = _resolve_arch_key()
            arch = self.arch_folder_map.get(arch_key)
            if arch is None:
                logger.debug(
                    "PyPISource: distribution %r has no known DLL layout for "
                    "this machine (arch_key=%r); skipping",
                    self.distribution,
                    arch_key,
                )
                return
            rel = rel.format(arch=arch)

        path = Path(str(dist.locate_file(rel)))
        if not path.exists():
            logger.warning(
                "PyPISource: distribution %r installed but DLL missing at %s",
                self.distribution,
                path,
            )
            return

        try:
            version: str | None = metadata.version(self.distribution)
        except Exception as e:
            logger.debug(
                "PyPISource: metadata.version(%r) failed: %s",
                self.distribution,
                e,
            )
            version = None

        for ep_name in self.eps:
            yield EPEntry(
                ep_name=ep_name,
                dll_path=path,
                source=self,
                version=version,
            )

    def iter_eps(self) -> Iterable[str]:
        """Return the canonical EP names this source provides."""
        return self.eps


@dataclass(frozen=True)
class NuGetSource(EPSource):
    """A NuGet-cached EP plugin package (``~/.nuget/packages/<id>/<ver>/...``).

    Mirrors :class:`PyPISource` but resolves against the global NuGet
    packages cache instead of the active Python environment. Useful for
    .NET-installed EP plugins (``Intel.ML.OnnxRuntime.EP.OpenVINO``,
    ``Qualcomm.ML.OnnxRuntime.QNN``) on machines where the user has the
    .NET SDK plus a project that has restored the package — the plugin
    DLL ships under
    ``runtimes/<rid>/native/`` and ORT can load it directly.

    Selection rules (in order):

    1. Resolve the cache root (``~/.nuget/packages``).
    2. Look up ``cache_root / self.distribution.lower()``. If the directory
       does not exist, yield nothing silently — the package is not in the
       local cache and that is the common case.
    3. Enumerate version subdirectories. Skip directories whose names do
       not parse as a NuGet version.
    4. Sort: highest stable wins; fall back to highest prerelease only if
       no stable is installed.
    5. For the chosen version, probe ``<version>/<relative_dll>`` (after
       ``arch_resolver`` substitution). Yield ``(ep_name, abs_path)`` for
       each ``ep`` in :attr:`eps` if the DLL exists.

    Args:
        distribution: Canonical NuGet package ID (e.g.
            ``"Intel.ML.OnnxRuntime.EP.OpenVINO"``). Case-insensitive on
            disk; the cache lower-cases the directory name.
        relative_dll: Path inside the package, POSIX-style, relative to
            the version directory. May contain ``{...}`` placeholders that
            ``arch_resolver`` substitutes (e.g., ``{rid}``).
        eps: Canonical EP names this source provides. Tuple-shaped for
            symmetry with :class:`PyPISource`; almost always a single name
            since one NuGet package maps to one EP.
        arch_resolver: Optional ``Callable[[str], str]`` that takes the
            ``relative_dll`` template and returns a substituted relative
            path tweaked per machine architecture. ``None`` means the
            ``relative_dll`` is used as-is.
    """

    distribution: str
    relative_dll: str
    eps: tuple[str, ...]
    arch_resolver: Callable[[str], str] | None = None

    def resolve(self) -> Iterator[EPEntry]:
        """Yield one :class:`EPEntry` per EP this source provides.

        The yielded entries carry ``version`` derived from the NuGet cache
        subdir name (the version folder selected as the highest stable —
        or highest prerelease if no stable is installed).
        """
        if "\\" in self.relative_dll:
            raise ValueError(
                f"NuGetSource.relative_dll must be POSIX-style "
                f"(forward-slash separators); got {self.relative_dll!r}"
            )

        cache_root = _nuget_packages_root()
        # NuGet stores under the lowercased package ID (per the v3 cache
        # spec). On case-sensitive filesystems this is load-bearing.
        pkg_dir = cache_root / self.distribution.lower()
        if not pkg_dir.is_dir():
            logger.debug(
                "NuGetSource: package %r not in cache at %s; skipping",
                self.distribution,
                pkg_dir,
            )
            return

        # Enumerate version subdirs and parse each as a SemVer version.
        candidates: list[tuple[Version, Path]] = []
        try:
            entries = list(pkg_dir.iterdir())
        except OSError as e:
            logger.warning("NuGetSource: failed to enumerate %s: %s", pkg_dir, e)
            return

        for entry in entries:
            if not entry.is_dir():
                continue
            try:
                v = Version(entry.name)
            except InvalidVersion:
                logger.debug("NuGetSource: skipping non-version folder %s", entry.name)
                continue
            candidates.append((v, entry))

        if not candidates:
            logger.debug("NuGetSource: no version subdirs under %s", pkg_dir)
            return

        # Sort by version descending; prefer stable over prerelease when
        # numeric versions are equal. packaging.Version handles this
        # correctly per SemVer 2.0.
        candidates.sort(key=lambda t: t[0], reverse=True)
        stable = [c for c in candidates if not c[0].is_prerelease]
        ordered = stable if stable else candidates

        rel = self.relative_dll
        if self.arch_resolver is not None:
            rel = self.arch_resolver(rel)

        for _version, version_dir in ordered:
            dll_path = version_dir / rel
            if dll_path.is_file():
                resolved = dll_path.resolve()
                # The version directory name is the canonical NuGet
                # version string (round-trips for both stable and
                # prerelease — packaging.Version normalizes on parse, but
                # the on-disk folder name is the source of truth).
                version = version_dir.name
                for ep_name in self.eps:
                    yield EPEntry(
                        ep_name=ep_name,
                        dll_path=resolved,
                        source=self,
                        version=version,
                    )
                return

        logger.debug(
            "NuGetSource: package %r present but no version has %s",
            self.distribution,
            rel,
        )

    def iter_eps(self) -> Iterable[str]:
        """Return the canonical EP names this source provides."""
        return self.eps


@dataclass(frozen=True)
class DirectorySource(EPSource):
    r"""A directory tree containing one or more registrable plugin DLLs.

    Covers the third-party-installer case (Ryzen AI), the unzipped-GitHub
    -release case (NVIDIA TensorRT-RTX), and the developer-custom-build
    case (``D:\src\onnxruntime\build\Release``).

    Args:
        root: Absolute path to scan, or a path relative to ``env_var``'s
            value. May be a glob pattern.
        dll_patterns: Mapping of canonical ep_name -> filename or relative
            glob to search for under ``root``.
        env_var: Optional environment variable name. If set and the env
            var is unset/empty, the source is silently skipped. If set
            and present, ``root`` is interpreted as relative to that env
            var's value.
        required_marker: Optional sibling filename that must exist in the
            resolved root before any DLL is yielded. Used as a lightweight
            sanity check (e.g., ``onnxruntime_providers_shared.dll`` for
            the Ryzen AI deployment directory).
    """

    root: Path
    dll_patterns: dict[str, str]
    env_var: str | None = None
    required_marker: str | None = None

    def resolve(self) -> Iterator[EPEntry]:
        """Yield one :class:`EPEntry` per matching pattern.

        DirectorySource has no version concept, so every yielded entry
        carries ``version=None``.
        """
        # Resolve env-var gate first: missing env var is a normal "not
        # installed" outcome, not a warning.
        base: Path
        if self.env_var is not None:
            env_value = os.environ.get(self.env_var)
            if not env_value:
                logger.debug("DirectorySource: env var %r unset; skipping", self.env_var)
                return
            env_root = Path(env_value)
            base = env_root / self.root if not self.root.is_absolute() else self.root
        else:
            base = self.root

        # If the user pointed us at a path that doesn't exist, that's
        # configuration drift worth a warning.
        if not base.exists():
            logger.warning("DirectorySource: root %s does not exist; skipping", base)
            return

        # Required-marker sanity check.
        if self.required_marker is not None:
            marker_path = base / self.required_marker
            if not marker_path.exists():
                logger.warning(
                    "DirectorySource: required marker %s missing under %s; skipping",
                    self.required_marker,
                    base,
                )
                return

        for ep_name, pattern in self.dll_patterns.items():
            # Each pattern may be a literal filename or a relative glob.
            matches = list(base.glob(pattern))
            if not matches:
                logger.debug("DirectorySource: no match for %s under %s", pattern, base)
                continue
            # First glob hit wins; multiple matches for one pattern is
            # unusual but tolerated (deterministic by glob order).
            yield EPEntry(
                ep_name=ep_name,
                dll_path=matches[0].resolve(),
                source=self,
            )

    def iter_eps(self) -> Iterable[str]:
        """Return the canonical EP names this source provides (the dll_patterns keys)."""
        return self.dll_patterns.keys()


# ---------------------------------------------------------------------------
# WinAppSDK ExecutionProviderCatalog singleton.
# ---------------------------------------------------------------------------
# Lazy, process-wide initialization of the WinAppSDK Application Runtime
# bootstrap handle and the ``ExecutionProviderCatalog``. The bootstrap
# handle holds the runtime alive for the lifetime of the process; we
# register an ``atexit`` cleanup so it is released on interpreter shutdown.
# ``__del__`` is intentionally NOT used — Python does not guarantee it is
# invoked on shutdown, which would leak the runtime activation.
_winml_catalog_warned_keys: set[str] = set()


def _release_winml_handle(handle: Any) -> None:
    """``atexit`` callback: release the WinAppSDK bootstrap handle."""
    try:
        # ``handle`` is the value returned by ``initialize(...)``; the
        # context-manager protocol's ``__exit__`` deactivates the runtime.
        handle.__exit__(None, None, None)
    except Exception as e:  # pragma: no cover - shutdown best-effort
        logger.debug("WinAppSDK bootstrap handle cleanup raised: %s", e)


@functools.cache
def _get_catalog() -> Any | None:
    """Return the cached ``ExecutionProviderCatalog`` or ``None``.

    Runs exactly once per process via ``functools.cache`` (thread-safe via
    its internal lock). Failures cache as ``None`` and never retry; tests
    reset state via ``_get_catalog.cache_clear()``.

    Returns ``None`` (not raises) when:

    * The WinAppSDK ML Python binding is not importable (no ``wasdk-*``
      install). Logged at DEBUG; this is the common case on machines
      without the optional ``winml-catalog`` extra.
    * The bootstrap initialize call raises. Logged at DEBUG with a
      pointer to the WinAppSDK runtime download page.
    * ``ExecutionProviderCatalog.get_default()`` raises. Logged at WARN.
    """
    # Lazy import so we do not pay the binding-load cost (or fail
    # outright on machines without the wasdk extra) at module import.
    try:
        import winui3.microsoft.windows.ai.machinelearning as winml
        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
            InitializeOptions,
            initialize,
        )
    except ImportError as e:
        logger.debug(
            "WinMLCatalogSource: WinAppSDK ML Python binding not "
            "installed; install the 'winml-catalog' extra to enable "
            "MSIX-delivered EP discovery (%s)",
            e,
        )
        return None

    # Initialize the WinAppSDK Application Runtime bootstrap. The handle
    # holds the runtime active for the rest of the process.
    #
    # InitializeOptions.NONE: silent fail if the OS-level Windows App
    # Runtime is not installed. We log at DEBUG (not WARN) for that case
    # because it's expected — users opt into the Python wasdk packages
    # via the [winml-catalog] extra but may not yet have the runtime
    # installed (which is a separate Microsoft installer at
    # https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads).
    # ON_NO_MATCH_SHOW_UI would open that page in a browser on every
    # invocation — too disruptive for an opt-in capability.
    try:
        handle = initialize(options=InitializeOptions.NONE)
        handle.__enter__()
    except Exception as e:
        logger.debug(
            "WinMLCatalogSource: WinAppSDK bootstrap initialize() "
            "failed (%s). Install the Windows App SDK runtime from "
            "https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads "
            "to enable MSIX-delivered EP discovery.",
            e,
        )
        return None

    # Register cleanup BEFORE accessing the catalog so a catalog
    # error still releases the runtime.
    atexit.register(_release_winml_handle, handle)

    try:
        return winml.ExecutionProviderCatalog.get_default()
    except Exception as e:
        logger.warning(
            "WinMLCatalogSource: ExecutionProviderCatalog.get_default() failed: %s",
            e,
        )
        return None


def _winml_warn_once(key: str, msg: str, *args: Any) -> None:
    """Emit a WARN log the first time we see ``key`` this process."""
    if key in _winml_catalog_warned_keys:
        logger.debug(msg, *args)
        return
    _winml_catalog_warned_keys.add(key)
    logger.warning(msg, *args)


@dataclass(frozen=True)
class WinMLCatalogSource(EPSource):
    """An MSIX EP delivered via the WinAppSDK ``ExecutionProviderCatalog``.

    The on-disk DLL path for an MSIX-delivered EP is decided by the
    Windows package manager and is queryable only at runtime via
    ``provider.library_path`` (populated after ``ensure_ready_async()
    .get()`` returns ``Success``). This source lazily binds to the
    WinAppSDK ML Python binding on first ``resolve()``; if the binding
    is not installed the source yields nothing silently (the optional
    ``winml-catalog`` extra ships the binding).

    Args:
        catalog_name: The provider name reported by the WinAppSDK
            catalog (e.g. ``"VitisAI"``, ``"QNN"``, ``"OpenVINO"``,
            ``"MIGraphX"``, ``"NvTensorRtRtx"``).
        eps: Canonical EP names this source provides. Typically a single
            name, but listed as a tuple for symmetry with the other
            sources.
        auto_download: If ``True``, providers in the ``NotPresent`` ready
            state will be (eventually) downloaded by ``ensure_ready_async``.
            Defaults to ``False`` to avoid surprising the user with a
            multi-second to multi-minute network operation on first call;
            see ``docs/ep-path-design.md`` Interaction section.
    """

    catalog_name: str
    eps: tuple[str, ...]
    auto_download: bool = False

    def resolve(self) -> Iterator[EPEntry]:
        """Yield one :class:`EPEntry` per ready provider.

        Yields nothing (silently) when the WinAppSDK binding is not
        installed. Logs a WARN (once per provider per process) when an
        installed-but-not-ready provider's ``ensure_ready_async`` returns
        a non-Success status. Per the design doc, registration is done
        by the caller via ``ort.register_execution_provider_library`` —
        this source does NOT call ``provider.TryRegister()`` or any of
        the WinAppSDK ``EnsureAndRegisterCertifiedAsync`` /
        ``RegisterCertifiedAsync`` methods.

        Yielded entries carry ``version=None``; probing
        ``provider.version`` is a follow-up (the OQ-2 deferral was
        accepted at the time of Batch A).
        """
        catalog = _get_catalog()
        if catalog is None:
            return

        try:
            providers = catalog.find_all_providers()
        except Exception as e:
            _winml_warn_once(
                f"find_all_providers:{self.catalog_name}",
                "WinMLCatalogSource(%s): find_all_providers() raised: %s",
                self.catalog_name,
                e,
            )
            return

        for provider in providers:
            # One bad provider must not abort the others.
            try:
                yield from self._resolve_provider(provider)
            except Exception as e:  # noqa: PERF203
                _winml_warn_once(
                    f"provider-error:{self.catalog_name}",
                    "WinMLCatalogSource(%s): provider iteration raised %s",
                    self.catalog_name,
                    e,
                )

    def _resolve_provider(self, provider: Any) -> Iterator[EPEntry]:
        """Yield one :class:`EPEntry` per matching, ready catalog provider."""
        # Filter by name first; one catalog returns providers for every
        # vendor and most rows will not match self.catalog_name.
        if getattr(provider, "name", None) != self.catalog_name:
            return

        # Skip providers that are not present on this machine. The design
        # doc explicitly forbids auto-downloading hundreds of MB without
        # opt-in; we honor that via auto_download=False (the default).
        ready_state = getattr(provider, "ready_state", None)
        if ready_state is not None and not self.auto_download and self._is_not_present(ready_state):
            logger.debug(
                "WinMLCatalogSource(%s): provider in NotPresent state; "
                "skipping (auto_download=False)",
                self.catalog_name,
            )
            return

        # ensure_ready_async().get() blocks until the EP is ready or
        # fails. ``library_path`` is populated only after Success.
        try:
            result = provider.ensure_ready_async().get()
        except Exception as e:
            _winml_warn_once(
                f"ensure-ready:{self.catalog_name}",
                "WinMLCatalogSource(%s): ensure_ready_async raised %s",
                self.catalog_name,
                e,
            )
            return

        status = getattr(result, "status", None)
        if status is not None and not self._is_success(status):
            _winml_warn_once(
                f"ensure-ready-status:{self.catalog_name}",
                "WinMLCatalogSource(%s): ensure_ready_async returned "
                "non-Success status %r; skipping",
                self.catalog_name,
                status,
            )
            return

        library_path = getattr(provider, "library_path", "") or ""
        if not library_path:
            # Empty library_path means the EP MSIX was not actually
            # downloaded (e.g., NotPresent provider whose runtime was
            # not gated above). Silent skip.
            logger.debug(
                "WinMLCatalogSource(%s): library_path empty after ensure_ready; skipping",
                self.catalog_name,
            )
            return

        path = Path(library_path)
        for ep_name in self.eps:
            yield EPEntry(
                ep_name=ep_name,
                dll_path=path,
                source=self,
            )

    @staticmethod
    def _is_not_present(ready_state: Any) -> bool:
        """Return True iff the provider's ready_state is ``NotPresent``.

        ``ready_state`` is an enum from the WinAppSDK ML binding; we
        avoid importing the enum type directly (which would mean another
        import that fails when the binding is absent) by comparing on the
        string ``name`` attribute, falling back to ``str(...)``. The
        wasdk 2.0 binding exposes the name as ``NOT_PRESENT``
        (UPPER_SNAKE_CASE) while WinML docs spell it ``NotPresent``
        (PascalCase); normalize underscores + casing to match either form.
        """
        name = getattr(ready_state, "name", None) or str(ready_state)
        return name.replace("_", "").lower().endswith("notpresent")

    @staticmethod
    def _is_success(status: Any) -> bool:
        """Return True iff the ensure-ready status enum is ``Success``.

        Accepts both ``SUCCESS`` and ``Success`` spellings (see
        ``_is_not_present`` rationale).
        """
        name = getattr(status, "name", None) or str(status)
        return name.replace("_", "").lower().endswith("success")

    def iter_eps(self) -> Iterable[str]:
        """Return the canonical EP names this source provides."""
        return self.eps


# ---------------------------------------------------------------------------
# Windows.Management.Deployment.PackageManager singleton (for MSIXPackageSource).
# ---------------------------------------------------------------------------
@functools.cache
def _get_pkg_manager() -> Any | None:
    """Return cached ``PackageManager`` or None when binding unavailable.

    Uses ``find_packages_by_user_security_id("")`` (empty SID = current user)
    which is the only enumeration overload that works without elevation.
    Tests reset via ``_get_pkg_manager.cache_clear()``.
    """
    try:
        from winrt.windows.management.deployment import PackageManager
    except ImportError as e:
        logger.debug(
            "MSIXPackageSource: WinRT PackageManager binding not installed; "
            "install the 'winml-catalog' extra to enable MSIX EP version "
            "discovery (%s)",
            e,
        )
        return None
    try:
        return PackageManager()
    except Exception as e:
        logger.warning("MSIXPackageSource: PackageManager() failed: %s", e)
        return None


def _pkg_version_tuple(version: Any) -> tuple[int, int, int, int]:
    """Convert a ``PackageVersion`` to a comparable tuple."""
    return (
        int(getattr(version, "major", 0)),
        int(getattr(version, "minor", 0)),
        int(getattr(version, "build", 0)),
        int(getattr(version, "revision", 0)),
    )


def _pkg_version_str(version: Any) -> str:
    """Render a ``PackageVersion`` as ``"M.m.b.r"``."""
    return ".".join(str(p) for p in _pkg_version_tuple(version))


@dataclass(frozen=True)
class MSIXPackageSource(EPSource):
    """An MSIX-delivered EP, identified by package-family-name prefix.

    Bypasses the WinAppSDK ``ExecutionProviderCatalog`` (which exposes
    only one version per EP-name) to load a specific installed MSIX
    package version. Use when you need to pin a non-current EP version
    (compat testing, regression isolation, multi-tenant scenarios).

    Args:
        family_name_prefix: Prefix matched against installed-package
            ``PackageFamilyName``. Granularity decides what gets pinned —
            ``"MicrosoftCorporationII.WinML.Qualcomm.QNN.EP."`` spans
            both v1.8 and v2 families; ``"...QNN.EP.1.8_"`` pins to the
            v1.8 line (any build); ``"...QNN.EP.1.8_8wekyb3d8bbwe_"``
            pins to one family exactly. The trailing character (``.``
            or ``_``) is the user's disambiguator against future name
            collisions (e.g., a hypothetical ``EP.10_`` family).
        relative_dll: POSIX-style relative path inside the package's
            ``InstalledPath``. For QNN EP MSIX (verified):
            ``"ExecutionProvider/onnxruntime_providers_qnn.dll"``.
        eps: Canonical EP names this package provides.
        version: Optional secondary pin to one exact installed version
            (e.g. ``"1.8.30.0"``). When ``None`` (typical), the highest
            installed version within any matched family wins.
    """

    family_name_prefix: str
    relative_dll: str
    eps: tuple[str, ...]
    version: str | None = None

    def resolve(self) -> Iterator[EPEntry]:
        """Yield one :class:`EPEntry` per EP this source provides.

        Selection rules (in order):

        1. Filter by ``family_name.startswith(self.family_name_prefix)``.
        2. If :attr:`version` is set, filter to packages whose version
           string equals it.
        3. If multiple packages remain, pick the one with the highest
           ``Package.Id.Version``.
        4. Verify the DLL exists at ``installed_path / relative_dll``.
        5. Yield :class:`EPEntry` for each ``ep`` in :attr:`eps`, with
           ``version`` set to the matched package's ``Package.Id.Version``
           (rendered as ``"M.m.b.r"``).

        Yields nothing (silently) when no matching package is installed,
        when the WinRT binding is unavailable, or when the DLL is missing
        from the matched package.

        Raises:
            ValueError: if :attr:`relative_dll` contains a backslash —
                the field's POSIX-style invariant is enforced at resolve
                time so a hand-constructed source with Windows separators
                fails loudly on Linux instead of silently returning empty.
        """
        if "\\" in self.relative_dll:
            raise ValueError(
                f"MSIXPackageSource.relative_dll must be POSIX-style "
                f"(forward-slash separators); got {self.relative_dll!r}"
            )
        manager = _get_pkg_manager()
        if manager is None:
            return

        try:
            packages = list(manager.find_packages_by_user_security_id(""))
        except Exception as e:
            logger.warning(
                "MSIXPackageSource: find_packages_by_user_security_id raised %s",
                e,
            )
            return

        matching = [
            p for p in packages if str(p.id.family_name).startswith(self.family_name_prefix)
        ]
        if self.version is not None:
            matching = [p for p in matching if _pkg_version_str(p.id.version) == self.version]

        if not matching:
            logger.debug(
                "MSIXPackageSource: no installed package matches prefix=%r version=%r",
                self.family_name_prefix,
                self.version,
            )
            return

        selected = max(matching, key=lambda p: _pkg_version_tuple(p.id.version))
        installed_path = Path(str(selected.installed_path))
        dll_path = installed_path / self.relative_dll
        if not dll_path.is_file():
            logger.warning(
                "MSIXPackageSource: package %s installed at %s but DLL missing at %s",
                selected.id.full_name,
                installed_path,
                dll_path,
            )
            return

        selected_version = _pkg_version_str(selected.id.version)
        for ep_name in self.eps:
            yield EPEntry(
                ep_name=ep_name,
                dll_path=dll_path,
                source=self,
                version=selected_version,
            )

    def iter_eps(self) -> Iterable[str]:
        """Return the canonical EP names this source provides."""
        return self.eps


def _list_msix_eps(
    family_name_prefixes: tuple[str, ...] = (
        "MicrosoftCorporationII.WinML.",
        "WindowsWorkload.EP.",
    ),
) -> list[MSIXPackageSource]:
    """Enumerate installed MSIX EP packages.

    Returns one fully-pinned :class:`MSIXPackageSource` per (family,
    version) found. Each return value is ready to drop into the default
    EP source list and resolvable via ``.resolve()``.

    EP names are auto-detected from the DLL filename inside each package,
    using :meth:`EPCatalog.ep_for_dll` on :data:`EP_CATALOG`. Packages with
    no recognizable EP DLL are skipped silently.

    Args:
        family_name_prefixes: Default catches both publishing channels:
            ``"MicrosoftCorporationII.WinML."`` for the public WinML EP
            catalog channel (Windows Update D-week KB delivery, the
            family the WinAppSDK ``ExecutionProviderCatalog`` binds), and
            ``"WindowsWorkload.EP."`` for the OEM/Windows-Workloads
            channel that provisions EP MSIXes via OEM imaging on
            Copilot+ silicon (e.g., the Intel OpenVINO EP shipping as
            ``WindowsWorkload.EP.Intel.OpenVINO.1.8_*`` on Lunar Lake
            devices). Override with a narrower tuple to filter (e.g.,
            ``("MicrosoftCorporationII.WinML.Qualcomm.",)`` for QNN-only
            listings) or pass ``("",)`` to enumerate every installed
            package.

    Returns:
        List of :class:`MSIXPackageSource` with ``family_name_prefix``
        set to the exact PackageFamilyName (no trailing separator) and
        ``version`` set to the exact installed ``Package.Id.Version``
        string. Round-trip exactness comes from the family-name plus
        version pin together: a ``startswith()`` match on the full
        family name only matches that one family, and the exact
        version filter narrows further to the specific build. Empty
        list if the binding is unavailable or no matching packages
        are installed.
    """
    manager = _get_pkg_manager()
    if manager is None:
        return []

    try:
        packages = list(manager.find_packages_by_user_security_id(""))
    except Exception as e:
        logger.warning("list_msix_eps: find_packages_by_user_security_id raised %s", e)
        return []

    matching = [
        p
        for p in packages
        if any(str(p.id.family_name).startswith(prefix) for prefix in family_name_prefixes)
    ]
    matching.sort(
        key=lambda p: (str(p.id.family_name), _pkg_version_tuple(p.id.version)),
    )

    results: list[MSIXPackageSource] = []
    for p in matching:
        installed_path = Path(str(p.installed_path))
        try:
            ep_dir = installed_path / "ExecutionProvider"
            candidates = list(ep_dir.glob("onnxruntime_providers_*.dll")) if ep_dir.is_dir() else []
            # Fallback: scan one level down if the conventional layout
            # is not used by some future vendor.
            if not candidates:
                candidates = list(installed_path.glob("**/onnxruntime_providers_*.dll"))
        except Exception as e:
            logger.debug(
                "list_msix_eps: cannot scan %s for EP DLLs (%s); skipping",
                installed_path,
                e,
            )
            continue

        ep_name: str | None = None
        chosen_dll: Path | None = None
        for dll in candidates:
            mapped = EP_CATALOG.ep_for_dll(dll.name)
            if mapped is not None:
                ep_name = mapped
                chosen_dll = dll
                break

        if ep_name is None or chosen_dll is None:
            logger.debug(
                "list_msix_eps: package %s has no recognizable EP DLL; skipping",
                p.id.full_name,
            )
            continue

        rel = chosen_dll.relative_to(installed_path).as_posix()
        # Use the exact full family_name (no trailing separator) so the
        # generated source resolves the same package via startswith().
        # Combined with self.version pin, this is exact-match round-trip.
        results.append(
            MSIXPackageSource(
                family_name_prefix=str(p.id.family_name),
                relative_dll=rel,
                eps=(ep_name,),
                version=_pkg_version_str(p.id.version),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Default EP source list.
# ---------------------------------------------------------------------------
def _default_ep_sources() -> list[EPSource]:
    """Default EP source list for this project.

    Order: PyPI sources first (most deterministic, locked by pyproject),
    then ``NuGetSource`` entries (opportunistic pickup of plugin EPs the
    user already restored into the global NuGet cache via a .NET
    project), then ``WinMLCatalogSource`` entries (opportunistic MSIX
    pickup for EPs we don't already have via PyPI / NuGet), then
    ``DirectorySource`` entries gated by env vars (Ryzen AI for
    VitisAI; user-specified for NvTRT-RTX), then ``list_msix_eps()``
    enumeration of every installed MSIX EP package — catches the OEM
    ``WindowsWorkload.EP.*`` channel (Lunar Lake et al.) and historical
    MSIX versions the catalog hides. Dedup by ``(ep_name, canonical
    dll_path)`` collapses any catalog/MSIX overlap; catalog precedence
    wins.

    The ``WinMLCatalogSource`` rows are live: they yield nothing
    silently when the optional ``winml-catalog`` extra is not installed
    (no ``wasdk-*`` packages on this machine). On machines with the
    extra installed, they pick up MSIX-delivered EPs that Windows Update
    has already provisioned.

    The ``NuGetSource`` rows are also live: they yield nothing silently
    when the relevant package is not in ``~/.nuget/packages``. Only EPs
    with a verified NuGet plugin-style package have rows here; vendors
    that ship only via PyPI / installer / MSIX (VitisAI, NvTensorRtRtx,
    MIGraphX as of 2026-05) are intentionally absent.
    """
    return [
        # 1. PyPI plugin wheels — primary source today.
        PyPISource(
            distribution="onnxruntime-ep-openvino",
            relative_dll=("onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll"),
            eps=("OpenVINOExecutionProvider",),
        ),
        PyPISource(
            distribution="onnxruntime-qnn",
            relative_dll="onnxruntime_qnn/libs/{arch}/onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
            arch_folder_map={
                "x64_native": "amd64",
                "x64_on_arm64": "arm64ec",
                # The wheel's "arm64ec" folder is actually an ARM64X hybrid
                # binary (confirmed via `dumpbin /headers`: machine (x64)
                # (ARM64X)) — loadable natively by a native-ARM64 process
                # too, not just an x64-hosted one. If a future onnxruntime-qnn
                # release ever splits this into separate arm64ec-only and
                # arm64x folders, this mapping needs updating by hand.
                "arm64_native": "arm64ec",
            },
        ),
        # 2. NuGet plugin packages — picked up if a .NET project on this
        #    machine has restored them into ``~/.nuget/packages``. Vendors
        #    we have NuGet rows for are limited to those with a public
        #    plugin-style package on nuget.org as of 2026-05; VitisAI,
        #    NvTensorRtRtx and MIGraphX have no canonical NuGet plugin
        #    package and are intentionally absent here. (See Phase 1
        #    research notes in the feat/update-pkg-deps PR.)
        NuGetSource(
            distribution="Intel.ML.OnnxRuntime.EP.OpenVINO",
            relative_dll=("runtimes/win-x64/native/onnxruntime_providers_openvino_plugin.dll"),
            eps=("OpenVINOExecutionProvider",),
        ),
        NuGetSource(
            distribution="Qualcomm.ML.OnnxRuntime.QNN",
            relative_dll=("runtimes/win-arm64/native/onnxruntime_providers_qnn.dll"),
            eps=("QNNExecutionProvider",),
        ),
        # 3. WinAppSDK ExecutionProviderCatalog — opportunistic MSIX
        #    pickup for any EP we don't already have via PyPI. Order
        #    matters: PyPI wins if both are present (more deterministic,
        #    locked by pyproject vs Windows-Update-managed MSIX).
        WinMLCatalogSource(
            catalog_name="OpenVINOExecutionProvider",
            eps=("OpenVINOExecutionProvider",),
        ),
        WinMLCatalogSource(
            catalog_name="QNNExecutionProvider",
            eps=("QNNExecutionProvider",),
        ),
        WinMLCatalogSource(
            catalog_name="VitisAIExecutionProvider",
            eps=("VitisAIExecutionProvider",),
        ),
        WinMLCatalogSource(
            catalog_name="MIGraphXExecutionProvider",
            eps=("MIGraphXExecutionProvider",),
        ),
        WinMLCatalogSource(
            catalog_name="NvTensorRTRTXExecutionProvider",
            eps=("NvTensorRTRTXExecutionProvider",),
        ),
        # 4. Well-known third-party installer drops, gated by env var so
        #    they no-op on machines without the installer present.
        DirectorySource(
            root=Path("deployment"),
            env_var="RYZEN_AI_INSTALLATION_PATH",
            dll_patterns={
                "VitisAIExecutionProvider": "onnxruntime_providers_vitisai.dll",
            },
            required_marker="onnxruntime_providers_shared.dll",
        ),
        # 5. NVIDIA TensorRT-RTX EP unzipped from the GitHub release.
        #    User points NVIDIA_TRT_RTX_EP at the ZIP root; we glob for
        #    the plugin DLL with no required marker (the ZIP is flat).
        #    Empty relative root means "use env_var value as-is".
        DirectorySource(
            root=Path(),
            env_var="NVIDIA_TRT_RTX_EP",
            dll_patterns={
                "NvTensorRTRTXExecutionProvider": ("onnxruntime_providers_nv_tensorrt_rtx.dll"),
            },
        ),
        # 6. Live MSIX enumeration — catches the OEM
        #    ``WindowsWorkload.EP.*`` channel and historical MSIX
        #    versions the WinML catalog hides. Appended after
        #    WinMLCatalogSource so catalog precedence wins on overlap;
        #    dedup in discover_all_eps collapses identical-DLL dups.
        *_list_msix_eps(),
    ]


# ---------------------------------------------------------------------------
# Override mechanisms.
# ---------------------------------------------------------------------------
def _parse_winmlcli_ep_path() -> list[EPSource]:
    """Parse the ``WINMLCLI_EP_PATH`` env var into ``DirectorySource`` entries.

    The env var is a path-list using OS-conventional separators (``;`` on
    Windows, ``:`` elsewhere — same semantics as the shell ``PATH``).
    Each entry is treated as a directory; we scan it for every plugin DLL
    filename in :data:`EP_CATALOG` so the user does not have to specify
    which EP the directory provides.

    Returns an empty list when ``WINMLCLI_EP_PATH`` is unset or empty.
    Non-existent entries log a WARN and are skipped (matches the
    ``DirectorySource`` "configured-but-nonexistent root" pattern).
    """
    raw = os.environ.get("WINMLCLI_EP_PATH")
    if not raw:
        return []
    entries = [e.strip() for e in raw.split(os.pathsep) if e.strip()]
    if not entries:
        return []

    sources: list[EPSource] = []
    for entry in entries:
        p = Path(entry)
        if not p.is_dir():
            logger.warning("WINMLCLI_EP_PATH entry %r is not a directory; skipping", entry)
            continue
        logger.debug("WINMLCLI_EP_PATH override: scanning %s", entry)
        for ep in EP_CATALOG.all_eps():
            dll = EP_CATALOG.dll_name_for(ep)
            if not dll:
                continue  # bundled EPs have no DLL filename
            sources.append(DirectorySource(root=p, dll_patterns={ep: dll}))
    return sources


# ---------------------------------------------------------------------------
# Discovery algorithm.
# ---------------------------------------------------------------------------
def discover_all_eps(
    extra_sources: list[EPSource] | None = None,
    *,
    extra_sources_after: list[EPSource] | None = None,
) -> list[EPEntry]:
    """Walk the default EP source list and return a flat ``list[EPEntry]``.

    Single canonical discovery entry point. Used by ``winml sys --list-ep``
    to enumerate every source contributing each EP (so users can see when
    a later source is being shadowed by a higher-precedence one) and by
    ``WinMLEPRegistry`` to populate its cached ``_discovered`` list.

    The returned list preserves source-walk order. The first entry per
    ``ep_name`` carries ``status="primary"``; subsequent entries for the
    same ``ep_name`` carry ``status="shadowed"`` (precedence determined
    by EPSource ordering — see below).

    Precedence (highest first):

    1. ``extra_sources`` (programmatic override; useful for tests)
    2. ``WINMLCLI_EP_PATH`` env-var entries (parsed into FilesystemSources)
    3. The default EP source list (``_default_ep_sources()``)
    4. ``extra_sources_after`` (lowest precedence; used by inventory CLI)
    """
    sources: list[EPSource] = []
    if extra_sources:
        sources.extend(extra_sources)
    sources.extend(_parse_winmlcli_ep_path())
    sources.extend(_default_ep_sources())
    if extra_sources_after:
        sources.extend(extra_sources_after)

    result: list[EPEntry] = []
    seen: set[str] = set()
    # Track (ep_name, canonical_dll_path) tuples already emitted so two
    # different sources resolving to the SAME on-disk DLL collapse to one
    # row. This is the WinML-Catalog vs MSIX-PackageManager overlap: both
    # legitimately surface the same Microsoft-published EP DLL, and we
    # want the higher-precedence source's attribution to win.
    seen_paths: set[tuple[str, str]] = set()
    for source in sources:
        try:
            it = source.resolve()
        except Exception as e:
            logger.error("Source %r failed to resolve: %s", source, e)
            continue

        try:
            for entry in it:
                # Skip the disk check for BuiltinSource entries: they
                # carry Path("") (sentinel — pre-loaded by ORT, no DLL
                # to discover), and is_file() would silently drop them.
                if not entry.is_built_in() and not entry.dll_path.is_file():
                    logger.warning(
                        "EP %s: source %r produced %s which is not a file",
                        entry.ep_name,
                        source,
                        entry.dll_path,
                    )
                    continue
                # Dedup by (ep_name, canonical dll_path). os.path.normcase
                # collapses Windows case-insensitivity (``C:\Foo`` vs
                # ``c:\foo``); os.path.normpath collapses ``..`` and
                # redundant separators. The first occurrence wins —
                # precedence order is preserved so the higher-precedence
                # source's attribution survives.
                path_key = os.path.normcase(os.path.normpath(str(entry.dll_path)))
                dedup_key = (entry.ep_name, path_key)
                if dedup_key in seen_paths:
                    logger.debug(
                        "EP %s: dedup — %r already attributed; dropping source %r",
                        entry.ep_name,
                        entry.dll_path,
                        source,
                    )
                    continue
                seen_paths.add(dedup_key)
                if entry.ep_name in seen:
                    # Source ordering decides precedence; later sources
                    # land as shadowed.
                    final = dataclasses.replace(entry, status="shadowed")
                else:
                    seen.add(entry.ep_name)
                    final = (
                        entry
                        if entry.status == "primary"
                        else dataclasses.replace(entry, status="primary")
                    )
                result.append(final)
                logger.debug(
                    "EP %s [%s] -> %s from %r",
                    final.ep_name,
                    final.status,
                    final.dll_path,
                    final.source,
                )
        except Exception as e:
            logger.error("Source %r failed mid-iteration: %s", source, e)
            continue

    return result


__all__ = [
    "EP_CATALOG",
    "BuiltinSource",
    "DirectorySource",
    "EPCatalog",
    "EPEntry",
    "EPSource",
    "MSIXPackageSource",
    "NuGetSource",
    "PyPISource",
    "WinMLCatalogSource",
    "discover_all_eps",
]
