# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unified execution-provider discovery via ``EP_PATH``.

This module replaces the legacy ``EP_PLUGIN_REGISTRY`` dict (which only
modeled PyPI-installed plugin EPs) with an ordered list of typed
``EpSource`` entries, analogous to the OS ``PATH`` environment variable.
Each entry knows how to resolve itself for the current machine and yields
``(ep_name, absolute_dll_path)`` pairs.

See ``docs/ep-path-design.md`` for the full design rationale, including
the per-origin x per-EP map and the migration plan.

Public API:

* :data:`EP_PATH`: ordered ``list[EpSource]`` consulted by the registry.
* :data:`EP_DLL_NAMES`: canonical EP-name -> list-of-DLL-filenames table.
* :class:`PyPiSource`: pip-installed plugin EP wheels.
* :class:`FilesystemSource`: directory drops (installer, unzipped archive,
  custom build).
* :class:`WinMlCatalogSource`: WinAppSDK ``ExecutionProviderCatalog``
  MSIX-delivered EPs. Lazily imports the WinAppSDK ML Python binding;
  yields nothing silently when the binding is not installed.
* :class:`EpSource`: tagged-union of the three.
* :func:`discover_eps`: walk ``EP_PATH`` (plus any extras) and yield
  ``(ep_name, dll_path, source)`` triples with first-hit-wins semantics.
"""

from __future__ import annotations

import atexit
import functools
import logging
import os
import platform
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EP-name alias table.
# ---------------------------------------------------------------------------
# Maps every known alias spelling to its canonical EP name. The canonical
# name is what ORT's WinML EP Catalog registers under (per Microsoft Learn
# documentation for the supported execution providers). Some upstream
# documentation and other call sites use a different spelling (e.g.
# NVIDIA's GitHub README uses PascalCase ``NvTensorRTRTXExecutionProvider``
# while WinML's ``ExecutionProviderCatalog`` registers under the camelCase
# ``NvTensorRtRtxExecutionProvider``). Use :func:`canonicalize_ep_name` at
# any boundary that compares an EP name string to a name reported by ORT.
#
# This is intentionally a finite, explicit allow-list: case-insensitive
# matching would also accept misspellings (e.g.
# ``NVTENSORRTRTXEXECUTIONPROVIDER``) that ORT itself would reject.
EP_NAME_ALIASES: dict[str, str] = {
    # NVIDIA TensorRT-RTX. Canonical form is camelCase per WinML EP
    # Catalog; PascalCase appears in NVIDIA's standalone-zip docs.
    "NvTensorRTRTXExecutionProvider": "NvTensorRtRtxExecutionProvider",
}


def canonicalize_ep_name(name: str) -> str:
    """Normalize an EP-name alias to its canonical form.

    The canonical form is the spelling under which ORT registers the EP
    (i.e. the spelling reported by ``ort.get_ep_devices()`` and used by
    ``register_execution_provider_library``). Names that are not in the
    alias table are returned unchanged so unknown EP spellings (including
    typos) flow through to ORT for diagnosis.

    Args:
        name: An EP name in any known spelling.

    Returns:
        The canonical EP name. Identity for unknown names.
    """
    return EP_NAME_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Canonical EP-name -> DLL filename table.
# ---------------------------------------------------------------------------
# Used by FilesystemSource (when scanning a directory for any registrable
# DLL) and by the ``MODELKIT_EP_PATH`` env-var override path. Keys are
# always the canonical EP name; pass non-canonical aliases through
# :func:`canonicalize_ep_name` before consulting this table.
EP_DLL_NAMES: dict[str, list[str]] = {
    "OpenVINOExecutionProvider": [
        "onnxruntime_providers_openvino_plugin.dll",
        "libonnxruntime_providers_openvino_plugin.so",
    ],
    "QNNExecutionProvider": [
        "onnxruntime_providers_qnn.dll",
    ],
    "VitisAIExecutionProvider": [
        "onnxruntime_providers_vitisai.dll",
    ],
    # TODO(ep_path): MIGraphX DLL leaf is unverified; mirrors the VitisAI
    # naming convention. Confirm by inspecting an installed MSIX. See
    # docs/ep-path-design.md TODO #4.
    "MIGraphXExecutionProvider": [
        "onnxruntime_providers_migraphx.dll",
    ],
    "NvTensorRtRtxExecutionProvider": [
        "onnxruntime_providers_nv_tensorrt_rtx.dll",
        "libonnxruntime_providers_nv_tensorrt_rtx.so",
    ],
}


# ---------------------------------------------------------------------------
# Reverse lookup: DLL filename -> canonical EP name.
# ---------------------------------------------------------------------------
# Inverse of :data:`EP_DLL_NAMES`, derived once at module load. Used by
# :func:`list_msix_eps` to identify which EP a discovered MSIX package
# provides without hardcoding vendor-specific naming patterns.
_DLL_TO_EP_NAME: dict[str, str] = {
    dll: ep
    for ep, dll_list in EP_DLL_NAMES.items()
    for dll in dll_list
}


# ---------------------------------------------------------------------------
# Hardware-compatibility table and helpers.
# ---------------------------------------------------------------------------
# Maps each canonical EP name to the vendor substrings (case-insensitive)
# that must appear in at least one detected hardware vendor string for the
# EP to be considered compatible with this machine. Empty set = no vendor
# requirement (e.g., CPU, DML, Azure work everywhere). Unknown EP names
# default to compatible (forward-compat for new EPs not yet in the table).
#
# Substring matching tolerates the variety of vendor-string spellings
# Windows reports (``"Intel(R) Corporation"`` vs ``"Intel"`` vs ``"Intel Corp"``)
# without needing per-system normalization.
_EP_VENDOR_REQUIREMENT: dict[str, set[str]] = {
    "QNNExecutionProvider":           {"Qualcomm"},
    "OpenVINOExecutionProvider":      {"Intel"},
    "VitisAIExecutionProvider":       {"AMD"},
    "MIGraphXExecutionProvider":      {"AMD"},
    "NvTensorRtRtxExecutionProvider": {"NVIDIA"},
    "DmlExecutionProvider":           set(),
    "CPUExecutionProvider":           set(),
    "AzureExecutionProvider":         set(),
}


@functools.cache
def _get_detected_vendors() -> frozenset[str]:
    """Return the union of vendor identification strings from sysinfo.

    Aggregates ``manufacturer`` and ``name`` across detected GPUs and
    NPUs (CPU vendor is not exposed by every WMI provider; CPU is
    treated as universally compatible via the empty-requirement entry
    in :data:`_EP_VENDOR_REQUIREMENT`). Both fields are included because
    Windows reports vendor inconsistently — sometimes the manufacturer
    is the IHV (``"Qualcomm Incorporated"``) and sometimes a parent
    company (``"Microsoft Corporation"`` for OEM-rebranded devices).
    The substring match in :func:`_ep_is_compatible` tolerates either.

    Cached process-wide; tests reset via ``_get_detected_vendors.cache_clear()``.
    Returns an empty frozenset if hardware detection fails — in which case
    every EP with a non-empty vendor requirement reports as incompatible.
    """
    strings: set[str] = set()
    try:
        from .sysinfo.hardware import GPU, NPU
    except ImportError as e:
        logger.debug("Hardware detection unavailable (%s); compat checks will fail", e)
        return frozenset()

    for cls in (GPU, NPU):
        try:
            for hw in cls.get_all():
                for attr in ("manufacturer", "name"):
                    value = getattr(hw, attr, None)
                    if value:
                        strings.add(str(value))
        except Exception as e:  # pragma: no cover - WMI failure best-effort
            logger.debug("%s.get_all() raised %s; skipping", cls.__name__, e)

    return frozenset(strings)


def _ep_is_compatible(ep_name: str) -> bool:
    """Return True iff ``ep_name`` has compatible hardware on this machine.

    Looks up the EP in :data:`_EP_VENDOR_REQUIREMENT`:

    * Empty requirement -> always compatible (CPU, DML, Azure).
    * Non-empty requirement -> compatible iff at least one required vendor
      substring appears (case-insensitively) in any detected vendor string.
    * Unknown EP name -> compatible (forward-compat default).

    Args:
        ep_name: Canonical EP name (caller should pass through
            :func:`canonicalize_ep_name` first if the input is an alias).
    """
    required = _EP_VENDOR_REQUIREMENT.get(ep_name, set())
    if not required:
        return True
    detected = _get_detected_vendors()
    return any(req.lower() in v.lower() for req in required for v in detected)


# ---------------------------------------------------------------------------
# Architecture resolver helpers.
# ---------------------------------------------------------------------------


def _qnn_arch_resolver(rel_template: str) -> str:
    """Pick ``arm64ec`` vs ``amd64`` for the QNN PyPI wheel layout.

    The ``onnxruntime-qnn`` wheel ships
    ``onnxruntime_qnn/libs/{amd64|arm64ec}/onnxruntime_providers_qnn.dll``;
    we choose the variant matching the host ``platform.machine()``.
    """
    arch = "arm64ec" if platform.machine().lower() in ("arm64", "aarch64") else "amd64"
    return rel_template.format(arch=arch)


# ---------------------------------------------------------------------------
# EpSource ABC + concrete dataclass implementations.
# ---------------------------------------------------------------------------


class EpSource(ABC):
    """Abstract base for any source that can yield ``(ep_name, dll_path)``.

    Four concrete subclasses cover the origins documented in
    ``docs/ep-path-design.md``: :class:`PyPiSource`,
    :class:`FilesystemSource`, :class:`WinMlCatalogSource`, and
    :class:`MsixPackageSource`. Subclasses are frozen dataclasses; this
    base provides the shared :meth:`is_compatible` body and documents
    the :meth:`resolve` / :meth:`iter_eps` contract.
    """

    @abstractmethod
    def resolve(self) -> Iterator[tuple[str, Path]]:
        """Yield ``(canonical_ep_name, absolute_dll_path)`` zero or more times.

        Errors during resolution should be logged and swallowed (yield
        nothing) — :func:`discover_eps` tolerates source-level failures
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
        return all(_ep_is_compatible(ep) for ep in self.iter_eps())


@dataclass(frozen=True)
class PyPiSource(EpSource):
    """A pip-installed plugin EP wheel.

    The DLL path is computed lazily via
    ``importlib.metadata.distribution(name).locate_file(rel)`` so it
    follows whichever venv is currently active.

    Args:
        distribution: PyPI distribution name, e.g. ``"onnxruntime-ep-openvino"``.
        relative_dll: Path inside the wheel, POSIX-style. May contain
            ``{arch}`` placeholders that ``arch_resolver`` substitutes.
        eps: Canonical EP names this source provides (typically a single name).
        arch_resolver: Optional ``Callable[[str], str]`` that takes the
            ``relative_dll`` template and returns a substituted relative
            path tweaked per machine architecture. ``None`` means the
            ``relative_dll`` is used as-is.
    """

    distribution: str
    relative_dll: str
    eps: tuple[str, ...]
    arch_resolver: Callable[[str], str] | None = None

    def resolve(self) -> Iterator[tuple[str, Path]]:
        """Yield ``(ep_name, abs_path)`` for each EP this source provides.

        Yields nothing (silently) when the distribution is not installed —
        that is the common case for optional EPs and is not an error.
        Logs a warning if the distribution is installed but the file is
        missing.
        """
        try:
            dist = metadata.distribution(self.distribution)
        except metadata.PackageNotFoundError:
            logger.debug(
                "PyPiSource: distribution %r not installed; skipping", self.distribution
            )
            return

        rel = self.relative_dll
        if self.arch_resolver is not None:
            rel = self.arch_resolver(rel)

        path = Path(str(dist.locate_file(rel)))
        if not path.exists():
            logger.warning(
                "PyPiSource: distribution %r installed but DLL missing at %s",
                self.distribution,
                path,
            )
            return

        for ep_name in self.eps:
            yield ep_name, path

    def iter_eps(self) -> Iterable[str]:
        """Return the canonical EP names this source provides."""
        return self.eps


@dataclass(frozen=True)
class FilesystemSource(EpSource):
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

    def resolve(self) -> Iterator[tuple[str, Path]]:
        """Yield ``(ep_name, abs_path)`` for each pattern that matches."""
        # Resolve env-var gate first: missing env var is a normal "not
        # installed" outcome, not a warning.
        base: Path
        if self.env_var is not None:
            env_value = os.environ.get(self.env_var)
            if not env_value:
                logger.debug(
                    "FilesystemSource: env var %r unset; skipping", self.env_var
                )
                return
            env_root = Path(env_value)
            base = env_root / self.root if not self.root.is_absolute() else self.root
        else:
            base = self.root

        # If the user pointed us at a path that doesn't exist, that's
        # configuration drift worth a warning.
        if not base.exists():
            logger.warning(
                "FilesystemSource: root %s does not exist; skipping", base
            )
            return

        # Required-marker sanity check.
        if self.required_marker is not None:
            marker_path = base / self.required_marker
            if not marker_path.exists():
                logger.warning(
                    "FilesystemSource: required marker %s missing under %s; skipping",
                    self.required_marker,
                    base,
                )
                return

        for ep_name, pattern in self.dll_patterns.items():
            # Each pattern may be a literal filename or a relative glob.
            matches = list(base.glob(pattern))
            if not matches:
                logger.debug(
                    "FilesystemSource: no match for %s under %s", pattern, base
                )
                continue
            # First glob hit wins; multiple matches for one pattern is
            # unusual but tolerated (deterministic by glob order).
            yield ep_name, matches[0].resolve()

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
            "WinMlCatalogSource: WinAppSDK ML Python binding not "
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
            "WinMlCatalogSource: WinAppSDK bootstrap initialize() "
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
            "WinMlCatalogSource: ExecutionProviderCatalog.get_default() "
            "failed: %s",
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
class WinMlCatalogSource(EpSource):
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

    def resolve(self) -> Iterator[tuple[str, Path]]:
        """Yield ``(ep_name, abs_path)`` for each EP this source provides.

        Yields nothing (silently) when the WinAppSDK binding is not
        installed. Logs a WARN (once per provider per process) when an
        installed-but-not-ready provider's ``ensure_ready_async`` returns
        a non-Success status. Per the design doc, registration is done
        by the caller via ``ort.register_execution_provider_library`` —
        this source does NOT call ``provider.TryRegister()`` or any of
        the WinAppSDK ``EnsureAndRegisterCertifiedAsync`` /
        ``RegisterCertifiedAsync`` methods.
        """
        catalog = _get_catalog()
        if catalog is None:
            return

        try:
            providers = catalog.find_all_providers()
        except Exception as e:
            _winml_warn_once(
                f"find_all_providers:{self.catalog_name}",
                "WinMlCatalogSource(%s): find_all_providers() raised: %s",
                self.catalog_name,
                e,
            )
            return

        for provider in providers:
            # One bad provider must not abort the others.
            try:
                yield from self._resolve_provider(provider)
            except Exception as e:
                _winml_warn_once(
                    f"provider-error:{self.catalog_name}",
                    "WinMlCatalogSource(%s): provider iteration raised %s",
                    self.catalog_name,
                    e,
                )

    def _resolve_provider(self, provider: Any) -> Iterator[tuple[str, Path]]:
        """Yield ``(ep_name, path)`` for a single catalog provider."""
        # Filter by name first; one catalog returns providers for every
        # vendor and most rows will not match self.catalog_name.
        if getattr(provider, "name", None) != self.catalog_name:
            return

        # Skip providers that are not present on this machine. The design
        # doc explicitly forbids auto-downloading hundreds of MB without
        # opt-in; we honor that via auto_download=False (the default).
        ready_state = getattr(provider, "ready_state", None)
        if (
            ready_state is not None
            and not self.auto_download
            and self._is_not_present(ready_state)
        ):
            logger.debug(
                "WinMlCatalogSource(%s): provider in NotPresent state; "
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
                "WinMlCatalogSource(%s): ensure_ready_async raised %s",
                self.catalog_name,
                e,
            )
            return

        status = getattr(result, "status", None)
        if status is not None and not self._is_success(status):
            _winml_warn_once(
                f"ensure-ready-status:{self.catalog_name}",
                "WinMlCatalogSource(%s): ensure_ready_async returned "
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
                "WinMlCatalogSource(%s): library_path empty after "
                "ensure_ready; skipping",
                self.catalog_name,
            )
            return

        path = Path(library_path)
        for ep_name in self.eps:
            yield ep_name, path

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
# Windows.Management.Deployment.PackageManager singleton (for MsixPackageSource).
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
            "MsixPackageSource: WinRT PackageManager binding not installed; "
            "install the 'winml-catalog' extra to enable MSIX EP version "
            "discovery (%s)",
            e,
        )
        return None
    try:
        return PackageManager()
    except Exception as e:
        logger.warning("MsixPackageSource: PackageManager() failed: %s", e)
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
class MsixPackageSource(EpSource):
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

    def resolve(self) -> Iterator[tuple[str, Path]]:
        """Yield ``(ep_name, abs_path)`` for the matched MSIX package.

        Selection rules (in order):

        1. Filter by ``family_name.startswith(self.family_name_prefix)``.
        2. If :attr:`version` is set, filter to packages whose version
           string equals it.
        3. If multiple packages remain, pick the one with the highest
           ``Package.Id.Version``.
        4. Verify the DLL exists at ``installed_path / relative_dll``.
        5. Yield ``(ep_name, abs_path)`` for each ``ep`` in :attr:`eps`.

        Yields nothing (silently) when no matching package is installed,
        when the WinRT binding is unavailable, or when the DLL is missing
        from the matched package.
        """
        manager = _get_pkg_manager()
        if manager is None:
            return

        try:
            packages = list(manager.find_packages_by_user_security_id(""))
        except Exception as e:
            logger.warning(
                "MsixPackageSource: find_packages_by_user_security_id raised %s",
                e,
            )
            return

        matching = [
            p for p in packages
            if str(p.id.family_name).startswith(self.family_name_prefix)
        ]
        if self.version is not None:
            matching = [p for p in matching if _pkg_version_str(p.id.version) == self.version]

        if not matching:
            logger.debug(
                "MsixPackageSource: no installed package matches prefix=%r version=%r",
                self.family_name_prefix,
                self.version,
            )
            return

        selected = max(matching, key=lambda p: _pkg_version_tuple(p.id.version))
        installed_path = Path(str(selected.installed_path))
        dll_path = installed_path / self.relative_dll
        if not dll_path.is_file():
            logger.warning(
                "MsixPackageSource: package %s installed at %s but DLL missing at %s",
                selected.id.full_name,
                installed_path,
                dll_path,
            )
            return

        for ep_name in self.eps:
            yield ep_name, dll_path

    def iter_eps(self) -> Iterable[str]:
        """Return the canonical EP names this source provides."""
        return self.eps


def list_msix_eps(
    family_name_prefix: str = "MicrosoftCorporationII.WinML.",
) -> list[MsixPackageSource]:
    """Enumerate installed MSIX EP packages.

    Returns one fully-pinned :class:`MsixPackageSource` per (family,
    version) found. Each return value is ``EP_PATH``-ready (drop into
    the list) and resolvable via ``.resolve()``.

    EP names are auto-detected from the DLL filename inside each package,
    using the inverse of :data:`EP_DLL_NAMES`. Packages with no
    recognizable EP DLL are skipped silently.

    Args:
        family_name_prefix: Default catches all WinML-catalog EP MSIXes
            published by Microsoft. Override with a narrower prefix to
            filter (e.g., ``"MicrosoftCorporationII.WinML.Qualcomm."``
            for QNN-only listings).

    Returns:
        List of :class:`MsixPackageSource` with ``family_name_prefix``
        set to the exact PackageFamilyName plus a trailing ``"_"`` for
        round-trip pinning, and ``version`` set to the exact installed
        ``Package.Id.Version`` string. Empty list if the binding is
        unavailable or no matching packages are installed.
    """
    manager = _get_pkg_manager()
    if manager is None:
        return []

    try:
        packages = list(manager.find_packages_by_user_security_id(""))
    except Exception as e:
        logger.warning(
            "list_msix_eps: find_packages_by_user_security_id raised %s", e
        )
        return []

    matching = [p for p in packages if str(p.id.family_name).startswith(family_name_prefix)]
    matching.sort(
        key=lambda p: (str(p.id.family_name), _pkg_version_tuple(p.id.version)),
    )

    results: list[MsixPackageSource] = []
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
            mapped = _DLL_TO_EP_NAME.get(dll.name)
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
            MsixPackageSource(
                family_name_prefix=str(p.id.family_name),
                relative_dll=rel,
                eps=(ep_name,),
                version=_pkg_version_str(p.id.version),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Default EP_PATH per platform.
# ---------------------------------------------------------------------------


def _default_ep_path_windows() -> list[EpSource]:
    """Default ``EP_PATH`` for Windows hosts.

    Order: PyPI sources first (most deterministic, locked by pyproject),
    then ``WinMlCatalogSource`` entries (opportunistic MSIX pickup for
    EPs we don't already have via PyPI), then ``FilesystemSource``
    entries gated by env vars (Ryzen AI for VitisAI; user-specified for
    NvTRT-RTX).

    The ``WinMlCatalogSource`` rows are live: they yield nothing
    silently when the optional ``winml-catalog`` extra is not installed
    (no ``wasdk-*`` packages on this machine). On machines with the
    extra installed, they pick up MSIX-delivered EPs that Windows Update
    has already provisioned.
    """
    return [
        # 1. PyPI plugin wheels — primary source today.
        PyPiSource(
            distribution="onnxruntime-ep-openvino",
            relative_dll=(
                "onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll"
            ),
            eps=("OpenVINOExecutionProvider",),
        ),
        PyPiSource(
            distribution="onnxruntime-qnn",
            relative_dll="onnxruntime_qnn/libs/{arch}/onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
            arch_resolver=_qnn_arch_resolver,
        ),
        # 2. WinAppSDK ExecutionProviderCatalog — opportunistic MSIX
        #    pickup for any EP we don't already have via PyPI. Order
        #    matters: PyPI wins if both are present (more deterministic,
        #    locked by pyproject vs Windows-Update-managed MSIX).
        WinMlCatalogSource(
            catalog_name="OpenVINOExecutionProvider",
            eps=("OpenVINOExecutionProvider",),
        ),
        WinMlCatalogSource(
            catalog_name="QNNExecutionProvider",
            eps=("QNNExecutionProvider",),
        ),
        WinMlCatalogSource(
            catalog_name="VitisAIExecutionProvider",
            eps=("VitisAIExecutionProvider",),
        ),
        WinMlCatalogSource(
            catalog_name="MIGraphXExecutionProvider",
            eps=("MIGraphXExecutionProvider",),
        ),
        WinMlCatalogSource(
            catalog_name="NvTensorRtRtxExecutionProvider",
            eps=("NvTensorRtRtxExecutionProvider",),
        ),
        # 3. Well-known third-party installer drops, gated by env var so
        #    they no-op on machines without the installer present.
        FilesystemSource(
            root=Path("deployment"),
            env_var="RYZEN_AI_INSTALLATION_PATH",
            dll_patterns={
                "VitisAIExecutionProvider": "onnxruntime_providers_vitisai.dll",
            },
            required_marker="onnxruntime_providers_shared.dll",
        ),
        # 4. NVIDIA TensorRT-RTX EP unzipped from the GitHub release.
        #    User points NVIDIA_TRT_RTX_EP at the ZIP root; we glob for
        #    the plugin DLL with no required marker (the ZIP is flat).
        #    Empty relative root means "use env_var value as-is".
        FilesystemSource(
            root=Path(),
            env_var="NVIDIA_TRT_RTX_EP",
            dll_patterns={
                "NvTensorRtRtxExecutionProvider": (
                    "onnxruntime_providers_nv_tensorrt_rtx.dll"
                ),
            },
        ),
    ]


def _default_ep_path_linux() -> list[EpSource]:
    """Default ``EP_PATH`` for Linux hosts.

    Only PyPI plugins; no MSIX, no Ryzen AI Windows installer.
    Note: ``onnxruntime-qnn`` ships Linux aarch64 wheels but no x86_64
    wheel as of 2026-04-27 (design doc TODO #6, resolved); we still list
    the source — it just yields nothing on x86_64 Linux because
    ``importlib.metadata.distribution`` will not find an installed wheel.
    """
    return [
        PyPiSource(
            distribution="onnxruntime-ep-openvino",
            relative_dll=(
                "onnxruntime_ep_openvino/"
                "libonnxruntime_providers_openvino_plugin.so"
            ),
            eps=("OpenVINOExecutionProvider",),
        ),
    ]


def _default_ep_path_for_platform() -> list[EpSource]:
    if os.name == "nt":
        return _default_ep_path_windows()
    if platform.system().lower() == "linux":
        return _default_ep_path_linux()
    # macOS / other: no plugin EPs ship today.
    return []


# Public default. Mutable on purpose so consumers / tests can append.
EP_PATH: list[EpSource] = _default_ep_path_for_platform()


# ---------------------------------------------------------------------------
# Override mechanisms.
# ---------------------------------------------------------------------------


def _parse_modelkit_ep_path() -> list[EpSource]:
    """Parse the ``MODELKIT_EP_PATH`` env var into ``FilesystemSource`` entries.

    The env var is a path-list using OS-conventional separators (``;`` on
    Windows, ``:`` elsewhere). Each entry is treated as a directory; we
    scan it for every filename in :data:`EP_DLL_NAMES` so the user does
    not have to specify which EP the directory provides.

    Returns an empty list when ``MODELKIT_EP_PATH`` is unset or empty.
    """
    raw = os.environ.get("MODELKIT_EP_PATH")
    if not raw:
        return []
    sep = ";" if os.name == "nt" else os.pathsep
    entries = [e.strip() for e in raw.split(sep) if e.strip()]
    if not entries:
        return []

    # Build the inverse-of-EP_DLL_NAMES dict_patterns so each entry
    # scans for every known plugin DLL filename. We pick the first DLL
    # name per EP as the search pattern; FilesystemSource will glob for
    # it under the root.
    patterns = {ep: dll_names[0] for ep, dll_names in EP_DLL_NAMES.items() if dll_names}
    sources: list[EpSource] = []
    for entry in entries:
        logger.debug("MODELKIT_EP_PATH override: scanning %s", entry)
        sources.append(
            FilesystemSource(
                root=Path(entry),
                dll_patterns=patterns,
            )
        )
    return sources


# ---------------------------------------------------------------------------
# Discovery algorithm.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedEp:
    """One (ep_name, dll_path, source) hit with resolution status.

    Returned by :func:`discover_eps` when ``return_shadowed=True``. The
    ``status`` field distinguishes the precedence-winner ("primary") from
    later sources for the same EP that were skipped under first-hit-wins
    ("shadowed").
    """

    ep_name: str
    dll_path: Path
    source: EpSource
    status: str  # "primary" | "shadowed"


def discover_eps(
    extra_sources: list[EpSource] | None = None,
    *,
    extra_sources_after: list[EpSource] | None = None,
    return_shadowed: bool = False,
) -> dict[str, tuple[Path, EpSource]] | dict[str, list[ResolvedEp]]:
    """Walk ``EP_PATH`` and return resolved EPs.

    Precedence (highest first):

    1. ``extra_sources`` (programmatic override, useful for tests).
    2. ``MODELKIT_EP_PATH`` env-var entries (parsed into FilesystemSources).
    3. The default :data:`EP_PATH` list.
    4. ``extra_sources_after`` (lowest precedence — used by the
       ``winml sys --list-ep`` CLI to inject :func:`list_msix_eps`
       results so non-current MSIX versions appear as ``"shadowed"``
       rather than overriding the user's normal precedence).

    Within that combined list, first-hit-wins per canonical EP name.

    Args:
        extra_sources: Optional list of EpSources prepended to the walk.
        extra_sources_after: Optional list of EpSources appended *after*
            the default :data:`EP_PATH`. Appears as ``"shadowed"`` unless
            the EP is otherwise unresolved.
        return_shadowed: When ``False`` (default, back-compat), returns
            ``dict[ep_name, (dll_path, source)]`` with one entry per EP —
            the precedence winner. When ``True``, returns
            ``dict[ep_name, list[ResolvedEp]]`` with all matching sources;
            the first entry is the ``"primary"``, the rest are
            ``"shadowed"``. Used by the inventory CLI.

    Returns:
        See ``return_shadowed``.
    """
    sources: list[EpSource] = []
    if extra_sources:
        sources.extend(extra_sources)
    sources.extend(_parse_modelkit_ep_path())
    sources.extend(EP_PATH)
    if extra_sources_after:
        sources.extend(extra_sources_after)

    # Always compute the full per-EP list; the legacy shape is derived
    # from it. This keeps the two return shapes consistent and the
    # precedence rules in one place.
    full: dict[str, list[ResolvedEp]] = {}
    for source in sources:
        try:
            it = source.resolve()
        except NotImplementedError as e:
            logger.debug("Skipping not-yet-implemented source %r: %s", source, e)
            continue
        except Exception as e:
            logger.error("Source %r failed to resolve: %s", source, e)
            continue

        try:
            for raw_ep_name, dll_path in it:
                # Normalize alias spellings (e.g. PascalCase
                # ``NvTensorRTRTXExecutionProvider``) to the canonical form
                # so two sources naming the same EP under different aliases
                # collapse to one entry.
                ep_name = canonicalize_ep_name(raw_ep_name)
                if not dll_path.is_file():
                    logger.warning(
                        "EP %s: source %r produced %s which is not a file",
                        ep_name,
                        source,
                        dll_path,
                    )
                    continue
                # Deduplicate (path, source) pairs so a source yielding
                # the same (ep_name, dll) twice doesn't appear twice.
                bucket = full.setdefault(ep_name, [])
                if any(e.dll_path == dll_path and e.source is source for e in bucket):
                    continue
                status = "primary" if not bucket else "shadowed"
                bucket.append(
                    ResolvedEp(
                        ep_name=ep_name,
                        dll_path=dll_path,
                        source=source,
                        status=status,
                    )
                )
                logger.debug(
                    "EP %s [%s] -> %s from %r", ep_name, status, dll_path, source
                )
        except NotImplementedError as e:
            logger.debug("Skipping not-yet-implemented source %r: %s", source, e)
            continue
        except Exception as e:
            logger.error("Source %r failed mid-iteration: %s", source, e)
            continue

    if return_shadowed:
        return full

    # Legacy shape: one (path, source) tuple per EP — the primary winner.
    return {ep: (entries[0].dll_path, entries[0].source) for ep, entries in full.items()}


__all__ = [
    "EP_DLL_NAMES",
    "EP_NAME_ALIASES",
    "EP_PATH",
    "EpSource",
    "FilesystemSource",
    "MsixPackageSource",
    "PyPiSource",
    "ResolvedEp",
    "WinMlCatalogSource",
    "canonicalize_ep_name",
    "discover_eps",
    "list_msix_eps",
]
