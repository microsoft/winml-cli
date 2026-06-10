# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Plugin-style ONNX Runtime execution provider registration.

Discovers EP plugin DLLs via the unified :mod:`winml.modelkit.ep_path`
discovery layer and registers them with ONNX Runtime via
``register_execution_provider_library()`` (added in ORT 1.24). Built-in
EPs (CPU, DML in 1.24+) are registered automatically by ORT and are not
listed here.

The legacy ``EP_PLUGIN_REGISTRY`` dict and ``resolve_plugin_dll()``
function are kept as backwards-compatibility shims that delegate to the
new ``ep_path`` module. They will be removed once no internal callers
remain (see ``docs/ep-path-design.md`` migration plan, step 5).
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

from .ep_path import (
    EP_CATALOG,
    DirectorySource,
    EPCatalog,
    EPEntry,
    EPSource,
    MSIXPackageSource,
    NuGetSource,
    PyPISource,
    WinMLCatalogSource,
    _default_ep_sources,
    discover_all_eps,
    list_msix_eps,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backwards-compat shims.
# ---------------------------------------------------------------------------
# The two PyPI-only entries we historically advertised. Kept as a frozen
# view of the default EP source list's ``PyPISource`` rows so callers that
# still iterate this dict (none in-tree at time of writing) keep working.
# New code should use ``_default_ep_sources()`` and ``discover_all_eps()`` instead.
def _legacy_ep_plugin_registry() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for source in _default_ep_sources():
        if not isinstance(source, PyPISource):
            continue
        rel = source.relative_dll
        if source.arch_resolver is not None:
            rel = source.arch_resolver(rel)
        for ep_name in source.eps:
            out.setdefault(ep_name, (source.distribution, rel))
    return out


EP_PLUGIN_REGISTRY: dict[str, tuple[str, str]] = _legacy_ep_plugin_registry()


def resolve_plugin_dll(ep_name: str) -> Path | None:
    """Resolve the absolute DLL path for a plugin EP, or None if unavailable.

    Backwards-compat shim. Walks the default EP source list via
    :func:`discover_all_eps` and returns the first absolute path resolved
    for ``ep_name`` (the primary entry), or ``None``.
    """
    for entry in discover_all_eps():
        if entry.ep_name == ep_name and entry.status == "primary":
            return entry.dll_path if entry.dll_path.exists() else None
    return None


# ---------------------------------------------------------------------------
# WinML singleton.
# ---------------------------------------------------------------------------

_winml_instance: WinML | None = None


class WinML:
    """Singleton class for managing WinML execution providers."""

    _initialized: bool

    def __new__(cls, *args: Any, **kwargs: Any) -> WinML:
        """Create or return the singleton instance."""
        global _winml_instance
        if _winml_instance is None:
            _winml_instance = super().__new__(cls, *args, **kwargs)
            _winml_instance._initialized = False
        return _winml_instance

    def __init__(self) -> None:
        """Initialize WinML execution provider catalog from the default EP source list."""
        if self._initialized:
            return
        self._initialized = True

        # Walk the default EP source list (plus WINMLCLI_EP_PATH env var
        # entries, if any) and capture (ep_name -> abs path) for the
        # primary entry per EP.
        self._resolved: dict[str, tuple[Path, EPSource]] = {
            e.ep_name: (e.dll_path, e.source)
            for e in discover_all_eps()
            if e.status == "primary"
        }
        # Preserve the legacy attribute name and shape so any external
        # caller that introspects the singleton sees the same dict it
        # used to. Only the abs path is exposed (not the source).
        self._ep_paths: dict[str, str] = {
            name: str(path) for name, (path, _) in self._resolved.items()
        }

        self._registered_eps: dict[str, list[str]] = {
            "onnxruntime": [],
            "onnxruntime_genai": [],
        }

    def register_execution_providers(
        self,
        ort: bool = True,
        ort_genai: bool = False,
        extra_sources: list[EPSource] | None = None,
    ) -> dict[str, list[str]]:
        """Register WinML execution providers for ONNX Runtime modules.

        Args:
            ort: Whether to register for ONNX Runtime.
            ort_genai: Whether to register for ONNX Runtime GenAI.
            extra_sources: Optional list of additional ``EPSource`` entries
                to consult before the default EP source list. Useful for
                tests and embedded apps. Has highest precedence.

        Returns:
            Dictionary of registered execution provider names by module.
        """
        # When extra_sources are supplied, refresh the resolved set so
        # the override takes precedence. Otherwise reuse the cached set
        # captured at __init__ to preserve singleton semantics.
        if extra_sources:
            resolved = {
                e.ep_name: (e.dll_path, e.source)
                for e in discover_all_eps(extra_sources=extra_sources)
                if e.status == "primary"
            }
            ep_paths = {name: str(path) for name, (path, _) in resolved.items()}
        else:
            ep_paths = self._ep_paths

        modules = []
        if ort:
            import onnxruntime

            modules.append(onnxruntime)
        if ort_genai:
            import onnxruntime_genai  # type: ignore[import-not-found]

            modules.append(onnxruntime_genai)
        # When extra_sources is supplied the caller is explicitly asking
        # for the override path to win — bypass the per-process registered
        # EP-name cache so a second call with new extra_sources isn't
        # silently no-op'd by the first call's registrations. ORT's
        # register_execution_provider_library is idempotent for the same
        # (name, path) pair and returns the existing handle; re-calling
        # with a different path replaces the registration, which is what
        # extra_sources callers want.
        skip_cache = extra_sources is not None
        for name, path in ep_paths.items():
            for module in modules:
                if not skip_cache and name in self._registered_eps[module.__name__]:
                    continue
                # Defensive guard: ORT's register_execution_provider_library is NOT
                # idempotent — a second call for the same DLL calls C++ exit(127) with
                # no Python traceback (surfaces as STATUS_DLL_NOT_FOUND / 0xC000026F).
                # WinMLEPRegistry (session/ep_registry.py) may have already registered
                # this EP in the same process.  Consult the live ORT device list first.
                try:
                    already_loaded = any(d.ep_name == name for d in module.get_ep_devices())
                except Exception:
                    already_loaded = False  # conservative: attempt the load
                if already_loaded:
                    if name not in self._registered_eps[module.__name__]:
                        self._registered_eps[module.__name__].append(name)
                    continue
                try:
                    module.register_execution_provider_library(name, path)
                    if name not in self._registered_eps[module.__name__]:
                        self._registered_eps[module.__name__].append(name)
                except Exception as e:
                    print(
                        f"Failed to register execution provider {name}: {e}",
                        file=sys.stderr,
                    )
        return self._registered_eps


def register_execution_providers(
    ort: bool = True,
    ort_genai: bool = False,
    extra_sources: list[EPSource] | None = None,
) -> dict[str, list[str]]:
    """Register WinML execution providers for ONNX Runtime and ORT GenAI.

    Args:
        ort (bool): Whether to register for ONNX Runtime.
        ort_genai (bool): Whether to register for ONNX Runtime GenAI.
        extra_sources: Optional list of additional ``EPSource`` entries
            with highest precedence. Defaults to ``None`` (no extras).

    Returns:
        dict[str, list[str]]: Dictionary of registered execution provider
        names by module.
    """
    return WinML().register_execution_providers(
        ort=ort, ort_genai=ort_genai, extra_sources=extra_sources
    )


def add_ep_for_device(
    session_options: Any,
    ep_name: str,
    device_type: Any,
    ep_options: dict | None = None,
) -> None:
    """Ensures correct EP device selection for WinML. NEVER modify this function.

    ep_name is one of:
        - "CPUExecutionProvider"
        - "DmlExecutionProvider"
        - "WebGpuExecutionProvider"
        - "QNNExecutionProvider"
        - "OpenVINOExecutionProvider"
        - "VitisAIExecutionProvider"
        - "NvTensorRtRtxExecutionProvider"

    device_type is one of:
        - ort.OrtHardwareDeviceType.CPU
        - ort.OrtHardwareDeviceType.GPU
        - ort.OrtHardwareDeviceType.NPU
    """
    import onnxruntime as ort

    # Exact-match by ORT's canonical EP name. Callers must pass the
    # spelling ORT registers under (e.g. ``NvTensorRtRtxExecutionProvider``,
    # camelCase) — no alias normalization layer.
    ep_devices = ort.get_ep_devices()
    for ep_device in ep_devices:
        if ep_device.ep_name == ep_name and ep_device.device.type == device_type:
            print(f"Adding {ep_name} for {device_type}")
            session_options.add_provider_for_devices(
                [ep_device], {} if ep_options is None else ep_options
            )
            break


__all__ = [
    "EP_CATALOG",
    "EP_PLUGIN_REGISTRY",
    "DirectorySource",
    "EPCatalog",
    "EPEntry",
    "EPSource",
    "MSIXPackageSource",
    "NuGetSource",
    "PyPISource",
    "WinML",
    "WinMLCatalogSource",
    "add_ep_for_device",
    "discover_all_eps",
    "list_msix_eps",
    "register_execution_providers",
    "resolve_plugin_dll",
]
