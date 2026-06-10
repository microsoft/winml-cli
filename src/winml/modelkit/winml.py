# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""DEPRECATED legacy entry points for plugin-EP bulk registration.

Every public symbol in this module is deprecated. Use the
:mod:`winml.modelkit.session` machinery instead — see
``docs/design/session/2_coreloop.md`` for the canonical Path A / Path B
flows.

Migration:

- ``WinML().register_execution_providers(ort=True)`` and the module-level
  ``register_execution_providers(...)`` →
  ``WinMLEPRegistry.instance().register_ep(entry)`` per discovered
  :class:`EPEntry`, or iterate :func:`discover_all_eps` for bulk.
- ``add_ep_for_device(sess_options, ep_name, device_type)`` →
  build :class:`EPDeviceTarget` → :func:`resolve_device` →
  :meth:`WinMLEPRegistry.auto_device` → call
  ``sess_options.add_provider_for_devices([ep_device.device._ort], options)``
  inline.

This module exists only to support in-tree ``analyze/*`` callers that
have not yet migrated. New code MUST NOT depend on these symbols.
"""

from __future__ import annotations

import logging
import sys
import warnings
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

from .ep_path import EPSource, discover_all_eps


logger = logging.getLogger(__name__)


_DEPRECATION_MSG = (
    "winml.modelkit.winml is deprecated; use winml.modelkit.session "
    "(WinMLEPRegistry, EPDeviceTarget, resolve_device) instead. "
    "See docs/design/session/2_coreloop.md."
)


_winml_instance: WinML | None = None


class WinML:
    """DEPRECATED singleton for bulk plugin-EP registration.

    Use :class:`winml.modelkit.session.WinMLEPRegistry` and its
    :meth:`register_ep` / :meth:`auto_device` methods instead.
    """

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
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)

        # Walk the default EP source list (plus WINMLCLI_EP_PATH env var
        # entries, if any) and capture (ep_name -> abs path) for the
        # primary entry per EP.
        self._resolved: dict[str, tuple[Path, EPSource]] = {
            e.ep_name: (e.dll_path, e.source)
            for e in discover_all_eps()
            if e.status == "primary"
        }
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
        """DEPRECATED. Returns ``{module_name: [registered_ep_names...]}``.

        Use :meth:`WinMLEPRegistry.register_ep` per :class:`EPEntry`, or
        loop ``discover_all_eps()`` for bulk.
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
                # this EP in the same process. Consult the live ORT device list first.
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
    """DEPRECATED. Thin wrapper that constructs the :class:`WinML` singleton.

    Use :meth:`WinMLEPRegistry.register_ep` per :class:`EPEntry`, or
    loop ``discover_all_eps()`` for bulk.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    return WinML().register_execution_providers(
        ort=ort, ort_genai=ort_genai, extra_sources=extra_sources
    )


def add_ep_for_device(
    session_options: Any,
    ep_name: str,
    device_type: Any,
    ep_options: dict | None = None,
) -> None:
    """DEPRECATED. Bind one (EP, device) ``OrtEpDevice`` to a ``SessionOptions``.

    Use the typed session-layer path instead::

        target = EPDeviceTarget(ep=short_ep_name(ep_name), device=device_type.name.lower())
        resolved = resolve_device(target)
        ep_device = WinMLEPRegistry.instance().auto_device(resolved)
        session_options.add_provider_for_devices(
            [ep_device.device._ort], ep_options or {},
        )

    See ``docs/design/session/2_coreloop.md`` for the full Path A flow.

    Args:
        session_options: An existing :class:`ort.SessionOptions` to mutate.
        ep_name: Canonical EP name as ORT registers it (full form, e.g.
            ``"QNNExecutionProvider"`` — no alias normalization).
        device_type: An :class:`ort.OrtHardwareDeviceType` enum value.
        ep_options: Optional per-EP provider options dict; ``None`` is
            treated as ``{}``.

    Silently no-ops when no ``OrtEpDevice`` matches the
    ``(ep_name, device_type)`` pair (loud failure semantics live in the
    new session-layer path via :meth:`WinMLEPRegistry.auto_device`).
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    import onnxruntime as ort

    # Exact-match by ORT's canonical EP name. Callers must pass the
    # spelling ORT registers under — no alias normalization layer.
    ep_devices = ort.get_ep_devices()
    for ep_device in ep_devices:
        if ep_device.ep_name == ep_name and ep_device.device.type == device_type:
            print(f"Adding {ep_name} for {device_type}")
            session_options.add_provider_for_devices(
                [ep_device], {} if ep_options is None else ep_options
            )
            break


__all__ = [
    "WinML",
    "add_ep_for_device",
    "register_execution_providers",
]
