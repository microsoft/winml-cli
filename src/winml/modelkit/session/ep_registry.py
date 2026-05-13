# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Execution Provider Registry using Windows App SDK.

This module discovers and registers execution providers via the
Windows Machine Learning API (WinML).
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnxruntime as ort

from .ep_device import EPNotDiscovered, EPRegistrationFailed


logger = logging.getLogger(__name__)

# Singleton instance
_winml_ep_registry: WinMLEPRegistry | None = None


class WinMLEPRegistry:
    """Execution Provider Registry using Windows App SDK.

    Discovers EPs via WinML ExecutionProviderCatalog and registers
    them with ONNX Runtime.

    Usage:
        registry = WinMLEPRegistry.get_instance()
        registry.register_to_ort()
        available = registry.get_available_eps()
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
        """Initialize WinML EP registry."""
        if self._initialized:
            return
        self._initialized = True

        self._ep_paths: dict[str, str] = {}
        self._registered_eps: list[str] = []
        self._registration_failures: dict[str, str] = {}
        self._winml_available = False
        self._win_app_sdk_handle = None

        self._discover_eps()

    def _discover_eps(self) -> None:
        """Discover execution providers via WinML."""
        try:
            self._fix_winrt_runtime()
            self._init_windows_app_sdk()
            self._load_ep_catalog()
            self._winml_available = True
            logger.debug("WinML EP discovery successful: %s", list(self._ep_paths.keys()))
        except ImportError as e:
            logger.warning("WinML not available (missing packages): %s", e)
            self._winml_available = False
        except Exception as e:
            # Include exception class so users can distinguish "no providers
            # in catalog" (expected) from "init crashed" (broken env).
            logger.warning("WinML EP discovery failed (%s: %s)", type(e).__name__, e)
            self._winml_available = False

    def _fix_winrt_runtime(self) -> None:
        """Fix msvcp140.dll conflict in winrt-runtime package."""
        try:
            from importlib import metadata

            site_packages_path = Path(str(metadata.distribution("winrt-runtime").locate_file("")))
            dll_path = site_packages_path / "winrt" / "msvcp140.dll"
            if dll_path.exists():
                dll_path.unlink()
                logger.debug("Removed conflicting msvcp140.dll from winrt-runtime")
        except Exception as e:
            # NFR-2: this function only runs in the known-needed init path —
            # a failure here matters. Surface at WARNING with exception class.
            logger.warning("Could not fix winrt-runtime (%s: %s)", type(e).__name__, e)

    def _init_windows_app_sdk(self) -> None:
        """Initialize Windows App SDK."""
        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
            InitializeOptions,
            initialize,
        )

        self._win_app_sdk_handle = initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI)
        self._win_app_sdk_handle.__enter__()

    def _load_ep_catalog(self) -> None:
        """Load EP catalog from WinML."""
        import winui3.microsoft.windows.ai.machinelearning as winml

        catalog = winml.ExecutionProviderCatalog.get_default()
        providers = catalog.find_all_providers()

        for provider in providers:
            provider.ensure_ready_async().get()
            if provider.library_path == "":
                continue
            self._ep_paths[provider.name] = provider.library_path
            logger.debug("Found EP: %s at %s", provider.name, provider.library_path)

    def register_to_ort(self) -> list[str]:
        """Register discovered EPs to ONNX Runtime.

        Returns:
            List of successfully registered EP names.
        """
        if not self._winml_available:
            logger.warning("WinML not available, skipping EP registration")
            return []

        import onnxruntime as ort

        for name, dll_path in self._ep_paths.items():
            if name in self._registered_eps:
                continue

            try:
                # Use ORT's native EP registration API
                ort.register_execution_provider_library(name, dll_path)
                self._registered_eps.append(name)
                # Clear any prior failure record on successful re-register.
                self._registration_failures.pop(name, None)
                logger.debug("Registered EP: %s -> %s", name, dll_path)
            except Exception as e:
                # NFR-2: surface EP name + exception class so users can
                # diagnose which provider failed to register and why.
                msg = f"{type(e).__name__}: {e}"
                self._registration_failures[name] = msg
                logger.warning("Failed to register EP %s (%s)", name, msg)

        return self._registered_eps.copy()

    def register_ep(self, ep_name: str) -> list[ort.OrtEpDevice]:
        """Register a single discovered EP and return its claimed devices.

        Idempotent: if already registered, returns the current device list
        without re-loading the DLL. Callers must pass canonicalize_ep_name(...)
        on user-supplied names first.

        Bundled EPs (e.g. ``CPUExecutionProvider``, ``DmlExecutionProvider``)
        ship with ORT itself rather than as plugin DLLs. They appear in
        ``ort.get_ep_devices()`` without ever needing
        ``register_execution_provider_library``. If the EP is already
        visible, this method short-circuits and returns its devices without
        consulting the catalog.

        Raises:
            EPNotDiscovered:      ep_name absent from both the catalog
                                  *and* ``ort.get_ep_devices()`` (i.e. not
                                  a bundled EP and not a discovered plugin).
            EPRegistrationFailed: ort.register_execution_provider_library
                                  raised (original exception chained).
        """
        # Plugin EP path: catalog knows about it, register from DLL.
        if ep_name in self._ep_paths:
            if ep_name not in self._registered_eps:
                # Defensive: another singleton (e.g. winml.py:WinML) may have
                # already called ort.register_execution_provider_library for
                # this EP in the same process.  ORT's C++ layer is NOT
                # idempotent — a second registration of the same DLL calls
                # exit(127) with no Python traceback.  Check ORT's live device
                # list before attempting the DLL load.
                already_loaded = any(d.ep_name == ep_name for d in ort.get_ep_devices())
                if already_loaded:
                    logger.debug(
                        "EP %s already loaded by another caller; skipping DLL register",
                        ep_name,
                    )
                    self._registered_eps.append(ep_name)
                else:
                    dll_path = self._ep_paths[ep_name]
                    try:
                        ort.register_execution_provider_library(ep_name, dll_path)
                    except Exception as exc:
                        raise EPRegistrationFailed(
                            f"ort.register_execution_provider_library({ep_name!r}, "
                            f"{dll_path!r}) failed: {exc}"
                        ) from exc
                    self._registered_eps.append(ep_name)
            return [d for d in ort.get_ep_devices() if d.ep_name == ep_name]

        # Not in catalog — might be a bundled EP (e.g. CPUExecutionProvider,
        # DmlExecutionProvider) that ships with ORT itself and is visible
        # via get_ep_devices() without ever needing register_execution_provider_library.
        bundled = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
        if bundled:
            return bundled

        raise EPNotDiscovered(
            f"EP {ep_name!r} not in discovered catalog and not visible via "
            f"ort.get_ep_devices(). Catalog: {sorted(self._ep_paths)}. "
            f"Hint: install the plugin or set MODELKIT_EP_PATH."
        )

    def get_ep_library_path(self, ep_name: str) -> str | None:
        """Get the library path for an EP."""
        return self._ep_paths.get(ep_name)

    def get_available_eps(self) -> dict[str, str]:
        """Get available EPs and their paths."""
        return self._ep_paths.copy()

    def get_registered_eps(self) -> list[str]:
        """Get list of EPs registered with ORT."""
        return self._registered_eps.copy()

    def is_ep_available(self, ep_name: str) -> bool:
        """Check if an EP is available."""
        return ep_name in self._ep_paths

    @property
    def winml_available(self) -> bool:
        """Whether WinML is available."""
        return self._winml_available

    @property
    def registration_failures(self) -> dict[str, str]:
        """Per-EP registration failures from the most recent ``register_to_ort()``.

        Maps EP name → ``"<ExcClass>: <message>"`` for any provider that
        failed to register. Empty when all registrations succeeded.
        Successful re-registration clears the corresponding entry.
        """
        return self._registration_failures.copy()

    def __del__(self) -> None:
        """Cleanup Windows App SDK handle."""
        if self._win_app_sdk_handle is not None:
            try:
                self._win_app_sdk_handle.__exit__(None, None, None)
            except Exception as e:
                logger.debug("Error cleaning up Windows App SDK: %s", e)

    @classmethod
    def get_instance(cls) -> WinMLEPRegistry:
        """Get singleton instance."""
        return cls()


def get_ort_available_providers(use_winml: bool = True) -> list[str]:
    """Get available execution providers from ONNX Runtime.

    This function first attempts WinML EP discovery to register any
    WinML-discovered providers, then returns the full list of available
    providers from ORT.

    Note:
        This function is for informational/debugging purposes only.
        WinMLSession uses policy-based device selection (PREFER_NPU, etc.)
        and does NOT use explicit EP provider names.

    Args:
        use_winml: Try WinML EP discovery first to register providers

    Returns:
        List of available provider names from ORT
    """
    import onnxruntime as ort

    # Try WinML discovery first to register any available providers
    if use_winml:
        try:
            registry = WinMLEPRegistry.get_instance()
            registry.register_to_ort()
        except Exception as e:
            # NFR-2: surface real failures at WARNING so users can diagnose.
            logger.warning("WinML discovery skipped (%s: %s)", type(e).__name__, e)

    return ort.get_available_providers()


def ensure_initialized() -> None:
    """Idempotent module-level entry point for WinML EP registration.

    Wraps ``WinMLEPRegistry.get_instance().register_to_ort()`` so callers
    (e.g. ``QNNMonitor.is_available``) can trigger EP registration without
    importing ``WinMLSession`` — breaks a latent import cycle.

    Safe to call multiple times. No-op if WinML is unavailable on this system.

    Failures during registration are logged at WARNING (NFR-2: must not be
    silent) and swallowed so callers can probe availability without raising.
    Subsequent calls retry — there is no module-level latch on failure.
    """
    try:
        registry = WinMLEPRegistry.get_instance()
        if registry.winml_available:
            registry.register_to_ort()
    except Exception as exc:
        # NFR-2: surface real environmental failures at WARNING with the
        # exception class so users can distinguish "not on Windows" from
        # "registration crashed".
        logger.warning(
            "ensure_initialized: WinML EP registration failed (%s: %s)",
            type(exc).__name__,
            exc,
        )
