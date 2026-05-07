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

from ..ep_path import EpSource, discover_eps


logger = logging.getLogger(__name__)

# Singleton instance
_winml_ep_registry: WinMLEPRegistry | None = None


class WinMLEPRegistry:
    """Execution Provider Registry for plugin-style ONNX Runtime EPs.

    Discovers plugin EPs via :func:`winml.modelkit.ep_path.discover_eps`
    (which walks the ``EP_PATH`` list and the ``MODELKIT_EP_PATH`` env-var
    override) and registers them with ONNX Runtime.

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
        """Discover plugin EPs from EP_PATH."""
        if self._initialized:
            return
        self._initialized = True

        self._ep_paths: dict[str, str] = {}
        self._ep_sources: dict[str, EpSource] = {}
        self._registered_eps: list[str] = []

        for ep_name, (path, source) in discover_eps().items():
            self._ep_paths[ep_name] = str(path)
            self._ep_sources[ep_name] = source
            logger.debug("Found EP: %s at %s (from %r)", ep_name, path, source)

    def register_to_ort(self) -> list[str]:
        """Register discovered EPs with ONNX Runtime.

        Returns:
            List of successfully registered EP names.
        """
        if not self._ep_paths:
            logger.debug("No plugin EPs found, skipping registration")
            return []

        import onnxruntime as ort

        for name, dll_path in self._ep_paths.items():
            if name in self._registered_eps:
                continue

            try:
                ort.register_execution_provider_library(name, dll_path)
                self._registered_eps.append(name)
                logger.debug("Registered EP: %s -> %s", name, dll_path)
            except Exception as e:
                logger.warning("Failed to register EP %s: %s", name, e)

        return self._registered_eps.copy()

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
        """Whether any plugin EP package is installed and resolvable."""
        return bool(self._ep_paths)

    @classmethod
    def get_instance(cls) -> WinMLEPRegistry:
        """Get singleton instance."""
        return cls()


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
            registry = WinMLEPRegistry.get_instance()
            registry.register_to_ort()
        except Exception as e:
            logger.debug("Plugin EP discovery skipped: %s", e)

    return ort.get_available_providers()
