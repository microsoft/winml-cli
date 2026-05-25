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
from typing import TYPE_CHECKING, cast


if TYPE_CHECKING:
    from ..utils.constants import EPName


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

        self._ep_paths: dict[EPName, str] = {}
        self._registered_eps: dict[str, list[EPName]] = {
            "onnxruntime": [],
            "onnxruntime_genai": [],
        }
        self._winml_available = False

        self._discover_eps()

    def _discover_eps(self) -> None:
        """Discover execution providers via WinML."""
        try:
            self._load_ep_catalog()
            self._winml_available = True
            logger.debug("WinML EP discovery successful: %s", list(self._ep_paths.keys()))
        except ImportError as e:
            logger.warning("WinML not available (missing packages): %s", e)
            self._winml_available = False
        except Exception as e:
            logger.warning("WinML EP discovery failed: %s", e)
            self._winml_available = False

    def _load_ep_catalog(self) -> None:
        """Load EP catalog from WinML."""
        from windowsml import EpCatalog

        with EpCatalog() as catalog:
            for provider in catalog.find_all_providers():
                try:
                    provider.ensure_ready()
                except Exception as e:
                    logger.debug("Failed to ensure EP %s is ready: %s", provider.name, e)
                    continue
                if provider.library_path == "":
                    continue
                self._ep_paths[cast("EPName", provider.name)] = provider.library_path
                logger.debug("Found EP: %s at %s", provider.name, provider.library_path)

    def register_to_ort(self) -> list[EPName]:
        """Register discovered EPs to ONNX Runtime.

        Returns:
            List of successfully registered EP names.
        """
        if not self._winml_available:
            logger.warning("WinML not available, skipping EP registration")
            return []

        result = self.register_execution_providers(ort=True)
        return result.get("onnxruntime", []).copy()

    def register_execution_providers(
        self, ort: bool = True, ort_genai: bool = False
    ) -> dict[str, list[EPName]]:
        """Register WinML execution providers for ONNX Runtime modules.

        Args:
            ort: Whether to register for ONNX Runtime.
            ort_genai: Whether to register for ONNX Runtime GenAI.

        Returns:
            Dictionary of registered execution provider names by module.
        """
        modules = []
        if ort:
            import onnxruntime

            modules.append(onnxruntime)
        if ort_genai:
            import onnxruntime_genai  # type: ignore[import-not-found]

            modules.append(onnxruntime_genai)
        for name, path in self._ep_paths.items():
            for module in modules:
                if name not in self._registered_eps[module.__name__]:
                    try:
                        module.register_execution_provider_library(name, path)
                        self._registered_eps[module.__name__].append(name)
                        logger.debug(
                            "Registered EP: %s from %s for module %s", name, path, module.__name__
                        )
                    except Exception:
                        logger.exception(
                            "Failed to register %s from %s for module %s",
                            name,
                            path,
                            module.__name__,
                        )
        return self._registered_eps

    def get_ep_library_path(self, ep_name: EPName) -> str | None:
        """Get the library path for an EP."""
        return self._ep_paths.get(ep_name)

    def get_available_eps(self) -> dict[EPName, str]:
        """Get available EPs and their paths."""
        return self._ep_paths.copy()

    def get_registered_eps(self) -> list[EPName]:
        """Get list of EPs registered with ORT."""
        return self._registered_eps["onnxruntime"].copy()

    def is_ep_available(self, ep_name: EPName) -> bool:
        """Check if an EP is available."""
        return ep_name in self._ep_paths

    @property
    def winml_available(self) -> bool:
        """Whether WinML is available."""
        return self._winml_available

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
            logger.debug("WinML discovery skipped: %s", e)

    return ort.get_available_providers()
