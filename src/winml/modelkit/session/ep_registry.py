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
import os
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from ..utils.constants import EPName


logger = logging.getLogger(__name__)


def _ep_download_timeout_default() -> int:
    """Read ``WINMLCLI_EP_DOWNLOAD_TIMEOUT`` (seconds) or fall back to 5 minutes.

    Lets users on slow networks raise the cap without code changes. Falls back
    to the default when the env var is unset, empty, or non-integer.
    """
    raw = os.environ.get("WINMLCLI_EP_DOWNLOAD_TIMEOUT")
    if not raw:
        return 5 * 60
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid WINMLCLI_EP_DOWNLOAD_TIMEOUT=%r; using default 300s.", raw)
        return 5 * 60


EP_DOWNLOAD_TIMEOUT_SECONDS = _ep_download_timeout_default()


class _NoopBar:
    """No-op stand-in for tqdm when the optional dependency is missing.

    Exposes the attribute (``n``) and methods (``refresh``, ``close``) that
    ``_ensure_provider_ready`` touches, so the helper can stay branch-free.
    """

    def __init__(self) -> None:
        self.n = 0

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        return None


def _make_progress_bar() -> Any:
    """Return a tqdm bar if tqdm is installed, else a silent no-op stand-in.

    tqdm is a dev-only optional dep in this package, so production installs
    without it must still complete EP downloads — they just lose the live bar.
    The pre-download Console notice is emitted by the caller and is unaffected.

    Format: ``Downloading... ████████████░░░░░░ 62%``
    """
    try:
        from tqdm import tqdm
    except ImportError:
        return _NoopBar()
    return tqdm(
        total=100,
        bar_format="Downloading... {bar} {percentage:3.0f}%",
        ascii="░█",
        leave=True,
    )


def _ensure_provider_ready(provider: Any) -> None:
    """Ensure an EP is ready, showing a tqdm progress bar when downloading.

    Providers already in the ``Ready`` state take the synchronous fast path so
    cached EPs do not flash a 0-100% bar. Otherwise drives a tqdm bar from
    ``ensure_ready_async``'s ``on_progress`` callback (cumulative fraction
    0.0-1.0, per windowsml docs) and waits for the ``on_complete`` callback
    via a threading.Event with a ``EP_DOWNLOAD_TIMEOUT_SECONDS`` timeout. On
    timeout the async op is cancelled and ``TimeoutError`` is raised.
    """
    import threading

    from windowsml import EpReadyState

    if provider.ready_state == EpReadyState.Ready:
        provider.ensure_ready()
        return

    # Lazy-import to keep ep_registry import cheap (rich pulls in pygments etc.);
    # this branch only runs on the cold "EP needs download" path.
    from ..utils.console import get_console

    console = get_console()
    console.print(
        f"[yellow][WinML] Installing Execution Provider: [bold]{provider.name}[/bold].[/yellow]"
    )

    bar = _make_progress_bar()
    done = threading.Event()

    def _on_progress(fraction: float) -> None:
        bar.n = max(0, min(100, int(fraction * 100)))
        bar.refresh()

    op = None
    success = False
    try:
        op = provider.ensure_ready_async(on_complete=done.set, on_progress=_on_progress)
        if not done.wait(timeout=EP_DOWNLOAD_TIMEOUT_SECONDS):
            op.cancel()
            raise TimeoutError(
                f"EP {provider.name!r} download did not complete within "
                f"{EP_DOWNLOAD_TIMEOUT_SECONDS}s; cancelled."
            )
        # Surface any native failure (raises OSError on error).
        op.get_status()
        # Success: providers usually fire on_progress(1.0) before on_complete,
        # but force the bar to 100 in case they didn't.
        bar.n = 100
        bar.refresh()
        success = True
    finally:
        bar.close()
        if op is not None:
            op.close()
        if not success:
            # Failure-path notice — kept in finally so it fires for every
            # non-success exit (launch failure, timeout, get_status OSError).
            # Printed after bar.close() so it appears below the bar's last frame.
            console.print(f"[red]❌ Failed to download {provider.name} EP[/red]")
            console.print("Try:")
            console.print("  1. Check your internet connection")
            console.print(
                "  2. Troubleshoot: "
                "https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/execution-provider-errors",
                soft_wrap=True,
            )

    console.print(f"[green]{provider.name} EP installed successfully.[/green]")
    console.print(f"- Version: {provider.version}")
    console.print(f"- Package Family Name: {provider.package_family_name}")


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
                    _ensure_provider_ready(provider)
                except OSError as e:
                    # windowsml maps native HRESULT failures to OSError; surface
                    # winerror so the HRESULT is grep-able in logs.
                    logger.info(
                        "Failed to ensure EP %s is ready: %s (winerror=%s)",
                        provider.name,
                        e,
                        getattr(e, "winerror", None),
                    )
                    continue
                except Exception as e:
                    logger.info("Failed to ensure EP %s is ready: %s", provider.name, e)
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
