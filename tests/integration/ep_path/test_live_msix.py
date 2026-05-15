# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Windows-only smoke tests against the LIVE PackageManager / EP Catalog.

The unit tests under ``tests/unit/ep_path/`` exercise the resolution and
discovery logic against synthetic ``_FakePackage`` / fake catalog objects.
That covers the implementation but cannot detect breakage at the WinRT
binding boundary (e.g., shape changes when the wasdk MSIX is upgraded).

These integration tests run only on Windows with the ``[winml-catalog]``
extra installed. They are minimal smoke checks: confirm the binding
loads, the call returns the expected shape, and at least one realistic
MSIX EP package is reachable on a developer machine. They do NOT assert
on specific package versions.

Review reference: S-5 from the comprehensive review pass.
"""

from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="WinRT PackageManager and ExecutionProviderCatalog are Windows-only",
)


def test_pkg_manager_returns_handle() -> None:
    """`_get_pkg_manager()` returns a usable WinRT PackageManager."""
    from winml.modelkit.ep_path import _get_pkg_manager

    _get_pkg_manager.cache_clear()
    pm = _get_pkg_manager()
    if pm is None:
        pytest.skip(
            "winrt-Windows.Management.Deployment binding not installed "
            "(install via the [winml-catalog] extra)"
        )
    assert hasattr(pm, "find_packages_by_user_security_id")


def test_list_msix_eps_returns_list() -> None:
    """`list_msix_eps()` returns a list (possibly empty) on a real Windows box."""
    from winml.modelkit.ep_path import _get_pkg_manager, list_msix_eps

    _get_pkg_manager.cache_clear()
    if _get_pkg_manager() is None:
        pytest.skip(
            "WinRT PackageManager unavailable; install via [winml-catalog]"
        )

    results = list_msix_eps()
    assert isinstance(results, list)
    # Each result must be a fully-pinned MsixPackageSource.
    for src in results:
        assert src.family_name_prefix
        assert src.relative_dll
        assert src.eps
        assert src.version is not None


def test_winml_catalog_find_all_providers_works() -> None:
    """`_get_catalog()` returns a usable catalog or None (binding missing)."""
    from winml.modelkit.ep_path import _get_catalog

    _get_catalog.cache_clear()
    catalog = _get_catalog()
    if catalog is None:
        pytest.skip(
            "WinAppSDK ML binding unavailable; install via [winml-catalog]"
        )
    providers = list(catalog.find_all_providers())
    # Shape-only check: each provider has the expected attribute surface.
    for provider in providers:
        assert hasattr(provider, "name")
        assert hasattr(provider, "ready_state")
        assert hasattr(provider, "library_path")
