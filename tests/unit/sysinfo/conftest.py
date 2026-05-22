# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared fixtures for sysinfo tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_device_caches() -> None:
    """Reset lru_cache on the device-detection helpers before each test.

    ``_get_available_devices`` and ``_get_available_eps`` are cached at
    module level (``@functools.lru_cache(maxsize=1)``) because hardware
    doesn't change during a process lifetime — but tests in this module
    mock the underlying probes (``get_registered_ep_devices``, etc.) and
    need each test to see fresh probe results, not cached output from
    whichever test ran first.
    """
    from winml.modelkit.sysinfo.device import (
        _get_available_devices,
        _get_available_eps,
    )

    _get_available_devices.cache_clear()
    _get_available_eps.cache_clear()
