# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for WinMLEPMonitor ABC default hook behavior."""

from __future__ import annotations

import pytest

from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor, WinMLEPMonitor


def test_null_monitor_default_get_session_options():
    """NullEPMonitor inherits empty session-options default."""
    assert NullEPMonitor().get_session_options() == {}


def test_null_monitor_default_get_provider_options():
    """NullEPMonitor inherits empty provider-options default."""
    assert NullEPMonitor().get_provider_options() == {}


def test_null_monitor_default_requires_teardown():
    """NullEPMonitor.requires_session_teardown is False by default."""
    assert NullEPMonitor.requires_session_teardown is False


def test_ep_monitor_is_abstract():
    """WinMLEPMonitor cannot be instantiated directly (still abstract)."""
    with pytest.raises(TypeError):
        WinMLEPMonitor()  # type: ignore[abstract]


def test_hooks_return_fresh_dicts():
    """get_*_options returns a fresh dict each call (not a shared mutable)."""
    m = NullEPMonitor()
    d1 = m.get_session_options()
    d1["injected"] = "1"
    d2 = m.get_session_options()
    assert "injected" not in d2


def test_requires_session_teardown_must_be_bool() -> None:
    """Shadowing requires_session_teardown with a non-bool fails at class-def time."""
    with pytest.raises(TypeError, match="requires_session_teardown must be a class-level bool"):

        class _BadMonitor(WinMLEPMonitor):
            requires_session_teardown = "yes"  # wrong type

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return None

            def to_dict(self):
                return {}

            @classmethod
            def is_available(cls):
                return True
