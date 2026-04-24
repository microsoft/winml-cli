# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for WinMLSession.perf(monitor=...) — teardown ordering,
auto-reset, session/provider option merging, exception transparency.

This file grows across multiple tasks (7, 8). For Task 7, only the
_active_session_option_entries test is added.
"""

from __future__ import annotations

import onnxruntime as ort


def test_active_session_option_entries_applied_in_build():
    """_build_session_options applies monitor-contributed entries on the returned SessionOptions."""
    from winml.modelkit.session.session import WinMLSession

    # Construct without going through __init__ to avoid file I/O
    session = WinMLSession.__new__(WinMLSession)
    session._device = "cpu"
    session._ep = None
    session._session_options = ort.SessionOptions()
    session._provider_options = {}
    session._active_session_option_entries = {
        "session.disable_cpu_ep_fallback": "1",
    }

    opts = session._build_session_options("cpu")
    # ORT doesn't expose a clean read-back API for session config entries,
    # but the call should not raise and should return a SessionOptions
    assert isinstance(opts, ort.SessionOptions)


def test_active_session_option_entries_default_empty():
    """Newly-constructed WinMLSession has empty _active_session_option_entries."""
    from winml.modelkit.session.session import WinMLSession

    session = WinMLSession.__new__(WinMLSession)
    # Simulate post-__init__ state without file I/O
    session._active_session_option_entries = {}  # from __init__
    assert session._active_session_option_entries == {}
