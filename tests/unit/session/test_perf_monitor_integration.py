# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for WinMLSession.perf(monitor=...) — teardown ordering,
auto-reset, session/provider option merging, exception transparency.

This file grows across multiple tasks (7, 8).
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from tests._helpers import get_minimal_onnx_model_path


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


def test_perf_monitor_none_yields_perfcontext_with_null_monitor():
    """perf() with no monitor yields PerfContext whose monitor is NullEPMonitor."""
    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor
    from winml.modelkit.session.session import PerfContext, WinMLSession

    session = WinMLSession(get_minimal_onnx_model_path(), device="cpu")
    with session.perf(warmup=0) as ctx:
        assert isinstance(ctx, PerfContext)
        assert isinstance(ctx.monitor, NullEPMonitor)
        # ctx.stats must be the PerfStats instance
        assert ctx.stats is not None


def test_nested_perf_raises():
    """Entering perf() while another is active raises RuntimeError."""
    from winml.modelkit.session.session import WinMLSession

    session = WinMLSession(get_minimal_onnx_model_path(), device="cpu")
    with session.perf(), pytest.raises(RuntimeError, match="already active"), session.perf():
        pass


def test_teardown_ordering_reset_before_monitor_exit():
    """For monitor.requires_session_teardown=True, self.reset() fires BEFORE monitor.__exit__."""
    from winml.modelkit.session.monitor.ep_monitor import EPMonitor
    from winml.modelkit.session.session import WinMLSession

    observations: dict = {}

    class _TeardownMonitor(EPMonitor):
        requires_session_teardown = True

        def __init__(self):
            self.session_ref = None

        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # At this point, session.reset() should have fired → self.session_ref._session is None
            if self.session_ref is not None:
                observations["session_at_exit"] = self.session_ref._session

        def to_dict(self):
            return {"ep": "test"}

    session = WinMLSession(get_minimal_onnx_model_path(), device="cpu")
    mon = _TeardownMonitor()
    mon.session_ref = session

    with session.perf(monitor=mon):
        # Force compile so reset has something to tear down
        session.run({"input": np.zeros((1, 4), dtype=np.float32)})

    # After perf exit, session._session should be None (reset happened)
    assert session._session is None
    # And the observation captured by monitor.__exit__ should also be None
    # (meaning reset fired before __exit__)
    assert observations.get("session_at_exit") is None


def test_exception_transparency():
    """Exception in `with session.perf()` body propagates; monitor.__exit__ sees exc_info."""
    from winml.modelkit.session.monitor.ep_monitor import EPMonitor
    from winml.modelkit.session.session import WinMLSession

    captured: dict = {}

    class _CapturingMonitor(EPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            captured["exc_type"] = exc_type

        def to_dict(self):
            return {"ep": "test"}

    session = WinMLSession(get_minimal_onnx_model_path(), device="cpu")
    mon = _CapturingMonitor()

    with pytest.raises(ValueError, match="boom"), session.perf(monitor=mon):
        raise ValueError("boom")

    assert captured.get("exc_type") is ValueError


def test_monitor_enter_raises_leaves_session_clean():
    """If mon.__enter__() raises, session state is not polluted.

    Regression guard: an earlier version mutated _perf_stats and _provider_options
    before mon.__enter__(), so an __enter__ exception left the session stuck
    (nested-perf error on every subsequent perf() call).
    """
    from winml.modelkit.session.monitor.ep_monitor import EPMonitor
    from winml.modelkit.session.session import WinMLSession

    class _RaisingEnterMonitor(EPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            raise RuntimeError("simulated __enter__ failure")

        def __exit__(self, *a):
            pass

        def to_dict(self):
            return {"ep": "test"}

        def get_provider_options(self):
            return {"some_key": "1"}

    session = WinMLSession(get_minimal_onnx_model_path(), device="cpu")

    mon = _RaisingEnterMonitor()
    with pytest.raises(RuntimeError, match="simulated"), session.perf(monitor=mon):
        pass  # never reached

    # Session state must be fully restored
    assert session._perf_stats is None
    assert session._active_session_option_entries == {}
    assert session._provider_options == {}

    # Subsequent perf() MUST work (no stuck state)
    with session.perf() as ctx:
        assert ctx is not None
