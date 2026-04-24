# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Auto-reset behavior: session.perf(monitor=...) with options on already-compiled session."""

from __future__ import annotations

import logging

from tests._helpers import get_minimal_onnx_model_path


def test_auto_reset_fires_when_options_contributed(caplog):
    """If session is already compiled AND monitor contributes provider_options,
    session.perf().__enter__ auto-resets with a WARNING log."""
    from winml.modelkit.session.monitor.ep_monitor import EPMonitor
    from winml.modelkit.session.session import WinMLSession

    class _ContributingMonitor(EPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def to_dict(self):
            return {"ep": "test"}

        def get_provider_options(self):
            return {"some_key": "1"}

    session = WinMLSession(get_minimal_onnx_model_path(), device="cpu")
    session.compile()
    assert session._session is not None
    pre_session = session._session

    with caplog.at_level(logging.WARNING), session.perf(monitor=_ContributingMonitor()):
        pass

    # The warning message must mention auto-reset
    messages = [r.message.lower() for r in caplog.records]
    assert any("auto-reset" in m for m in messages), f"no auto-reset log. records={messages}"
    # Old session object was dropped
    assert session._session is None or session._session is not pre_session


def test_no_auto_reset_when_monitor_empty():
    """If monitor contributes NO options, no reset occurs."""
    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor
    from winml.modelkit.session.session import WinMLSession

    session = WinMLSession(get_minimal_onnx_model_path(), device="cpu")
    session.compile()
    pre_session = session._session
    assert pre_session is not None

    with session.perf(monitor=NullEPMonitor()):
        pass

    # Session should NOT have been reset
    assert session._session is pre_session
