# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Auto-reset behavior: session.perf(monitor=...) with options on already-compiled session."""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock

import onnxruntime as ort

from tests._helpers import get_minimal_onnx_model_path


def _get_real_cpu_ort_device():
    """Return the CPUExecutionProvider OrtEpDevice from ort.get_ep_devices()."""
    import pytest

    devs = [d for d in ort.get_ep_devices() if d.ep_name == "CPUExecutionProvider"]
    if not devs:
        pytest.skip("CPUExecutionProvider not available in ort.get_ep_devices()")
    return devs[0]


def _make_cpu_session(model_path):
    """Create a WinMLSession bound to CPU with a stub WinMLEPDevice."""
    from winml.modelkit.session.session import WinMLSession

    from .conftest import make_stub_winml_ep_device

    cpu_dev = _get_real_cpu_ort_device()
    cpu_ep_device = make_stub_winml_ep_device(cpu_dev, "CPUExecutionProvider")
    return WinMLSession(model_path, ep_device=cpu_ep_device), cpu_dev, cpu_ep_device


def test_auto_reset_fires_when_options_contributed(caplog):
    """If session is already compiled AND monitor contributes provider_options,
    session.perf().__enter__ auto-resets with verbose-only diagnostic logging.
    """
    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor
    from winml.modelkit.session.session import WinMLSession

    from .conftest import make_stub_winml_ep_device

    class _ContributingMonitor(WinMLEPMonitor):
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

    cpu_dev = _get_real_cpu_ort_device()
    cpu_ep_device = make_stub_winml_ep_device(cpu_dev, "CPUExecutionProvider")

    session = WinMLSession(get_minimal_onnx_model_path(), ep_device=cpu_ep_device)

    session.compile()
    assert session._session is not None
    pre_session = session._session

    with caplog.at_level(logging.INFO), session.perf(monitor=_ContributingMonitor()):
        pass

    # NFR-3: the verbatim phrase MUST appear as a substring of the log.
    expected = "auto-resetting compiled session to apply monitor session/provider options"
    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any(expected in m for m in info_messages), (
        f"NFR-3 verbatim phrase not in INFO records. expected substring: "
        f"{expected!r}; got: {info_messages}"
    )
    assert not any(expected in r.message for r in caplog.records if r.levelno >= logging.WARNING)
    # Old session object was dropped
    assert session._session is None or session._session is not pre_session


def test_eager_session_creation_suppresses_native_stdout(monkeypatch, capfd):
    """Native QNN compiler stdout must not leak during eager InferenceSession creation."""
    from winml.modelkit.session import session as session_module
    from winml.modelkit.session.session import WinMLSession

    from .conftest import make_stub_winml_ep_device

    class _FakeOrtSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    def _fake_inference_session(*_args, **_kwargs):
        os.write(1, b"native compiler progress should be hidden\n")
        return _FakeOrtSession()

    cpu_dev = _get_real_cpu_ort_device()
    cpu_ep_device = make_stub_winml_ep_device(cpu_dev, "CPUExecutionProvider")

    monkeypatch.setattr(session_module.ort, "InferenceSession", _fake_inference_session)

    WinMLSession(get_minimal_onnx_model_path(), ep_device=cpu_ep_device)

    assert "native compiler progress" not in capfd.readouterr().out


def test_native_stdout_suppression_refreshes_click_handle(monkeypatch):
    """Restoring fd 1 must repair Click's cached Windows console handle."""
    from winml.modelkit.session import session as session_module

    calls = []
    monkeypatch.setattr(session_module, "_refresh_click_windows_console_stream", calls.append)

    with session_module._suppress_native_output():
        pass

    assert calls == [1]


def test_auto_reset_suppresses_native_stdout(monkeypatch, capfd):
    """Monitor-triggered session rebuild/restore also suppresses native stdout."""
    from winml.modelkit.session import session as session_module
    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor
    from winml.modelkit.session.session import WinMLSession

    from .conftest import make_stub_winml_ep_device

    class _ContributingMonitor(WinMLEPMonitor):
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

    def _fake_inference_session(*_args, **_kwargs):
        os.write(1, b"native compiler progress should be hidden\n")
        return MagicMock()

    cpu_dev = _get_real_cpu_ort_device()
    cpu_ep_device = make_stub_winml_ep_device(cpu_dev, "CPUExecutionProvider")
    session = WinMLSession(get_minimal_onnx_model_path(), ep_device=cpu_ep_device)
    session.compile()

    monkeypatch.setattr(session_module.ort, "InferenceSession", _fake_inference_session)

    with session.perf(monitor=_ContributingMonitor()):
        pass

    assert "native compiler progress" not in capfd.readouterr().out


def test_auto_reset_suppresses_unclassified_native_stderr(monkeypatch, capfd):
    """Monitor-triggered session rebuild hides native diagnostics without severity tokens."""
    from winml.modelkit.session import session as session_module
    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor
    from winml.modelkit.session.session import WinMLSession

    from .conftest import make_stub_winml_ep_device

    class _ContributingMonitor(WinMLEPMonitor):
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

    def _fake_inference_session(*_args, **_kwargs):
        os.write(2, b"DSP_INFO UNSUPPORTED_KEY: 49\n")
        os.write(2, b"2026 [E:custom-native:, file.cc:2 ErrorFunc] useful error\n")
        return MagicMock()

    cpu_dev = _get_real_cpu_ort_device()
    cpu_ep_device = make_stub_winml_ep_device(cpu_dev, "CPUExecutionProvider")
    session = WinMLSession(get_minimal_onnx_model_path(), ep_device=cpu_ep_device)
    session.compile()

    monkeypatch.setattr(session_module.ort, "InferenceSession", _fake_inference_session)

    with session.perf(monitor=_ContributingMonitor()):
        pass

    stderr = capfd.readouterr().err
    assert "DSP_INFO" not in stderr
    assert "useful error" in stderr


def test_no_auto_reset_when_monitor_empty():
    """If monitor contributes NO options, no reset occurs."""
    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

    session, _cpu_dev, _cpu_ep = _make_cpu_session(get_minimal_onnx_model_path())

    session.compile()
    pre_session = session._session
    assert pre_session is not None

    with session.perf(monitor=NullEPMonitor()):
        pass

    # Session should NOT have been reset
    assert session._session is pre_session
