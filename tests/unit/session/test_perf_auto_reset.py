# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Auto-reset behavior: session.perf(monitor=...) with options on already-compiled session."""

from __future__ import annotations

import logging
from unittest.mock import patch

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
    """Create a WinMLSession bound to CPU with mocked WinMLEPRegistry."""
    from winml.modelkit.session import WinMLEPDevice
    from winml.modelkit.session.session import WinMLSession

    cpu_dev = _get_real_cpu_ort_device()
    cpu_ep_device = WinMLEPDevice(
        ep="CPUExecutionProvider",
        device="cpu",
        vendor_id=cpu_dev.device.vendor_id,
        device_id=cpu_dev.device.device_id,
    )
    with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = [cpu_dev]
        return WinMLSession(model_path, ep_device=cpu_ep_device), cpu_dev, cpu_ep_device


def test_auto_reset_fires_when_options_contributed(caplog):
    """If session is already compiled AND monitor contributes provider_options,
    session.perf().__enter__ auto-resets with a WARNING log.

    The WinMLEPRegistry mock must be active during perf() as well as construction,
    because the auto-reset + rebuild path calls _build_session_options().
    """
    from winml.modelkit.session import WinMLEPDevice
    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor
    from winml.modelkit.session.session import WinMLSession

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
    cpu_ep_device = WinMLEPDevice(
        ep="CPUExecutionProvider",
        device="cpu",
        vendor_id=cpu_dev.device.vendor_id,
        device_id=cpu_dev.device.device_id,
    )

    # Registry mock must stay active across both __init__ and perf() rebuild.
    with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = [cpu_dev]
        session = WinMLSession(get_minimal_onnx_model_path(), ep_device=cpu_ep_device)

        session.compile()
        assert session._session is not None
        pre_session = session._session

        with caplog.at_level(logging.WARNING), session.perf(monitor=_ContributingMonitor()):
            pass

    # NFR-3: the verbatim phrase MUST appear as a substring of the log.
    expected = "auto-resetting compiled session to apply monitor session/provider options"
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any(expected in m for m in warnings), (
        f"NFR-3 verbatim phrase not in WARNING records. expected substring: "
        f"{expected!r}; got: {warnings}"
    )
    # Old session object was dropped
    assert session._session is None or session._session is not pre_session


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
