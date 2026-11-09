# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import onnxruntime as ort
import pytest

from winml.modelkit.session.gpu_capture import GpuCapture, active_capture
from winml.modelkit.session.session import _build_session_options


@pytest.mark.parametrize("allow_fallback", [False, True])
@pytest.mark.parametrize(
    ("optimization", "threads"),
    [
        (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 0),
        (ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, 4),
        (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 2),
    ],
)
def test_cgep_capture_session_options(
    tmp_path, allow_fallback, optimization, threads
):
    capture = GpuCapture(tmp_path)
    capture.execution_path = "winmlcgep"
    capture.allow_cpu_fallback = allow_fallback
    token = active_capture.set(capture)
    options = MagicMock()
    options.graph_optimization_level = optimization
    options.intra_op_num_threads = threads
    device = SimpleNamespace(
        device=SimpleNamespace(
            ep_name="WinMLCGExecutionProvider", _ort=object(), device_type="gpu"
        ),
        ep=SimpleNamespace(arg0="winmlcg"),
    )
    try:
        with patch("winml.modelkit.session.session._build_provider_options", return_value={}):
            _build_session_options(device, session_options_factory=lambda: options)
        assert options.graph_optimization_level == optimization
        assert options.intra_op_num_threads == threads
        options.add_session_config_entry.assert_any_call(
            "session.disable_cpu_ep_fallback", "0" if allow_fallback else "1"
        )
        options.add_provider_for_devices.assert_called_once()
    finally:
        active_capture.reset(token)


@pytest.mark.parametrize("with_monitor", [False, True])
def test_capture_preserves_ordinary_perf_options(tmp_path, with_monitor):
    defaults = ort.SessionOptions()
    observed = []
    device = SimpleNamespace(
        device=SimpleNamespace(
            ep_name="WinMLCGExecutionProvider", _ort=object(), device_type="gpu"
        ),
        ep=SimpleNamespace(arg0="winmlcg"),
    )
    monitor = MagicMock() if with_monitor else None
    if monitor is not None:
        monitor.get_session_options.return_value = {}

        def configure(options):
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            options.intra_op_num_threads = 3

        monitor.configure_session_options.side_effect = configure

    for traced in (False, True):
        options = MagicMock()
        options.graph_optimization_level = defaults.graph_optimization_level
        options.intra_op_num_threads = defaults.intra_op_num_threads
        capture = GpuCapture(tmp_path / str(traced))
        capture.execution_path = "winmlcgep"
        token = active_capture.set(capture if traced else None)
        try:
            with (
                patch("winml.modelkit.session.session.ort.SessionOptions", return_value=options),
                patch("winml.modelkit.session.session._build_provider_options", return_value={}),
            ):
                _build_session_options(device, ep_monitor=monitor)
            observed.append((options.graph_optimization_level, options.intra_op_num_threads))
        finally:
            active_capture.reset(token)

    expected = (
        (ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, 3)
        if with_monitor
        else (defaults.graph_optimization_level, defaults.intra_op_num_threads)
    )
    assert observed == [expected, expected]
