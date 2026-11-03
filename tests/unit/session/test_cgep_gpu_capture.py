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


@pytest.mark.parametrize(('target', 'flag'), [('dml', '1'), ('auto', '0')])
def test_cgep_capture_session_options(tmp_path, target, flag):
    capture = GpuCapture(tmp_path)
    capture.execution_path = 'winmlcgep'
    capture.lowering_target = target
    token = active_capture.set(capture)
    options = MagicMock()
    device = SimpleNamespace(device=SimpleNamespace(ep_name='WinMLCGExecutionProvider',
                             _ort=object(), device_type='gpu'), ep=SimpleNamespace(arg0='winmlcg'))
    try:
        with patch('winml.modelkit.session.session._build_provider_options', return_value={}):
            _build_session_options(device, session_options_factory=lambda: options)
        assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.add_session_config_entry.assert_any_call('session.disable_cpu_ep_fallback', '1')
        options.add_session_config_entry.assert_any_call(
            'ep.winmlcgexecutionprovider.compile_target_flags', flag)
        options.add_provider_for_devices.assert_called_once()
    finally:
        active_capture.reset(token)
