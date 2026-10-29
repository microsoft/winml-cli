# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""DXGI memory observations retain counter errors and byte-to-MiB semantics."""
from unittest.mock import Mock

from winml.modelkit.session.monitor.memory_tracker import ProcessMemoryTracker


def test_dxgi_observation_units_and_missing_value():
    tracker = ProcessMemoryTracker(backend='dxgi')
    reader = Mock()
    reader.sample.return_value = {
        'local': {'value_bytes': 1048576, 'hresult': '0x00000000'},
        'nonlocal': {'value_bytes': None, 'hresult': '0x80004005'},
    }
    tracker._dxgi_reader = reader
    values, errors = tracker._observe()
    assert values['vram_local'] == 1
    assert values['vram_shared'] is None
    assert errors['vram_shared'] == '0x80004005'
    tracker.close()
    reader.close.assert_called_once()


def test_phase_summary_preserves_private_commit_and_errors():
    tracker = ProcessMemoryTracker(backend='dxgi')
    tracker._phase_samples = {'steady_inference': {'private_commit': [2.0, 4.0, 3.0]}}
    tracker._phase_errors = {'steady_inference': 1}
    report = tracker.sampled()
    assert report['phase_statistics']['steady_inference']['private_commit'] == {
        'median_mib': 3.0, 'peak_mib': 4.0, 'samples': 3}
    assert report['phase_query_errors']['steady_inference'] == 1
