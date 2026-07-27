# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for _build_session_options / _build_provider_options.

Post-Batch-C: these helpers take a fully-resolved :class:`WinMLEPDevice`
rather than a :class:`EPDeviceTarget`. No internal registry call, no
handle filtering — the caller pre-selected the device.
"""

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.session import WinMLEPDevice
from winml.modelkit.session.session import (
    _build_provider_options,
    _build_session_options,
    _ep_defaults,
)

from .conftest import QNN_VENDOR_ID, make_stub_winml_ep_device


def _ort_dev(ep_name: str, dev_type: str, vid: int, did: int) -> MagicMock:
    """Mock OrtEpDevice matching the attributes WinMLDevice/_build_* read."""
    d = MagicMock()
    d.ep_name = ep_name
    d.device.type.name = dev_type
    d.device.vendor_id = vid
    d.device.device_id = did
    return d


@pytest.fixture
def qnn_npu() -> WinMLEPDevice:
    ort_dev = _ort_dev("QNNExecutionProvider", "NPU", QNN_VENDOR_ID, 0x0001)
    return make_stub_winml_ep_device(ort_dev, "QNNExecutionProvider")


@pytest.fixture
def cpu_ep() -> WinMLEPDevice:
    ort_dev = _ort_dev("CPUExecutionProvider", "CPU", 0x8086, 0x0000)
    return make_stub_winml_ep_device(ort_dev, "CPUExecutionProvider")


def _stub_monitor(prov: dict[str, str], sess: dict[str, str] | None = None) -> MagicMock:
    m = MagicMock()
    m.get_provider_options.return_value = prov
    m.get_session_options.return_value = sess or {}
    return m


def test_build_provider_options_qnn_defaults_only(qnn_npu: WinMLEPDevice) -> None:
    """No config, no monitor -> burst-mode defaults from EPDeviceSpec catalog.

    QNNExecutionProvider does not need ``backend_type`` when using
    add_provider_for_devices() — the OrtEpDevice handle already encodes the
    backend target (NPU->HTP). Passing backend_type crashes ORT 1.23.5.

    The QNN-NPU catalog entry ships with htp_performance_mode='burst' and
    htp_graph_finalization_optimization_mode='3' (verified 2026-05-13:
    +3x throughput on ResNet-50 vs default mode).
    """
    opts = _build_provider_options(qnn_npu, ep_config=None, ep_monitor=None)
    assert opts == {
        "htp_performance_mode": "burst",
        "htp_graph_finalization_optimization_mode": "3",
    }


def test_build_provider_options_user_overrides_defaults(qnn_npu: WinMLEPDevice) -> None:
    """ep_config.provider_options overrides EP defaults."""
    ep_config = MagicMock()
    ep_config.provider_options = {"backend_type": "gpu", "custom_key": "custom_val"}
    opts = _build_provider_options(qnn_npu, ep_config=ep_config, ep_monitor=None)
    assert opts["backend_type"] == "gpu"
    assert opts["custom_key"] == "custom_val"


def test_build_provider_options_monitor_overrides_user(qnn_npu: WinMLEPDevice) -> None:
    """Monitor wins last — tracing correctness invariant."""
    ep_config = MagicMock()
    ep_config.provider_options = {"profiling_level": "off"}
    monitor = _stub_monitor({"profiling_level": "detailed", "profiling_file_path": "/traces/x"})
    opts = _build_provider_options(qnn_npu, ep_config=ep_config, ep_monitor=monitor)
    assert opts["profiling_level"] == "detailed"
    assert opts["profiling_file_path"] == "/traces/x"


def test_ep_defaults_unknown_ep_returns_empty(cpu_ep: WinMLEPDevice) -> None:
    """_ep_defaults returns {} for any EP that doesn't need a backend hint."""
    assert _ep_defaults(cpu_ep) == {}


def test_build_session_options_no_monitor_qnn_npu(qnn_npu: WinMLEPDevice) -> None:
    """qnn+npu with no monitor: SessionOptions bound to the device's OrtEpDevice.

    Burst-mode defaults from the EPDeviceSpec catalog are passed as provider_options.
    """
    fake_so = MagicMock()
    with patch("winml.modelkit.session.session.ort.SessionOptions", return_value=fake_so):
        result = _build_session_options(qnn_npu, ep_config=None, ep_monitor=None)
    assert result is fake_so
    fake_so.add_provider_for_devices.assert_called_once_with(
        [qnn_npu.device._ort],
        {
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        },
    )
    fake_so.add_session_config_entry.assert_not_called()


def test_build_session_options_monitor_plumbs_session_options(qnn_npu: WinMLEPDevice) -> None:
    """Monitor's get_session_options() entries land via add_session_config_entry."""
    monitor = _stub_monitor(
        prov={"profiling_level": "detailed"},
        sess={"session.disable_cpu_ep_fallback": "1"},
    )
    fake_so = MagicMock()
    with patch("winml.modelkit.session.session.ort.SessionOptions", return_value=fake_so):
        _build_session_options(qnn_npu, ep_config=None, ep_monitor=monitor)
    fake_so.add_session_config_entry.assert_called_once_with("session.disable_cpu_ep_fallback", "1")
    fake_so.add_provider_for_devices.assert_called_once()
    args, _ = fake_so.add_provider_for_devices.call_args
    assert args[1]["profiling_level"] == "detailed"


def test_ort_session_options_same_key_overwrites() -> None:
    """ORT's add_session_config_entry overwrites on repeated same-key calls.

    Pins the semantic that _build_session_options relies on: when called twice
    with the same base_session_options, the second monitor's session-config entries
    overwrite the first rather than accumulating silently.  If ORT ever changes to
    raise or append, this assertion catches the regression immediately.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    # Second write to the same key must NOT raise and must overwrite.
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
    assert so.get_session_config_entry("session.disable_cpu_ep_fallback") == "0", (
        "ORT add_session_config_entry must overwrite same-key entries. "
        "If this fails, ORT semantics have changed and _build_session_options "
        "needs a defensive copy to prevent monitor entries from accumulating."
    )


def test_build_session_options_repeated_calls_do_not_accumulate(qnn_npu: WinMLEPDevice) -> None:
    """A fresh factory output receives each monitor's session-config entries."""
    monitor_a = _stub_monitor(prov={}, sess={"session.disable_cpu_ep_fallback": "1"})
    monitor_b = _stub_monitor(prov={}, sess={"session.disable_cpu_ep_fallback": "0"})
    first = MagicMock()
    second = MagicMock()
    factory = MagicMock(side_effect=[first, second])

    _build_session_options(qnn_npu, None, monitor_a, factory)
    _build_session_options(qnn_npu, None, monitor_b, factory)

    first.add_session_config_entry.assert_called_once_with("session.disable_cpu_ep_fallback", "1")
    second.add_session_config_entry.assert_called_once_with("session.disable_cpu_ep_fallback", "0")


def test_session_options_factory_creates_fresh_bound_options(qnn_npu: WinMLEPDevice) -> None:
    """Each ordinary session build gets a fresh caller-configured options object."""
    first = MagicMock()
    first.intra_op_num_threads = 4
    second = MagicMock()
    second.intra_op_num_threads = 4
    factory = MagicMock(side_effect=[first, second])

    _build_session_options(qnn_npu, session_options_factory=factory)
    _build_session_options(qnn_npu, session_options_factory=factory)

    assert factory.call_count == 2
    assert first.intra_op_num_threads == 4
    assert second.intra_op_num_threads == 4
    first.add_provider_for_devices.assert_called_once()
    second.add_provider_for_devices.assert_called_once()
