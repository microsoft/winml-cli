# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for _build_session_options / _build_provider_options."""

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.session import AmbiguousMatch, DeviceNotFound, EPDevice
from winml.modelkit.session.session import (
    _build_provider_options,
    _build_session_options,
    _ep_defaults,
)


@pytest.fixture
def qnn_npu() -> EPDevice:
    return EPDevice(
        ep="QNNExecutionProvider",
        device="npu",
        vendor_id=0x4D4F,
        device_id=0x0001,
        vendor="Qualcomm",
    )


@pytest.fixture
def cpu_ep() -> EPDevice:
    return EPDevice(
        ep="CPUExecutionProvider",
        device="cpu",
        vendor_id=0x8086,
        device_id=0x0000,
    )


def _stub_monitor(prov: dict[str, str], sess: dict[str, str] | None = None) -> MagicMock:
    m = MagicMock()
    m.get_provider_options.return_value = prov
    m.get_session_options.return_value = sess or {}
    return m


def _ort_dev(name: str, vid: int, did: int) -> MagicMock:
    d = MagicMock()
    d.device.type.name = name
    d.device.vendor_id = vid
    d.device.device_id = did
    return d


def test_build_provider_options_qnn_defaults_only(qnn_npu: EPDevice) -> None:
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


def test_build_provider_options_user_overrides_defaults(qnn_npu: EPDevice) -> None:
    """ep_config.provider_options overrides EP defaults."""
    ep_config = MagicMock()
    ep_config.provider_options = {"backend_type": "gpu", "custom_key": "custom_val"}
    opts = _build_provider_options(qnn_npu, ep_config=ep_config, ep_monitor=None)
    assert opts["backend_type"] == "gpu"
    assert opts["custom_key"] == "custom_val"


def test_build_provider_options_monitor_overrides_user(qnn_npu: EPDevice) -> None:
    """Monitor wins last — tracing correctness invariant."""
    ep_config = MagicMock()
    ep_config.provider_options = {"profiling_level": "off"}
    monitor = _stub_monitor({"profiling_level": "detailed", "profiling_file_path": "/traces/x"})
    opts = _build_provider_options(qnn_npu, ep_config=ep_config, ep_monitor=monitor)
    assert opts["profiling_level"] == "detailed"
    assert opts["profiling_file_path"] == "/traces/x"
    # backend_type is NOT injected by _ep_defaults — OrtEpDevice handle encodes the backend.


def test_ep_defaults_unknown_ep_returns_empty(cpu_ep: EPDevice) -> None:
    """_ep_defaults returns {} for any EP that doesn't need a backend hint."""
    assert _ep_defaults(cpu_ep) == {}


def test_build_session_options_no_monitor_qnn_npu(qnn_npu: EPDevice) -> None:
    """qnn+npu with no monitor: returns SessionOptions bound to the matching OrtEpDevice.

    Burst-mode defaults from the EPDeviceSpec catalog are passed as provider_options.
    """
    chosen = _ort_dev("NPU", 0x4D4F, 0x0001)
    sibling = _ort_dev("GPU", 0x4D4F, 0x0002)
    fake_so = MagicMock()
    with (
        patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg,
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=fake_so),
    ):
        mock_reg.get_instance.return_value.register_ep.return_value = [chosen, sibling]
        result = _build_session_options(qnn_npu, ep_config=None, ep_monitor=None)
    assert result is fake_so
    fake_so.add_provider_for_devices.assert_called_once_with(
        [chosen],
        {
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        },
    )
    fake_so.add_session_config_entry.assert_not_called()


def test_build_session_options_monitor_plumbs_session_options(qnn_npu: EPDevice) -> None:
    """Monitor's get_session_options() entries land via add_session_config_entry."""
    chosen = _ort_dev("NPU", 0x4D4F, 0x0001)
    monitor = _stub_monitor(
        prov={"profiling_level": "detailed"},
        sess={"session.disable_cpu_ep_fallback": "1"},
    )
    fake_so = MagicMock()
    with (
        patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg,
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=fake_so),
    ):
        mock_reg.get_instance.return_value.register_ep.return_value = [chosen]
        _build_session_options(qnn_npu, ep_config=None, ep_monitor=monitor)
    fake_so.add_session_config_entry.assert_called_once_with("session.disable_cpu_ep_fallback", "1")
    fake_so.add_provider_for_devices.assert_called_once()
    args, _ = fake_so.add_provider_for_devices.call_args
    assert args[1]["profiling_level"] == "detailed"


def test_build_session_options_device_not_found_raises(qnn_npu: EPDevice) -> None:
    """Registry returns only a GPU — npu request raises DeviceNotFound."""
    only_gpu = _ort_dev("GPU", 0x4D4F, 0x0002)
    with (
        patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg,
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        mock_reg.get_instance.return_value.register_ep.return_value = [only_gpu]
        with pytest.raises(DeviceNotFound):
            _build_session_options(qnn_npu, ep_config=None, ep_monitor=None)


def test_build_session_options_ambiguous_match_raises(qnn_npu: EPDevice) -> None:
    """Two registry entries with identical IDs trigger AmbiguousMatch (registry bug signal)."""
    a = _ort_dev("NPU", 0x4D4F, 0x0001)
    b = _ort_dev("NPU", 0x4D4F, 0x0001)
    with (
        patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg,
        patch("winml.modelkit.session.session.ort.SessionOptions", return_value=MagicMock()),
    ):
        mock_reg.get_instance.return_value.register_ep.return_value = [a, b]
        with pytest.raises(AmbiguousMatch):
            _build_session_options(qnn_npu, ep_config=None, ep_monitor=None)


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


def test_build_session_options_repeated_calls_do_not_accumulate(qnn_npu: EPDevice) -> None:
    """Repeatedly calling _build_session_options with the same base (MagicMock) and
    different monitors writes the correct session-config entries each time.

    Uses a MagicMock as the base so add_provider_for_devices is not type-checked
    against real OrtEpDevice.  The ORT overwrite semantics are pinned in the
    companion test test_ort_session_options_same_key_overwrites.
    """
    chosen = _ort_dev("NPU", 0x4D4F, 0x0001)
    monitor_a = _stub_monitor(prov={}, sess={"session.disable_cpu_ep_fallback": "1"})
    monitor_b = _stub_monitor(prov={}, sess={"session.disable_cpu_ep_fallback": "0"})
    fake_so = MagicMock()

    with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg:
        mock_reg.get_instance.return_value.register_ep.return_value = [chosen]
        _build_session_options(qnn_npu, None, monitor_a, fake_so)
        _build_session_options(qnn_npu, None, monitor_b, fake_so)

    # add_session_config_entry should have been called exactly twice.
    assert fake_so.add_session_config_entry.call_count == 2
    calls = fake_so.add_session_config_entry.call_args_list
    assert calls[0] == (("session.disable_cpu_ep_fallback", "1"),)
    assert calls[1] == (("session.disable_cpu_ep_fallback", "0"),)
