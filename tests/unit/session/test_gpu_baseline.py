# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Baseline device lifetime and unknown counters do not contaminate deltas."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from winml.modelkit.session.monitor import _gpu_baseline as native
from winml.modelkit.session.monitor import memory_tracker as mt


def observation(value=None, absent=False):
    return {
        k: {
            **mt.metric(value, "test", "absent" if absent else None),
            **({"discovery_status": "absent_unconfirmed"} if absent else {}),
        }
        for k in ("local", "shared")
    }


def test_device_created_before_snapshot_and_retained_until_close(monkeypatch):
    active = []
    device = Mock()

    def create(luid):
        assert luid == "selected"
        active.append(True)
        return device

    monkeypatch.setattr(native, "GpuBaselineDevice", create)
    monkeypatch.setattr(
        mt,
        "sample_vram",
        lambda _: observation(10 * mt.MIB) if active else observation(absent=True),
    )
    tracker = mt.MemoryTracker("selected", baseline="legacy", device="gpu")
    tracker.capture("before_model_load")
    assert tracker.checkpoints["before_model_load"]["vram_local"]["value_bytes"] == 10 * mt.MIB
    assert tracker.evidence()["gpu_baseline_preparation"]["status"] == "initialized"
    device.close.assert_not_called()
    monkeypatch.setattr(mt, "sample_vram", lambda _: observation(30 * mt.MIB))
    for phase in ("baseline", "after_compile", "after_inference"):
        tracker.capture(phase)
    tracker.capture("before_model_load")
    assert tracker.profile()["vram_local_total_from_before_model_load_delta_mb"] == 20
    tracker.close()
    tracker.close()
    device.close.assert_called_once()


def test_existing_valid_zero_is_not_initialized(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(native, "GpuBaselineDevice", factory)
    monkeypatch.setattr(mt, "sample_vram", lambda _: observation(0))
    tracker = mt.MemoryTracker("selected", baseline="legacy", device="gpu")
    tracker.capture("before_model_load")
    factory.assert_not_called()
    assert tracker.checkpoints["before_model_load"]["vram_local"]["value_origin"] == "measured_zero"


def test_failed_initialization_keeps_unknown_baseline_and_inference_can_proceed(monkeypatch):
    monkeypatch.setattr(native, "GpuBaselineDevice", Mock(side_effect=OSError("unsupported")))
    monkeypatch.setattr(mt, "sample_vram", lambda _: observation(absent=True))
    tracker = mt.MemoryTracker("selected", baseline="legacy", device="gpu")
    tracker.capture("before_model_load")
    assert tracker.evidence()["gpu_baseline_preparation"]["status"] == "failed"
    monkeypatch.setattr(mt, "sample_vram", lambda _: observation(100))
    for phase in ("baseline", "after_compile", "after_inference"):
        tracker.capture(phase)
    assert tracker.profile()["vram_local_total_from_before_model_load_delta_mb"] is None
    tracker.close()


@pytest.mark.parametrize(
    "device,phase",
    [("cpu", "before_model_load"), ("npu", "before_model_load"), ("gpu", "baseline")],
)
def test_other_targets_and_preloaded_phase_do_not_create_device(monkeypatch, device, phase):
    factory = Mock()
    monkeypatch.setattr(native, "GpuBaselineDevice", factory)
    monkeypatch.setattr(mt, "sample_vram", lambda _: observation(absent=True))
    tracker = mt.MemoryTracker("selected", baseline="legacy", device=device)
    tracker.capture(phase)
    factory.assert_not_called()


def test_preparation_timeout_retains_null_without_backfill(monkeypatch):
    device = Mock()
    monkeypatch.setattr(native, "GpuBaselineDevice", lambda _: device)
    monkeypatch.setattr(mt, "sample_vram", lambda _: observation(absent=True))
    clock = iter([0, 3])
    monkeypatch.setattr(mt.time, "monotonic", lambda: next(clock))
    tracker = mt.MemoryTracker("selected", baseline="legacy", device="gpu")
    tracker.capture("before_model_load")
    assert tracker.evidence()["gpu_baseline_preparation"]["status"] == "counter_unavailable"
    assert tracker.checkpoints["before_model_load"]["vram_local"]["value_bytes"] is None
    tracker.close()
    device.close.assert_called_once()


@pytest.mark.parametrize("failure", [False, True])
def test_perf_releases_baseline_on_success_or_failure(monkeypatch, failure):
    from winml.modelkit.commands.perf import BenchmarkConfig, PerfBenchmark

    bench = PerfBenchmark(BenchmarkConfig(model_id="unused"))
    tracker = Mock()
    bench._memory_tracker = tracker

    def run():
        tracker.close.assert_not_called()
        if failure:
            raise RuntimeError("model failed")
        return "result"

    monkeypatch.setattr(bench, "_run_with_memory", run)
    if failure:
        with pytest.raises(RuntimeError, match="model failed"):
            bench.run()
    else:
        assert bench.run() == "result"
    tracker.close.assert_called_once()


def test_luid_rejected_before_native_calls():
    with pytest.raises(ValueError, match="LUID"):
        native.GpuBaselineDevice("adapter-zero")


@pytest.mark.parametrize("failure", [None, "factory", "adapter", "device"])
def test_native_device_releases_acquired_interfaces(monkeypatch, failure):
    released = []

    def acquire(stage, value, output):
        if failure == stage:
            return -2147467259
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = value
        return 0

    factory = Mock(side_effect=lambda iid, out: acquire("factory", 11, out))
    create = Mock(side_effect=lambda adapter, level, iid, out: acquire("device", 33, out))
    monkeypatch.setattr(
        native.ctypes,
        "WinDLL",
        lambda name: (
            SimpleNamespace(CreateDXGIFactory1=factory)
            if name == "dxgi"
            else SimpleNamespace(D3D12CreateDevice=create)
        ),
    )

    def method(pointer, index, result, *args):
        if index == 2:
            return lambda value: released.append(value.value)
        assert pointer.value == 11 and index == 26

        def adapter_by_luid(factory, luid, iid, output):
            assert luid.low == 2 and luid.high == -1
            return acquire("adapter", 22, output)

        return adapter_by_luid

    monkeypatch.setattr(native, "_method", method)
    if failure:
        with pytest.raises(OSError, match="HRESULT"):
            native.GpuBaselineDevice("0xffffffff_0x00000002")
        assert released == {"factory": [], "adapter": [11], "device": [22, 11]}[failure]
    else:
        device = native.GpuBaselineDevice("0xffffffff_0x00000002")
        assert released == [22, 11]
        device.close()
        device.close()
        assert released == [22, 11, 33]


@pytest.mark.parametrize("failure", [False, True])
def test_process_tracker_prepares_gpu_and_releases_after_model(monkeypatch, failure):
    from winml.modelkit.commands.perf import BenchmarkConfig, PerfBenchmark

    device = Mock()
    active = []

    def create(luid):
        active.append(luid)
        return device

    monkeypatch.setattr(native, "GpuBaselineDevice", create)
    monkeypatch.setattr(
        mt,
        "sample_vram",
        lambda _: observation(10 * mt.MIB) if active else observation(absent=True),
    )
    monkeypatch.setattr(mt.ProcessMemoryTracker, "_sample", lambda _: None)
    monkeypatch.setattr(
        mt,
        "_device_memory",
        lambda _: (
            {"vram_local": 10.0 if active else None, "vram_shared": 10.0 if active else None},
            {},
        ),
    )
    bench = PerfBenchmark(BenchmarkConfig(model_id="unused", memory=True))

    def load():
        tracker = bench._memory_tracker
        tracker.bind_before_load("selected", device="gpu")
        assert active == ["selected"]
        assert tracker.metadata()["gpu_baseline_preparation"]["status"] == "initialized"
        assert tracker.checkpoints["baseline"]["vram_local"] == 10
        device.close.assert_not_called()
        if failure:
            raise RuntimeError("model failed")
        bench._model = Mock()

    monkeypatch.setattr(bench, "_load_model", load)
    monkeypatch.setattr(bench, "_run_single", lambda: "result")
    if failure:
        with pytest.raises(RuntimeError, match="model failed"):
            bench.run()
    else:
        assert bench.run() == "result"
    device.close.assert_called_once()


@pytest.fixture(autouse=True)
def windows_measurement_gate(monkeypatch):
    """Mock the platform gate without changing process-wide sys.platform."""
    monkeypatch.setattr(mt, "sys", SimpleNamespace(platform="win32"))
