# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Lifecycle and missing-counter regression tests without native EP dependencies."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from winml.modelkit.commands.perf import BenchmarkConfig, BenchmarkResult, PerfBenchmark
from winml.modelkit.session.monitor import ProcessMemoryTracker, get_vram_mb
from winml.modelkit.session.monitor import memory_tracker as memory


def test_missing_gpu_counter_is_null_and_query_always_closes(monkeypatch):
    query = MagicMock()
    query.collect.return_value = {"local_0": 0, "shared_0": None}
    monkeypatch.setattr(
        "winml.modelkit.session.monitor._pdh.memory_instances", lambda *args: ["node"]
    )
    monkeypatch.setattr(memory.sys, "platform", "win32")
    monkeypatch.setattr("winml.modelkit.session.monitor._pdh.PdhQuery", lambda: query)
    values, errors = memory._device_memory("test-luid")
    assert values == {"vram_local": 0.0, "vram_shared": None}
    assert errors == {"vram_shared": "counter_data_unavailable"}
    query.collect.assert_called_once_with(timeout=0.25, interval=0.05)
    query.close.assert_called_once()
    query.reset_mock()
    query.collect.side_effect = OSError("unavailable")
    assert get_vram_mb("test-luid") == (None, None)
    query.close.assert_called_once()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), None])
def test_invalid_gpu_readings_are_not_zero(monkeypatch, value):
    query = MagicMock()
    query.collect.return_value = {"local_0": value, "shared_0": 1048576}
    monkeypatch.setattr(
        "winml.modelkit.session.monitor._pdh.memory_instances", lambda *args: ["node"]
    )
    monkeypatch.setattr(memory.sys, "platform", "win32")
    monkeypatch.setattr("winml.modelkit.session.monitor._pdh.PdhQuery", lambda: query)
    assert get_vram_mb("test-luid") == (None, 1.0)


def test_unknown_adapter_does_not_query_an_arbitrary_gpu(monkeypatch):
    monkeypatch.setattr(memory.sys, "platform", "win32")
    values, reasons = memory._device_memory(None)
    assert set(values.values()) == {None}
    assert set(reasons.values()) == {"adapter_unresolved"}


def test_snapshot_deltas_preserve_negative_and_missing_values(monkeypatch):
    tracker = ProcessMemoryTracker()
    observations = iter(
        [
            ({"rss": 100.0, "vram_local": None, "vram_shared": 0.0}, {"vram_local": "missing"}),
            ({"rss": 160.0, "vram_local": 12.0, "vram_shared": 8.0}, {}),
            ({"rss": 80.0, "vram_local": 10.0, "vram_shared": 7.0}, {}),
        ]
    )
    monkeypatch.setattr(tracker, "_observe", lambda: next(observations))
    for phase in ("baseline", "after_compile", "after_inference"):
        tracker.checkpoint(phase)
    data = tracker.to_dict()
    assert data["rss_total_delta_mb"] == -20
    assert data["rss_checkpoint_peak_mb"] == 160
    assert data["vram_local_total_delta_mb"] is None
    assert data["vram_local_inference_delta_mb"] == -2
    assert data["vram_local_checkpoint_peak_mb"] is None
    assert data["vram_shared_total_delta_mb"] == 7
    assert tracker.metadata()["missing_reasons"]["baseline"] == {"vram_local": "missing"}
    json.dumps(data, allow_nan=False)


def test_sampler_tracks_temporary_peak_separately(monkeypatch):
    tracker = ProcessMemoryTracker(interval=0.001)
    observed = iter([12.0, 64.0, 15.0])

    def sample():
        value = next(observed)
        if value == 15:
            tracker._stop.set()
        return {"rss": value, "vram_local": None, "vram_shared": 0.0}, {}

    monkeypatch.setattr(tracker, "_observe", sample)
    tracker._sample()
    tracker.stop()
    result = tracker.sampled()
    assert result["rss_peak_mb"] == 64
    assert result["rss_samples"] == 3
    assert result["local_peak_mb"] is None and result["local_samples"] == 0
    assert result["shared_peak_mb"] == 0 and result["shared_samples"] == 3


def test_sampling_reports_observed_cadence_not_configured_delay(monkeypatch):
    tracker = ProcessMemoryTracker(interval=0.05)
    clock = iter([10.0, 10.3, 10.8])
    monkeypatch.setattr(memory.time, "monotonic", lambda: next(clock))
    calls = []

    def observe():
        calls.append(True)
        if len(calls) == 3:
            tracker._stop.set()
        return {"rss": 100.0}, {}

    monkeypatch.setattr(tracker, "_observe", observe)
    tracker._sample()
    tracker._ended = tracker._started + 1
    result = tracker.sampled()
    assert result["configured_poll_delay_sec"] == 0.05
    assert result["observed_mean_interval_sec"] == pytest.approx(0.4)
    assert result["observed_max_interval_sec"] == pytest.approx(0.5)
    assert "sampling_interval_sec" not in result


def test_sampling_without_two_observations_has_no_cadence():
    result = ProcessMemoryTracker().sampled()
    assert result["observed_mean_interval_sec"] is None
    assert result["observed_max_interval_sec"] is None


@pytest.mark.parametrize(
    "runtime,artifact",
    [
        ("winml-ort", "generated.onnx"),
        ("winml-runtime", "generated.onnx"),
        ("winml-runtime", "generated.mlir"),
    ],
)
def test_baseline_precedes_eager_loading_and_device_properties(monkeypatch, runtime, artifact):
    benchmark = PerfBenchmark(BenchmarkConfig(model_id=artifact, runtime=runtime, memory=True))
    state = {"rss": 100.0, "gpu": None}
    events = []
    monkeypatch.setattr(memory, "get_rss_mb", lambda: state["rss"])
    monkeypatch.setattr(ProcessMemoryTracker, "_sample", lambda _: None)
    monkeypatch.setattr(
        memory,
        "_device_memory",
        lambda luid: (
            {"vram_local": state["gpu"], "vram_shared": state["gpu"]},
            {}
            if state["gpu"] is not None
            else {"vram_local": "unavailable", "vram_shared": "unavailable"},
        ),
    )

    def load():
        assert benchmark._memory_tracker.checkpoints["baseline"]["rss"] == 100
        benchmark._memory_tracker.bind_before_load("bound-luid")
        state.update(rss=300.0, gpu=200.0)
        events.append("load")
        benchmark._model = MagicMock()

    def inputs():
        events.append("inputs")
        state["rss"] = 310.0

    def inference():
        events.append("inference")
        state.update(rss=330.0, gpu=220.0)
        return object()

    def collect(_):
        return BenchmarkResult(
            config=benchmark.config,
            memory_profile=benchmark._memory,
            memory_measurement=benchmark._memory_measurement,
            process_memory=benchmark._process_memory,
            load_memory=benchmark._load_memory,
        )

    monkeypatch.setattr(benchmark, "_load_model", load)
    monkeypatch.setattr(benchmark, "_generate_inputs", inputs)
    monkeypatch.setattr(benchmark, "_run_benchmark", inference)
    monkeypatch.setattr(benchmark, "_collect_results", collect)
    monkeypatch.setattr("winml.modelkit.commands.perf.print_pre_bench_block", lambda *a, **kw: None)
    benchmark._ep_device = MagicMock()
    with patch.object(
        PerfBenchmark, "_is_composite", new_callable=PropertyMock, return_value=False
    ):
        result = benchmark.run()
    assert events == ["load", "inputs", "inference"]
    assert result.memory_profile["rss_baseline_mb"] == 100
    assert result.memory_profile["rss_after_load_mb"] == 300
    assert result.memory_profile["rss_model_load_delta_mb"] == 200
    assert result.load_memory["rss"]["ready_extra_mb"] == 200
    assert result.load_memory["rss"]["ready_mb"] == 300
    assert result.load_memory["status"] == "measured"
    assert result.memory_profile["rss_total_delta_mb"] == 230
    assert result.memory_profile["vram_local_baseline_mb"] is None
    assert result.memory_profile["vram_local_total_delta_mb"] is None
    assert result.memory_measurement["scope"] == "before_model_load_through_inference"
    assert benchmark._memory_tracker is None


def test_tracker_is_stopped_on_load_failure(monkeypatch):
    benchmark = PerfBenchmark(BenchmarkConfig(model_id="generated.onnx"))
    trackers = []
    original_start = ProcessMemoryTracker.start

    def start(tracker):
        trackers.append(tracker)
        original_start(tracker)

    def load():
        raise RuntimeError("load failed")

    monkeypatch.setattr(ProcessMemoryTracker, "start", start)
    monkeypatch.setattr(benchmark, "_load_model", load)
    with pytest.raises(RuntimeError, match="load failed"):
        benchmark.run()
    assert len(trackers) == 1 and not trackers[0]._thread.is_alive()
    assert benchmark._memory_tracker is None


def test_disabled_memory_does_not_start_tracker(monkeypatch):
    benchmark = PerfBenchmark(BenchmarkConfig(model_id="generated.onnx", memory=False))
    start = MagicMock()
    monkeypatch.setattr(ProcessMemoryTracker, "start", start)
    monkeypatch.setattr(
        benchmark, "_load_model", lambda: setattr(benchmark, "_model", SimpleNamespace())
    )
    monkeypatch.setattr(benchmark, "_run_single", lambda: "result")
    with patch.object(
        PerfBenchmark, "_is_composite", new_callable=PropertyMock, return_value=False
    ):
        assert benchmark.run() == "result"
    start.assert_not_called()


def test_load_binds_gpu_before_from_onnx(monkeypatch, tmp_path):
    """Exercise the real loader's pre-construction binding hook."""
    from winml.modelkit.models import WinMLAutoModel

    source = tmp_path / "generated.onnx"
    source.write_bytes(b"not read by the mocked model constructor")
    benchmark = PerfBenchmark(BenchmarkConfig(model_id=str(source)))
    benchmark._memory_tracker = ProcessMemoryTracker()
    monkeypatch.setattr(ProcessMemoryTracker, "_sample", lambda _: None)
    benchmark._memory_tracker.start()
    benchmark._ep_device = MagicMock()
    monkeypatch.setattr(benchmark, "_resolve_device_ep", lambda: None)
    monkeypatch.setattr(benchmark, "_resolve_adapter_luid", lambda: "selected-luid")
    monkeypatch.setattr(
        memory, "_device_memory", lambda _: ({"vram_local": 0.0, "vram_shared": 0.0}, {})
    )

    def construct(**_):
        assert benchmark._memory_tracker.adapter_luid == "selected-luid"
        assert benchmark._memory_tracker.checkpoints["baseline"]["vram_local"] == 0
        return MagicMock()

    monkeypatch.setattr(WinMLAutoModel, "from_onnx", construct)
    try:
        benchmark._load_model()
    finally:
        benchmark._memory_tracker.stop()


def test_nullable_memory_serialization_and_console():
    from io import StringIO

    from winml.modelkit.commands.perf import display_console_report
    from winml.modelkit.utils.console import SafeConsole

    result = BenchmarkResult(
        config=BenchmarkConfig(model_id="generated.onnx"),
        memory_profile={
            "rss_after_inference_mb": 100.0,
            "rss_total_delta_mb": -20.0,
            "vram_local_after_inference_mb": None,
            "vram_shared_after_inference_mb": 0.0,
        },
        memory_measurement={
            "scope": "before_model_load_through_inference",
            "missing_reasons": {"baseline": {"vram_local": "missing"}},
        },
        process_memory={"rss_peak_mb": 120.0, "sample_count": 3},
    )
    output = StringIO()
    display_console_report(result, SafeConsole(file=output, width=200))
    assert "N/A" in output.getvalue() and "-20.0" in output.getvalue()
    serialized = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert serialized["memory"]["vram_local_after_inference_mb"] is None
    assert serialized["process_memory"]["rss_peak_mb"] == 120.0


def test_genai_nullable_counter_consumer(monkeypatch):
    from winml.modelkit.commands._perf_genai import _GenaiMemoryTracker

    tracker = _GenaiMemoryTracker(adapter_luid="bound-luid")
    monkeypatch.setattr("winml.modelkit.commands._perf_genai._get_rss_mb", lambda: 100.0)
    monkeypatch.setattr("winml.modelkit.commands._perf_genai._get_vram_mb", lambda _: (None, 0.0))
    tracker.record_baseline()
    tracker.record_after_load()
    tracker.record_after_benchmark()
    result = tracker.to_dict()
    assert result["vram_local_total_delta_mb"] is None
    assert result["vram_local_checkpoint_peak_mb"] is None
    assert result["vram_shared_total_delta_mb"] == 0
    json.dumps(result, allow_nan=False)


def test_os_high_water_is_separate_from_sampled_peak(monkeypatch):
    tracker = ProcessMemoryTracker()
    peaks = iter([(120.0, None), (200.0, None)])
    monkeypatch.setattr(memory, "_rss_high_water", lambda: next(peaks))
    monkeypatch.setattr(tracker, "_sample", lambda: None)
    monkeypatch.setattr(tracker, "_observe", lambda: ({"rss": 100.0}, {}))
    tracker.start()
    tracker.stop()
    tracker.stop()
    values = tracker.sampled()
    assert values["rss_peak_mb"] is None
    assert values["os_rss_peak_before_mb"] == 120
    assert values["os_rss_peak_after_mb"] == 200
    assert "process lifetime" in values["os_rss_peak_scope"]


def test_os_high_water_unavailable_is_not_zero(monkeypatch):
    monkeypatch.setattr(
        memory.psutil,
        "Process",
        lambda _: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100)),
    )
    assert memory._rss_high_water() == (None, "process_high_water_unavailable")


def test_load_only_peak_excludes_later_inference_and_old_high_water(monkeypatch):
    tracker = ProcessMemoryTracker()
    state = {"rss": 100.0, "vram_local": 20.0, "vram_shared": 20.0}
    monkeypatch.setattr(tracker, "_observe", lambda: (dict(state), {}))
    monkeypatch.setattr(
        memory,
        "_device_memory",
        lambda _: ({"vram_local": state["vram_local"], "vram_shared": state["vram_shared"]}, {}),
    )
    # A larger old peak must not be attributed to loading this model.
    monkeypatch.setattr(memory, "_rss_high_water", lambda: (900.0, None))
    tracker.checkpoint("baseline")
    tracker.bind_before_load("bound-luid")
    state.update(rss=250.0, vram_local=60.0, vram_shared=60.0)
    tracker.record_model_ready()
    state.update(rss=1500.0, vram_local=500.0, vram_shared=500.0)
    tracker.checkpoint("after_inference")
    result = tracker.load_memory()
    assert result["rss"]["peak_mb"] == 250
    assert result["rss"]["peak_extra_mb"] == 150
    assert result["rss"]["peak_method"] == "sampled_and_endpoint_maximum"
    assert result["vram_local"]["ready_extra_mb"] == 40
    assert result["version"] == 1


def test_new_os_peak_in_load_window_includes_temporary_compile_allocations(monkeypatch):
    tracker = ProcessMemoryTracker()
    state = {"rss": 100.0, "vram_local": None, "vram_shared": 0.0}
    monkeypatch.setattr(tracker, "_observe", lambda: (dict(state), {"vram_local": "unavailable"}))
    monkeypatch.setattr(
        memory,
        "_device_memory",
        lambda _: ({"vram_local": None, "vram_shared": 0.0}, {"vram_local": "unavailable"}),
    )
    peaks = iter([(110.0, None), (450.0, None)])
    monkeypatch.setattr(memory, "_rss_high_water", lambda: next(peaks))
    tracker.checkpoint("baseline")
    tracker.bind_before_load("bound-luid")
    state["rss"] = 200.0
    tracker.record_model_ready()
    result = tracker.load_memory()
    assert result["rss"]["peak_extra_mb"] == 350
    assert result["rss"]["ready_extra_mb"] == 100
    assert result["rss"]["peak_method"] == "new_os_process_high_water_during_load"
    assert result["vram_local"]["peak_extra_mb"] is None
    assert result["vram_local"]["missing_reason"] == "unavailable"
    assert ProcessMemoryTracker().load_memory()["status"] == "unavailable"


@pytest.mark.parametrize("missing_local_baseline", [False, True])
def test_load_peak_includes_checkpoints_without_counting_them_as_samples(
    monkeypatch, missing_local_baseline
):
    tracker = ProcessMemoryTracker()
    state = {
        "rss": 100.0,
        "vram_local": None if missing_local_baseline else 10.0,
        "vram_shared": 10.0,
    }
    monkeypatch.setattr(tracker, "_observe", lambda: (dict(state), {}))
    monkeypatch.setattr(
        memory,
        "_device_memory",
        lambda _: ({key: value for key, value in state.items() if key != "rss"}, {}),
    )
    monkeypatch.setattr(memory, "_rss_high_water", lambda: (900.0, None))
    tracker.checkpoint("baseline")
    tracker.bind_before_load("bound-luid")
    state.update(rss=400.0, vram_local=200.0, vram_shared=100.0)
    tracker.checkpoint("after_load")
    state.update(rss=150.0, vram_local=30.0, vram_shared=20.0)
    tracker.record_model_ready()
    state.update(rss=1500.0, vram_local=500.0, vram_shared=300.0)
    tracker.checkpoint("after_inference")
    result = tracker.load_memory()
    for family, peak, ready in (
        ("rss", 400.0, 150.0),
        ("vram_local", 200.0, 30.0),
        ("vram_shared", 100.0, 20.0),
    ):
        assert result[family]["peak_mb"] == peak
        assert result[family]["ready_mb"] == ready
        assert result[family]["sample_count"] == 0
    assert result["rss"]["peak_extra_mb"] == 300.0
    assert result["rss"]["peak_method"] == "sampled_and_endpoint_maximum"
    assert result["vram_local"]["peak_extra_mb"] == (None if missing_local_baseline else 190.0)
    assert tracker.sampled()["sample_count"] == 0
