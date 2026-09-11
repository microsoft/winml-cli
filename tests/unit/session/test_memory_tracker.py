# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Memory validity and phase boundaries; no model download or GPU required."""

from unittest.mock import Mock

import pytest

from winml.modelkit.session.monitor import memory_tracker as mt


def test_invalid_counter_is_null_but_valid_zero_survives(monkeypatch):
    from winml.modelkit.session.monitor import _pdh

    query = Mock()
    query.collect.return_value = {"local_0": None, "shared_0": 0}
    query.diagnostics = {"local_0": {"data_status": 0x800007D1}}
    monkeypatch.setattr(_pdh, "memory_instances", lambda *args: ["pid_1_luid_x_phys_2"])
    monkeypatch.setattr(_pdh, "PdhQuery", lambda: query)
    sample = mt.sample_vram("0x0_0x1")
    assert sample["local"]["value_bytes"] is None
    assert sample["local"]["status"] == "unavailable"
    assert sample["shared"]["value_bytes"] == 0
    assert sample["shared"]["status"] == "valid"
    assert sample["local"]["diagnostics"]["local_0"]["data_status"] == 0x800007D1
    query.close.assert_called_once()


def test_query_closes_on_collect_failure(monkeypatch):
    from winml.modelkit.session.monitor import _pdh

    query = Mock()
    query.collect.side_effect = RuntimeError("counter disappeared")
    monkeypatch.setattr(_pdh, "memory_instances", lambda *args: ["node"])
    monkeypatch.setattr(_pdh, "PdhQuery", lambda: query)
    assert mt.sample_vram("x")["local"]["value_bytes"] is None
    query.close.assert_called_once()


def test_missing_checkpoint_never_becomes_delta_or_peak(monkeypatch):
    tracker = mt.MemoryTracker(None, baseline="test", device="gpu")
    for phase, rss, local in (
        ("baseline", 100, None),
        ("after_compile", 200, 20),
        ("after_inference", 150, 30),
    ):
        tracker.checkpoints[phase] = {
            k: mt.metric(v * mt.MIB if v is not None else None, "test")
            for k, v in (("rss", rss), ("vram_local", local), ("vram_shared", 0))
        }
    profile = tracker.profile()
    assert profile["rss_inference_delta_mb"] == -50
    assert profile["rss_checkpoint_peak_mb"] == 200
    assert profile["vram_local_total_delta_mb"] is None
    assert profile["vram_local_checkpoint_peak_mb"] is None
    assert profile["vram_local_inference_delta_mb"] == 10
    assert profile["vram_shared_total_delta_mb"] == 0


def test_api_status_and_counter_status_are_distinct():
    from winml.modelkit.session.monitor._pdh import _pdh_data_ok, _pdh_ok

    assert _pdh_data_ok(1)
    assert _pdh_data_ok(0)
    assert not _pdh_ok(1)
    assert not _pdh_data_ok(0x800007D1)


def test_node_sum_and_idle_memory_do_not_become_zero():
    from winml.modelkit.session.monitor._pdh import PdhPoller, _sum_memory_nodes

    assert (
        _sum_memory_nodes(
            {"memory_local_bytes": 10, "memory_local_bytes_1": 20}, "memory_local_bytes"
        )
        == 30
    )
    assert (
        _sum_memory_nodes(
            {"memory_local_bytes": 10, "memory_local_bytes_1": None}, "memory_local_bytes"
        )
        is None
    )
    assert _sum_memory_nodes({}, "memory_local_bytes") is None
    poller = PdhPoller(device="cpu")
    assert poller.ram_used_mb is None
    assert poller.peak_ram_used_mb is None
    assert poller.peak_memory_mb is None
    poller._memory_local_bytes = [0]
    assert poller.peak_memory_mb == 0


def test_memory_evidence_serializes_nulls(monkeypatch):
    import json

    from winml.modelkit.commands.perf import BenchmarkConfig, BenchmarkResult

    tracker = mt.MemoryTracker(None, baseline="before model", device="cpu")
    for phase in ("baseline", "after_compile", "after_inference"):
        tracker.capture(phase)
    result = BenchmarkResult(
        config=BenchmarkConfig(model_id="test"),
        memory_profile=tracker.profile(),
        memory_measurement=tracker.evidence(),
    )
    data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert data["memory"]["vram_local_total_delta_mb"] is None
    assert (
        data["memory_measurement"]["checkpoints"]["baseline"]["vram_local"]["status"]
        == "not_applicable"
    )


def test_pdh_collect_preserves_new_data_and_errors(monkeypatch):
    from winml.modelkit.session.monitor import _pdh

    query = _pdh.PdhQuery()
    entry = _pdh._CounterEntry("memory", "path", registered=True)
    query._counters = [entry]
    monkeypatch.setattr(_pdh._pdh, "PdhCollectQueryData", lambda *args: 0)

    def formatted(handle, fmt, counter_type, value):
        value._obj.CStatus = 1
        value._obj.largeValue = 123
        return 0

    monkeypatch.setattr(_pdh._pdh, "PdhGetFormattedCounterValue", formatted)
    assert query._collect_once() == {"memory": 123}
    assert query.diagnostics["memory"]["data_status"] == 1
    monkeypatch.setattr(_pdh._pdh, "PdhCollectQueryData", lambda *args: 0x800007D5)
    assert query._collect_once() == {"memory": None}


def test_perf_baseline_precedes_eager_factory(monkeypatch):
    from winml.modelkit.commands.perf import BenchmarkConfig, PerfBenchmark

    events = []
    bench = PerfBenchmark(BenchmarkConfig(model_id="test"))
    monkeypatch.setattr(bench, "_resolve_device_ep", lambda: events.append("resolve"))
    monkeypatch.setattr(
        bench,
        "_start_memory",
        lambda baseline, **kwargs: events.append(kwargs.get("phase", "baseline")),
    )

    def load():
        events.append("eager_load")
        bench._model = object()

    monkeypatch.setattr(bench, "_load_model", load)
    monkeypatch.setattr(PerfBenchmark, "_is_composite", property(lambda self: False))
    monkeypatch.setattr(bench, "_run_single", lambda: events.append("inference"))
    bench.run()
    assert events == ["resolve", "before_model_load", "eager_load", "inference"]


def test_adapter_resolution_never_reads_lazy_model(monkeypatch):
    from winml.modelkit.commands.perf import BenchmarkConfig, PerfBenchmark

    bench = PerfBenchmark(BenchmarkConfig(model_id="test"))
    monkeypatch.setattr(
        "winml.modelkit.commands.perf._get_ep_device_binding",
        lambda *args: ("0x00000000_0x0000007B", "gpu"),
    )
    monkeypatch.setattr(
        PerfBenchmark, "_single", property(lambda self: pytest.fail("model read before baseline"))
    )
    assert bench._resolve_adapter_luid() == "0x00000000_0x0000007B"


def test_main_mean_metrics_keep_unavailable_and_zero_distinct():
    from winml.modelkit.session.monitor._pdh import PdhPoller

    poller = PdhPoller(device="cpu")
    assert poller.mean_ram_used_mb is None
    assert poller.mean_memory_local_mb is None
    assert poller.mean_memory_shared_mb is None
    poller._memory_local_bytes = [0, 0]
    assert poller.mean_memory_local_mb == 0


def test_genai_consumer_propagates_missing_vram_without_crashing(monkeypatch):
    from winml.modelkit.commands import _perf_genai as genai

    monkeypatch.setattr(genai, "_get_rss_mb", lambda: 100.0)
    samples = iter([(None, 0.0), (20.0, 0.0), (30.0, 0.0)])
    monkeypatch.setattr(genai, "_get_vram_mb", lambda _: next(samples))
    tracker = genai._GenaiMemoryTracker(adapter_luid="bound-luid")
    tracker.record_baseline()
    tracker.record_after_load()
    tracker.record_after_benchmark()
    result = tracker.to_dict()
    assert result["vram_local_baseline_mb"] is None
    assert result["vram_local_total_delta_mb"] is None
    assert result["vram_local_checkpoint_peak_mb"] is None
    assert result["vram_local_inference_delta_mb"] == 10.0
    assert result["vram_shared_total_delta_mb"] == 0.0


def test_genai_display_accepts_unavailable_ram_and_vram():
    from io import StringIO
    from pathlib import Path

    from rich.console import Console

    from winml.modelkit.commands._perf_genai import (
        GenaiBenchmarkResult,
        GenaiPerfConfig,
        display_genai_report,
    )

    output = StringIO()
    result = GenaiBenchmarkResult(
        config=GenaiPerfConfig(bundle_dir=Path("test")),
        hw_monitor={"ram": {"used_mb": None}},
        memory_profile={
            "rss_after_inference_mb": 100,
            "rss_model_load_delta_mb": 20,
            "rss_inference_delta_mb": 0,
            "rss_total_delta_mb": 20,
            "vram_local_after_inference_mb": None,
            "vram_shared_after_inference_mb": None,
        },
    )
    display_genai_report(result, Console(file=output, width=160))
    assert "RAM: unavailable" in output.getvalue()
    assert "GPU local/shared: unavailable/unavailable" in output.getvalue()


def test_additional_baseline_does_not_change_legacy_fields():
    tracker = mt.MemoryTracker(None, baseline="legacy", device="gpu")
    for phase, value in (
        ("before_model_load", 300),
        ("baseline", 100),
        ("after_compile", 180),
        ("after_inference", 150),
    ):
        tracker.checkpoints[phase] = {
            key: mt.metric(value * mt.MIB, "test") for key in ("rss", "vram_local", "vram_shared")
        }
    result = tracker.profile()
    for key in ("rss", "vram_local", "vram_shared"):
        assert result[f"{key}_baseline_mb"] == 100
        assert result[f"{key}_model_load_delta_mb"] == 80
        assert result[f"{key}_inference_delta_mb"] == -30
        assert result[f"{key}_total_delta_mb"] == 50
        assert result[f"{key}_checkpoint_peak_mb"] == 180  # excludes new 300-MiB point
        assert result[f"{key}_before_model_load_mb"] == 300
        assert result[f"{key}_model_factory_delta_mb"] == -200
        assert result[f"{key}_total_from_before_model_load_delta_mb"] == -150


def test_preloaded_model_does_not_invent_before_load_baseline():
    tracker = mt.MemoryTracker(None, baseline="preloaded", device="cpu")
    for phase in ("baseline", "after_compile", "after_inference"):
        tracker.capture(phase)
    result = tracker.profile()
    assert result["rss_before_model_load_mb"] is None
    assert result["rss_total_from_before_model_load_delta_mb"] is None
    assert tracker.evidence()["before_model_load_status"] == "unavailable"


def test_perf_records_both_boundaries_around_factory_and_inputs(monkeypatch):
    from unittest.mock import MagicMock

    from winml.modelkit.commands import perf as perf_module

    events = []
    bench = perf_module.PerfBenchmark(perf_module.BenchmarkConfig(model_id="test"))
    monkeypatch.setattr(bench, "_resolve_device_ep", lambda: None)
    monkeypatch.setattr(perf_module, "_get_ep_device_binding", lambda *args: (None, "cpu"))
    monkeypatch.setattr(mt.MemoryTracker, "capture", lambda self, phase: events.append(phase))
    monkeypatch.setattr(mt.MemoryTracker, "profile", lambda self: {})
    single = MagicMock()
    single._session.compile.side_effect = lambda: events.append("compile")

    def factory():
        events.append("factory")
        bench._model = single
        bench._ep_device = MagicMock()

    monkeypatch.setattr(bench, "_load_model", factory)
    monkeypatch.setattr(perf_module.PerfBenchmark, "_is_composite", property(lambda self: False))
    monkeypatch.setattr(bench, "_generate_inputs", lambda: events.append("inputs"))
    monkeypatch.setattr(bench, "_run_benchmark", lambda: events.append("inference"))
    monkeypatch.setattr(bench, "_collect_results", lambda stats: None)
    monkeypatch.setattr(perf_module, "_pre_bench_kwargs_from_ep_device", lambda *a, **k: {})
    monkeypatch.setattr(perf_module, "print_pre_bench_block", lambda *a, **k: None)
    bench.run()
    assert events == [
        "before_model_load",
        "factory",
        "baseline",
        "inputs",
        "compile",
        "after_compile",
        "inference",
        "after_inference",
    ]
