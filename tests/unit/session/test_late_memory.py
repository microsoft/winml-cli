# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Late GPU instances, measured zeros and unknown coverage stay distinct."""

from unittest.mock import Mock

import pytest

from winml.modelkit.session.monitor import _pdh
from winml.modelkit.session.monitor import memory_tracker as mt


@pytest.fixture
def query(monkeypatch):
    q = _pdh.PdhQuery()
    q.watch_memory(123, "bound")
    monkeypatch.setattr(_pdh._pdh, "PdhAddEnglishCounterW", lambda *args: 0)
    monkeypatch.setattr(_pdh._pdh, "PdhCollectQueryData", lambda *args: 0)

    def formatted(handle, fmt, counter_type, value):
        value._obj.CStatus = 0
        value._obj.largeValue = 64 * 1048576
        return 0

    monkeypatch.setattr(_pdh._pdh, "PdhGetFormattedCounterValue", formatted)
    return q


def test_initial_absence_recovers_and_keeps_gap(query, monkeypatch):
    instances = []
    monkeypatch.setattr(_pdh, "memory_instances", lambda pid, luid: list(instances))
    poller = _pdh.PdhPoller(device="gpu")
    values = query._collect_once()
    poller._record_memory(query, values)
    assert query.memory_readings["local"]["state"] == "absent_unconfirmed"
    assert poller.peak_memory_local_mb is None
    instances.append("pid_123_luid_bound_phys_1")
    query._memory_next_discovery = 0
    poller._record_memory(query, query._collect_once())
    assert poller.peak_memory_local_mb == 64
    assert poller.mean_memory_local_mb == 64  # unknown sample was NOT zero
    assert poller.memory_coverage["local"]["coverage"] == "partial"
    assert poller.memory_coverage["local"]["missing_samples"] == 1
    query._memory_next_discovery = 0
    query._collect_once()
    assert len(query._counters) == 2


def test_enumeration_failure_distinct_and_retries(query, monkeypatch):
    discover = Mock(side_effect=[RuntimeError("PDH failed"), ["node"]])
    monkeypatch.setattr(_pdh, "memory_instances", discover)
    query._collect_once()
    assert query.memory_readings["local"]["state"] == "error"
    query._memory_next_discovery = 0
    assert _pdh._sum_memory_nodes(query._collect_once(), "memory_local_bytes") == 64 * 1048576


def test_failed_registration_retried_without_duplicate_counter(query, monkeypatch):
    monkeypatch.setattr(_pdh, "memory_instances", lambda *args: ["node"])
    monkeypatch.setattr(_pdh._pdh, "PdhAddEnglishCounterW", lambda *args: 0xC0000BB9)
    values = query._collect_once()
    assert _pdh._sum_memory_nodes(values, "memory_local_bytes") is None
    monkeypatch.setattr(_pdh._pdh, "PdhAddEnglishCounterW", lambda *args: 0)
    query._memory_next_discovery = 0
    assert _pdh._sum_memory_nodes(query._collect_once(), "memory_local_bytes") == 64 * 1048576
    assert len(query._counters) == 2


def test_disappeared_instance_is_not_stale_or_zero(query, monkeypatch):
    discover = Mock(side_effect=[["node"], [], ["node"]])
    monkeypatch.setattr(_pdh, "memory_instances", discover)
    query._collect_once()
    query._memory_next_discovery = 0
    assert _pdh._sum_memory_nodes(query._collect_once(), "memory_local_bytes") is None
    query._memory_next_discovery = 0
    assert _pdh._sum_memory_nodes(query._collect_once(), "memory_local_bytes") == 64 * 1048576
    assert len(query._counters) == 2


def test_valid_zero_is_confirmed_by_counter_not_absence(query, monkeypatch):
    monkeypatch.setattr(_pdh, "memory_instances", lambda *args: ["node"])

    def zero(handle, fmt, ct, value):
        value._obj.CStatus = 1  # valid NEW_DATA
        value._obj.largeValue = 0
        return 0

    monkeypatch.setattr(_pdh._pdh, "PdhGetFormattedCounterValue", zero)
    query._collect_once()
    assert query.memory_readings["local"] == {
        "value_bytes": 0,
        "state": "measured_zero",
        "reason": None,
    }
    assert mt.metric(0, "PDH")["value_origin"] == "measured_zero"
    assert mt.metric(None, "PDH", "absent")["value_origin"] == "unknown"


def test_new_nodes_do_not_reset_old_stats(query, monkeypatch):
    instances = ["node1"]
    monkeypatch.setattr(_pdh, "memory_instances", lambda *args: list(instances))
    first = query._collect_once()
    assert _pdh._sum_memory_nodes(first, "memory_local_bytes") == 64 * 1048576
    instances.append("node2")
    query._memory_next_discovery = 0
    assert _pdh._sum_memory_nodes(query._collect_once(), "memory_local_bytes") == 128 * 1048576
    assert len(query._counters) == 4


def test_checkpoint_retry_can_observe_late_instance(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(mt.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(mt.time, "sleep", lambda n: clock.__setitem__(0, clock[0] + n))
    monkeypatch.setattr(_pdh, "memory_instances", Mock(side_effect=[[], [], ["node"]]))
    q = Mock()
    q.collect.return_value = {"local_0": 0, "shared_0": 0}
    q.diagnostics = {}
    monkeypatch.setattr(_pdh, "PdhQuery", lambda: q)
    assert mt.sample_vram("bound")["local"]["value_origin"] == "measured_zero"
    assert clock[0] == 0.1
    q.close.assert_called_once()


def test_genai_monitor_recovers_when_load_creates_instance(query, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from winml.modelkit.commands._perf_genai import GenaiPerfBenchmark, GenaiPerfConfig

    instances = []
    monkeypatch.setattr(_pdh, "memory_instances", lambda *args: list(instances))
    poller = _pdh.PdhPoller(device="gpu")

    class Monitor:
        def __enter__(self):
            poller._record_memory(query, query._collect_once())
            assert poller.peak_memory_local_mb is None
            return self

        def __exit__(self, *args):
            pass

        def to_dict(self):
            return {
                "device_memory": {
                    "local_peak_mb": poller.peak_memory_local_mb,
                    "coverage": poller.memory_coverage,
                }
            }

    session = Mock()
    session.load.side_effect = lambda: instances.append("late-node")
    bench = GenaiPerfBenchmark(
        GenaiPerfConfig(
            bundle_dir=Path("test"), warmup=0, iterations=1, memory=False, monitor=True
        ),
        session=session,
    )
    monkeypatch.setattr(bench, "_build_hw_monitor", Monitor)

    def generate(*args, **kwargs):
        query._memory_next_discovery = 0
        poller._record_memory(query, query._collect_once())
        return object()

    monkeypatch.setattr(bench, "_time_one_generation", generate)
    monkeypatch.setattr(bench, "_aggregate", lambda *a, **kw: SimpleNamespace())
    result = bench.run()
    assert result.hw_monitor["device_memory"]["local_peak_mb"] == 64
    assert result.hw_monitor["device_memory"]["coverage"]["local"]["missing_samples"] == 1
