# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for WinMLEPMonitor, VitisAIMonitor, and internal helpers."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.session import PerfStats, VitisAIMonitor, WinMLEPMonitor


# ============================================================================
# WinMLEPMonitor (ABC) tests
# ============================================================================


class TestEPMonitor:
    """Test WinMLEPMonitor abstract base class."""

    def test_cannot_instantiate(self):
        """WinMLEPMonitor is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            WinMLEPMonitor()

    def test_subclass_must_implement_all_methods(self):
        """Concrete subclass missing the remaining abstract methods raises TypeError.

        Post-v2.4 the ABC's abstract surface is ``__enter__``, ``__exit__``,
        and ``is_available`` — ``to_dict`` is no longer in the contract.
        """

        class IncompleteMonitor(WinMLEPMonitor):
            pass

        with pytest.raises(TypeError):
            IncompleteMonitor()

    def test_concrete_subclass_works(self):
        """A fully-implemented subclass can be instantiated."""

        class DummyMonitor(WinMLEPMonitor):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            @classmethod
            def is_available(cls):
                return True

        mon = DummyMonitor()
        assert isinstance(mon, WinMLEPMonitor)
        assert DummyMonitor.is_available() is True
        # Default v2.4 typed-accessor contract — None unless populated.
        assert mon.result is None

    def test_null_ep_monitor(self):
        """NullEPMonitor implements Null Object Pattern correctly."""
        from winml.modelkit.session import NullEPMonitor

        mon = NullEPMonitor()
        assert NullEPMonitor.is_available() is True
        # v2.4: NullEPMonitor exposes no data via the typed accessor.
        assert mon.result is None

        # Context manager works
        with mon as m:
            assert m is mon


# ============================================================================
# PerfStats tests
# ============================================================================


class TestPerfStatsImport:
    """Verify PerfStats imports work from session package."""

    def test_import_from_submodule(self):
        from winml.modelkit.session import PerfStats

        assert PerfStats is not None

    def test_import_from_session(self):
        from winml.modelkit.session import PerfStats

        assert PerfStats is not None

    def test_basic_functionality(self):
        stats = PerfStats(warmup=2)
        for _ in range(5):
            stats.record(lambda: time.sleep(0.001))
        assert stats.total_count == 5
        assert stats.count == 3  # 5 - 2 warmup
        assert stats.mean_ms > 0


# ============================================================================
# _xrt_smi tests
# ============================================================================


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
class TestXrtSmiClient:
    """Test _xrt_smi.XrtSmiClient with mocked subprocess."""

    def test_is_available_checks_exe_exists(self):
        from winml.modelkit.session.monitor._xrt_smi import XrtSmiClient

        client = XrtSmiClient(exe_path=Path("/nonexistent/path"))
        assert client.is_available is False

    def test_snapshot_returns_empty_when_unavailable(self):
        from winml.modelkit.session.monitor._xrt_smi import XrtSmiClient

        client = XrtSmiClient(exe_path=Path("/nonexistent/path"))
        assert client.snapshot() == {}

    def test_get_hw_contexts_parses_json(self):
        from winml.modelkit.session.monitor._xrt_smi import XrtSmiClient

        sample_data = {
            "devices": [
                {
                    "aie_partitions": {
                        "partitions": [
                            {
                                "hw_contexts": [
                                    {
                                        "pid": "12345",
                                        "context_id": "0",
                                        "status": "Active",
                                        "command_submissions": "100",
                                        "command_completions": "99",
                                        "gops": "N/A",
                                        "fps": "N/A",
                                        "latency": "N/A",
                                        "priority": "Low",
                                        "errors": "0",
                                    },
                                    {
                                        "pid": "99999",
                                        "context_id": "1",
                                        "status": "Idle",
                                        "command_submissions": "50",
                                        "command_completions": "50",
                                        "gops": "N/A",
                                        "fps": "N/A",
                                        "latency": "N/A",
                                        "priority": "Normal",
                                        "errors": "0",
                                    },
                                ]
                            }
                        ]
                    }
                }
            ]
        }

        client = XrtSmiClient()
        with patch.object(client, "snapshot", return_value=sample_data):
            # All contexts
            all_ctxs = client.get_hw_contexts()
            assert len(all_ctxs) == 2

            # Filter by PID
            ctxs = client.get_hw_contexts(pid=12345)
            assert len(ctxs) == 1
            assert ctxs[0].pid == 12345
            assert ctxs[0].command_submissions == 100
            assert ctxs[0].command_completions == 99
            assert ctxs[0].status == "Active"

            # No match
            ctxs = client.get_hw_contexts(pid=0)
            assert len(ctxs) == 0

    def test_get_command_submissions_sums_across_contexts(self):
        from winml.modelkit.session.monitor._xrt_smi import HwContext, XrtSmiClient

        client = XrtSmiClient()
        contexts = [
            HwContext(
                pid=100,
                context_id=0,
                status="Active",
                command_submissions=50,
                command_completions=50,
                gops="N/A",
                fps="N/A",
                latency="N/A",
                priority="Low",
                errors=0,
            ),
            HwContext(
                pid=100,
                context_id=1,
                status="Idle",
                command_submissions=30,
                command_completions=30,
                gops="N/A",
                fps="N/A",
                latency="N/A",
                priority="Low",
                errors=0,
            ),
        ]
        with patch.object(client, "get_hw_contexts", return_value=contexts):
            assert client.get_command_submissions(pid=100) == 80


# ============================================================================
# _pdh tests
# ============================================================================


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
class TestPdhModule:
    """Test _pdh module functions."""

    def test_discover_npu_luid_returns_string_or_none(self):
        from winml.modelkit.sysinfo.pdh_adapters import discover_npu_luid

        result = discover_npu_luid()
        # On a system with NPU, returns a LUID string; otherwise None
        if result is not None:
            assert isinstance(result, str)
            assert "0x" in result

    def test_enumerate_adapters_returns_dict(self):
        from winml.modelkit.sysinfo.pdh_adapters import enumerate_adapters

        adapters = enumerate_adapters()
        assert isinstance(adapters, dict)
        # May be empty in containers without WDDM drivers.

    def test_adapter_info_npu_heuristic(self):
        from winml.modelkit.sysinfo.pdh_adapters import AdapterInfo

        # Compute-only → NPU
        npu = AdapterInfo(luid="test", engine_types={"Compute"})
        assert npu.is_npu is True

        # Has 3D → GPU
        gpu = AdapterInfo(luid="test", engine_types={"3D", "Compute", "Copy"})
        assert gpu.is_npu is False

        # Empty → not NPU
        empty = AdapterInfo(luid="test", engine_types=set())
        assert empty.is_npu is False

    def test_pdh_query_lifecycle(self):
        from winml.modelkit.session.monitor._pdh import PdhQuery

        query = PdhQuery()
        query.open()
        # Add a counter that always exists on Windows
        registered = query.add_counter(
            "test_cpu",
            r"\Processor(_Total)\% Processor Time",
            fmt="double",
        )
        assert registered is True
        assert "test_cpu" in query.counter_names

        query.prime()
        values = query.collect()
        assert "test_cpu" in values
        assert values["test_cpu"] is not None, f"PDH returned None for CPU counter: {values}"

        query.close()

    def test_build_npu_query_on_npu_system(self):
        from winml.modelkit.session.monitor._pdh import build_npu_query
        from winml.modelkit.sysinfo.pdh_adapters import discover_npu_luid

        luid = discover_npu_luid()
        if luid is None:
            pytest.skip("No NPU detected")

        query = build_npu_query(luid)
        names = query.counter_names
        # build_npu_query registers one util_* and one running_time_* counter
        # per Compute_* engine on the adapter, plus the shared memory pair.
        assert any(n.startswith("util_Compute") for n in names)
        assert any(n.startswith("running_time_Compute") for n in names)
        assert "memory_local_bytes" in names
        assert "memory_shared_bytes" in names
        query.close()


# ============================================================================
# VitisAIMonitor tests
# ============================================================================


class TestVitisAIMonitor:
    """Test VitisAIMonitor."""

    def test_is_available_returns_bool(self):
        result = VitisAIMonitor.is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_context_manager_lifecycle(self):
        """Monitor can enter and exit without errors."""
        with VitisAIMonitor() as hw:
            time.sleep(0.2)

        # After exit, metrics should be accessible
        assert isinstance(hw.npu_proven, bool)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_to_dict_structure(self):
        """to_dict returns expected keys."""
        with VitisAIMonitor() as hw:
            time.sleep(0.1)

        d = hw.to_dict()
        assert d["ep"] == "VitisAI"
        assert "npu_proven" in d
        assert "xrt_smi" in d
        assert "command_submissions" in d["xrt_smi"]
        assert "command_completions" in d["xrt_smi"]
        assert "hw_context_status" in d["xrt_smi"]

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_idle_shows_no_npu_activity(self):
        """Without inference, npu_proven should be False (or very low)."""
        with VitisAIMonitor() as hw:
            time.sleep(0.3)

        # No inference ran — NPU should be idle
        assert hw.command_submissions == 0

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_companion_pattern_with_perfstats(self):
        """VitisAIMonitor works alongside PerfStats."""
        stats = PerfStats(warmup=2)
        with VitisAIMonitor() as hw:
            for _ in range(5):
                stats.record(lambda: time.sleep(0.01))

        assert stats.count == 3  # 5 - 2 warmup
        assert stats.mean_ms > 0
        assert isinstance(hw.to_dict(), dict)


# ============================================================================
# PdhPoller tests
# ============================================================================


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
class TestPdhPoller:
    """Test the composable PdhPoller helper."""

    def test_is_npu_available_returns_bool(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        result = PdhPoller.is_npu_available()
        assert isinstance(result, bool)

    def test_start_stop_lifecycle(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        poller = PdhPoller(poll_interval_ms=50)
        poller.start()
        time.sleep(0.2)
        poller.stop()

        assert isinstance(poller.mean_utilization_pct, float)
        assert isinstance(poller.peak_utilization_pct, float)
        assert isinstance(poller.peak_memory_mb, float)

    def test_collects_samples_over_time(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        poller = PdhPoller(poll_interval_ms=50)
        poller.start()
        time.sleep(0.3)
        poller.stop()

        # On an idle system the NPU counter may return None for every poll,
        # so utilization_samples can legitimately be empty.
        samples = poller.utilization_samples
        assert isinstance(samples, list)
        for val in samples:
            assert isinstance(val, float)

    def test_running_time_delta(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        poller = PdhPoller(poll_interval_ms=50)
        poller.start()
        time.sleep(0.2)
        poller.stop()

        assert poller.running_time_delta_ns >= 0

    def test_adapter_luid_populated_after_start(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller
        from winml.modelkit.sysinfo.pdh_adapters import discover_npu_luid

        luid = discover_npu_luid()
        if luid is None:
            pytest.skip("No NPU detected")

        poller = PdhPoller(poll_interval_ms=50, device="npu")
        poller.start()
        poller.stop()
        assert poller.adapter_luid == luid

    def test_memory_samples_mb_returns_list(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        poller = PdhPoller(poll_interval_ms=50)
        poller.start()
        time.sleep(0.2)
        poller.stop()

        result = poller.memory_samples_mb
        assert isinstance(result, list)
        for val in result:
            assert isinstance(val, float)

    def test_poll_loop_takes_max_util_across_engines(self):
        """`_poll_loop` must reduce per-engine ``util_*`` samples with max.

        Pins the contract that multi-engine util collapses to the busiest
        engine's reading -- a future swap to mean/sum would silently change
        every perf summary, so guard it here.
        """
        from winml.modelkit.session.monitor._pdh import PdhPoller

        poller = PdhPoller.__new__(PdhPoller)
        poller._stop_event = threading.Event()
        poller._lock = threading.Lock()
        poller._poll_interval_s = 0.0
        poller._util_samples = []
        poller._memory_local_bytes = []
        poller._memory_shared_bytes = []
        poller._cpu_samples = []
        poller._ram_used_bytes = []
        poller._gpu_counter_names = []
        poller._gpu_samples = []

        sample = {
            "util_Compute_0": 80.0,
            "util_Compute_1": 30.0,
            "util_3D": 10.0,
            "memory_local_bytes": None,
            "memory_shared_bytes": None,
            "cpu_pct_raw": None,
            "ram_working_set_bytes": None,
        }

        def collect_then_stop():
            poller._stop_event.set()
            return sample

        poller._query = MagicMock()
        poller._query._collect_once.side_effect = collect_then_stop

        poller._poll_loop()

        assert poller._util_samples == [80.0]

    def test_running_time_delta_sums_across_engines(self):
        """``running_time_delta_ns`` must add per-engine deltas.

        Each engine's Running Time counter is independent wall-clock work,
        so total adapter compute time is additive. A future swap to max
        would silently halve numbers on multi-engine workloads.
        """
        from winml.modelkit.session.monitor._pdh import PdhPoller

        poller = PdhPoller.__new__(PdhPoller)
        poller._running_time_start_ns = {
            "running_time_Compute_0": 1000,
            "running_time_3D": 500,
        }
        poller._running_time_end_ns = {
            "running_time_Compute_0": 1500,
            "running_time_3D": 800,
        }
        # (1500 - 1000) + (800 - 500) = 800
        assert poller.running_time_delta_ns == 800


# ============================================================================
# HWMonitor tests
# ============================================================================


class TestHWMonitor:
    """Test universal PDH-based HWMonitor."""

    def test_is_available_returns_bool(self):
        from winml.modelkit.session import HWMonitor

        result = HWMonitor.is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_context_manager_lifecycle(self):
        from winml.modelkit.session import HWMonitor

        with HWMonitor(poll_interval_ms=50) as hw:
            time.sleep(0.2)

        assert isinstance(hw.mean_utilization_pct, float)
        assert isinstance(hw.peak_utilization_pct, float)
        assert isinstance(hw.peak_memory_mb, float)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_to_dict_structure(self):
        from winml.modelkit.session import HWMonitor

        with HWMonitor(poll_interval_ms=50) as hw:
            time.sleep(0.1)

        d = hw.to_dict()
        assert d["monitor"] == "HWMonitor"
        # CPU section
        assert "cpu" in d
        assert "mean_pct" in d["cpu"]
        assert "peak_pct" in d["cpu"]
        assert "sample_count" in d["cpu"]
        # RAM section
        assert "ram" in d
        assert "used_mb" in d["ram"]
        assert "peak_mb" in d["ram"]
        # Aggregate GPU telemetry is independent of the selected inference
        # adapter, so this section is always present.
        assert "gpu" in d
        assert "mean_pct" in d["gpu"]
        assert "peak_pct" in d["gpu"]
        assert "sample_count" in d["gpu"]
        # Selected adapter section.
        kind = d["device_kind"]
        if kind == "npu":
            assert "npu" in d
            assert "mean_pct" in d["npu"]
            assert "peak_pct" in d["npu"]
            assert "sample_count" in d["npu"]
        elif kind == "gpu":
            assert "npu" not in d
        else:
            assert "npu" not in d
        # Device memory + running time
        assert "device_memory" in d
        assert "local_peak_mb" in d["device_memory"]
        assert "shared_peak_mb" in d["device_memory"]
        assert "running_time_ns" in d

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_idle_shows_low_utilization(self):
        from winml.modelkit.session import HWMonitor

        with HWMonitor(poll_interval_ms=50) as hw:
            time.sleep(0.2)

        assert hw.mean_utilization_pct < 50.0

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_utilization_samples_accessible(self):
        from winml.modelkit.session import HWMonitor

        with HWMonitor(poll_interval_ms=50) as hw:
            time.sleep(0.2)

        samples = hw.utilization_samples
        assert isinstance(samples, list)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_companion_pattern_with_perfstats(self):
        """HWMonitor works alongside PerfStats."""
        from winml.modelkit.session import HWMonitor

        stats = PerfStats(warmup=2)
        with HWMonitor(poll_interval_ms=50) as hw:
            for _ in range(5):
                stats.record(lambda: time.sleep(0.01))

        assert stats.count == 3
        assert stats.mean_ms > 0
        assert isinstance(hw.to_dict(), dict)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_cpu_metrics_available(self):
        from winml.modelkit.session import HWMonitor

        with HWMonitor(poll_interval_ms=50) as hw:
            time.sleep(0.2)

        assert isinstance(hw.mean_cpu_pct, float)
        assert isinstance(hw.peak_cpu_pct, float)
        assert hw.mean_cpu_pct >= 0.0

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_ram_metrics_available(self):
        from winml.modelkit.session import HWMonitor

        with HWMonitor(poll_interval_ms=50) as hw:
            time.sleep(0.2)

        assert isinstance(hw.ram_used_mb, float)
        assert hw.ram_used_mb > 0.0  # System always uses some RAM
        assert isinstance(hw.peak_ram_used_mb, float)
        assert hw.peak_ram_used_mb >= hw.ram_used_mb  # Peak >= current

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_cpu_samples_accessible(self):
        from winml.modelkit.session import HWMonitor

        with HWMonitor(poll_interval_ms=50) as hw:
            time.sleep(0.2)

        samples = hw.cpu_samples
        assert isinstance(samples, list)
        for val in samples:
            assert isinstance(val, float)


# ============================================================================
# QNNMonitor tests — moved to tests/unit/session/monitor/test_qnn_monitor.py
# (QNNMonitor is no longer a placeholder; it is a full implementation).
# ============================================================================


# ============================================================================
# Import / re-export tests
# ============================================================================


class TestMonitorImports:
    """Verify all monitors are importable from submodules and session."""

    def test_import_hw_monitor_from_submodule(self):
        from winml.modelkit.session import HWMonitor

        assert HWMonitor is not None

    def test_import_qnn_monitor_from_submodule(self):
        from winml.modelkit.session import QNNMonitor

        assert QNNMonitor is not None

    def test_import_hw_monitor_from_session(self):
        from winml.modelkit.session import HWMonitor

        assert HWMonitor is not None

    def test_import_qnn_monitor_from_session(self):
        from winml.modelkit.session import QNNMonitor

        assert QNNMonitor is not None


# ============================================================================
# PdhPoller graceful degradation tests
# ============================================================================


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
class TestPdhPollerGracefulDegradation:
    """Test PdhPoller when NPU is not available."""

    def test_no_npu_returns_zero_metrics(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        # device="cpu" skips NPU/GPU adapter discovery entirely.
        poller = PdhPoller(poll_interval_ms=50, device="cpu")
        poller.start()
        poller.stop()

        assert poller.mean_utilization_pct == 0.0
        assert poller.peak_utilization_pct == 0.0
        assert poller.peak_memory_mb == 0.0
        assert poller.adapter_luid is None
        assert poller.running_time_delta_ns == 0
        assert poller.is_active is False

    def test_no_npu_sample_lists_empty(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        poller = PdhPoller(poll_interval_ms=50, device="cpu")
        poller.start()
        poller.stop()

        assert poller.utilization_samples == []
        assert poller.memory_samples_mb == []
        assert poller.utilization_sample_count == 0
        assert poller.memory_sample_count == 0

    def test_no_npu_still_collects_cpu_and_ram(self):
        """CPU and RAM should still be collected when no NPU is present."""
        from winml.modelkit.session.monitor._pdh import PdhPoller

        poller = PdhPoller(poll_interval_ms=50, device="cpu")
        poller.start()
        time.sleep(0.3)
        poller.stop()

        # CPU and RAM should have samples even without NPU
        assert poller.cpu_sample_count >= 1
        assert poller.mean_cpu_pct >= 0.0
        assert poller.ram_used_mb > 0.0  # System always uses some RAM


# ============================================================================
# Device parameter routing tests (issue #445)
# ============================================================================


class TestResolveAdapterLuid:
    """Verify ORT-first LUID resolution with PDH fallback."""

    def test_invalid_kind_returns_none(self):
        from winml.modelkit.sysinfo.pdh_adapters import resolve_adapter_luid

        assert resolve_adapter_luid("tpu") is None
        assert resolve_adapter_luid("") is None

    def test_prefers_ort_metadata_when_available(self):
        """When ORT publishes LUID metadata that PDH also enumerates, the
        resolver returns the formatted value WITHOUT calling the PDH-only
        fallback helpers."""
        from winml.modelkit.sysinfo import pdh_adapters

        fake_dev = type(
            "FakeEpDevice",
            (),
            {
                "ep_name": "QNNExecutionProvider",
                "device": type(
                    "FakeHwDev",
                    (),
                    {"type": "GPU_TYPE", "metadata": {"LUID": "74191"}},
                )(),
            },
        )()

        fake_ort = type(
            "FakeOrt",
            (),
            {
                "get_ep_devices": lambda: [fake_dev],
                "OrtHardwareDeviceType": type("Types", (), {"NPU": "NPU_TYPE", "GPU": "GPU_TYPE"}),
            },
        )

        # 74191 == 0x121CF; pretend PDH knows this adapter so validation passes.
        fake_pdh = {"0x00000000_0x000121CF": object()}

        with (
            patch.dict("sys.modules", {"onnxruntime": fake_ort}),
            patch.object(pdh_adapters, "enumerate_adapters", return_value=fake_pdh),
            patch.object(pdh_adapters, "discover_gpu_luid") as mock_pdh,
        ):
            luid = pdh_adapters.resolve_adapter_luid("gpu")

        assert luid == "0x00000000_0x000121CF"
        mock_pdh.assert_not_called()

    def test_filters_by_ep_name(self):
        """When ep_name is given, the resolver picks the matching ep_device
        even if another EP appears first in the list."""
        from winml.modelkit.sysinfo import pdh_adapters

        dml_dev = type(
            "FakeEpDevice",
            (),
            {
                "ep_name": "DmlExecutionProvider",
                "device": type(
                    "FakeHwDev",
                    (),
                    {"type": "GPU_TYPE", "metadata": {"LUID": "111"}},
                )(),
            },
        )()
        qnn_dev = type(
            "FakeEpDevice",
            (),
            {
                "ep_name": "QNNExecutionProvider",
                "device": type(
                    "FakeHwDev",
                    (),
                    {"type": "GPU_TYPE", "metadata": {"LUID": "222"}},
                )(),
            },
        )()

        fake_ort = type(
            "FakeOrt",
            (),
            {
                "get_ep_devices": lambda: [dml_dev, qnn_dev],
                "OrtHardwareDeviceType": type("Types", (), {"NPU": "NPU_TYPE", "GPU": "GPU_TYPE"}),
            },
        )

        # PDH must enumerate the matched adapter or validation will reject it.
        # 111 == 0x6F, 222 == 0xDE.
        fake_pdh = {
            "0x00000000_0x0000006F": object(),
            "0x00000000_0x000000DE": object(),
        }

        with (
            patch.dict("sys.modules", {"onnxruntime": fake_ort}),
            patch.object(pdh_adapters, "enumerate_adapters", return_value=fake_pdh),
        ):
            luid = pdh_adapters.resolve_adapter_luid("gpu", ep_name="QNNExecutionProvider")

        assert luid == "0x00000000_0x000000DE"

    def test_falls_back_to_pdh_when_ort_has_no_luid_metadata(self):
        """Some EPs may register without LUID metadata; PDH is the fallback."""
        from winml.modelkit.sysinfo import pdh_adapters

        bare_dev = type(
            "FakeEpDevice",
            (),
            {
                "ep_name": "SomeEP",
                "device": type("FakeHwDev", (), {"type": "GPU_TYPE", "metadata": {}})(),
            },
        )()

        fake_ort = type(
            "FakeOrt",
            (),
            {
                "get_ep_devices": lambda: [bare_dev],
                "OrtHardwareDeviceType": type("Types", (), {"NPU": "NPU_TYPE", "GPU": "GPU_TYPE"}),
            },
        )

        with (
            patch.dict("sys.modules", {"onnxruntime": fake_ort}),
            patch.object(
                pdh_adapters,
                "discover_gpu_luid",
                return_value="0x00000000_0xFEED",
            ),
        ):
            luid = pdh_adapters.resolve_adapter_luid("gpu")

        assert luid == "0x00000000_0xFEED"

    def test_falls_back_when_ort_get_ep_devices_raises(self):
        """A misbehaving ORT build must not break monitoring."""
        from winml.modelkit.sysinfo import pdh_adapters

        def boom():
            raise RuntimeError("autoEP unavailable")

        fake_ort = type(
            "FakeOrt",
            (),
            {
                "get_ep_devices": staticmethod(boom),
                "OrtHardwareDeviceType": type("Types", (), {"NPU": "NPU_TYPE", "GPU": "GPU_TYPE"}),
            },
        )

        with (
            patch.dict("sys.modules", {"onnxruntime": fake_ort}),
            patch.object(
                pdh_adapters,
                "discover_npu_luid",
                return_value="0x00000000_0x00012C89",
            ),
        ):
            luid = pdh_adapters.resolve_adapter_luid("npu")

        assert luid == "0x00000000_0x00012C89"

    def test_falls_back_when_ort_luid_not_in_pdh_enumeration(self):
        """If ORT publishes a LUID PDH doesn't enumerate, the resolver skips
        it and falls through to the PDH-only fallback. Without this guard,
        ``build_adapter_query`` would later raise for the unknown LUID."""
        from winml.modelkit.sysinfo import pdh_adapters

        ghost_dev = type(
            "FakeEpDevice",
            (),
            {
                "ep_name": "SomeEP",
                "device": type(
                    "FakeHwDev",
                    (),
                    {"type": "NPU_TYPE", "metadata": {"LUID": "22278"}},
                )(),
            },
        )()

        fake_ort = type(
            "FakeOrt",
            (),
            {
                "get_ep_devices": lambda: [ghost_dev],
                "OrtHardwareDeviceType": type("Types", (), {"NPU": "NPU_TYPE", "GPU": "GPU_TYPE"}),
            },
        )

        # PDH knows *some* adapters but not the one ORT named. A non-empty
        # dict triggers validation; the ORT LUID is rejected and we fall
        # through to discover_npu_luid().
        fake_pdh = {"0x00000000_0xC0FFEE": object()}

        with (
            patch.dict("sys.modules", {"onnxruntime": fake_ort}),
            patch.object(pdh_adapters, "enumerate_adapters", return_value=fake_pdh),
            patch.object(
                pdh_adapters,
                "discover_npu_luid",
                return_value="0x00000000_0xFA11BACC",
            ) as mock_pdh,
        ):
            luid = pdh_adapters.resolve_adapter_luid("npu")

        # 22278 == 0x5706 (the failing-test LUID); rejected, fallback wins.
        assert luid == "0x00000000_0xFA11BACC"
        mock_pdh.assert_called_once()

    def test_skips_malformed_ep_device(self):
        """Accessing .device or .metadata may raise on bad ep_devices; the
        resolver swallows and continues to the next entry."""
        from winml.modelkit.sysinfo import pdh_adapters

        class _Boom:
            @property
            def device(self):
                raise AttributeError("device unavailable")

            ep_name = "BrokenEP"

        good_dev = type(
            "FakeEpDevice",
            (),
            {
                "ep_name": "QNNExecutionProvider",
                "device": type(
                    "FakeHwDev",
                    (),
                    {"type": "NPU_TYPE", "metadata": {"LUID": "76937"}},
                )(),
            },
        )()

        fake_ort = type(
            "FakeOrt",
            (),
            {
                "get_ep_devices": lambda: [_Boom(), good_dev],
                "OrtHardwareDeviceType": type("Types", (), {"NPU": "NPU_TYPE", "GPU": "GPU_TYPE"}),
            },
        )

        # 76937 == 0x12C89
        fake_pdh = {"0x00000000_0x00012C89": object()}

        with (
            patch.dict("sys.modules", {"onnxruntime": fake_ort}),
            patch.object(pdh_adapters, "enumerate_adapters", return_value=fake_pdh),
        ):
            luid = pdh_adapters.resolve_adapter_luid("npu")

        assert luid == "0x00000000_0x00012C89"


class TestAdapterLabel:
    """The ``adapter_label`` helper centralises chart/status row wording."""

    def test_labels(self):
        from winml.modelkit.session.monitor.hw_monitor import adapter_label

        assert adapter_label("npu") == "NPU"
        assert adapter_label("gpu") == "GPU"
        # Unknown / CPU-only mode falls back to a neutral label rather than
        # claiming an adapter type that isn't being polled.
        assert adapter_label(None) == "Adapter"
        assert adapter_label("cpu") == "Adapter"


class TestPollerDeviceRouting:
    """Verify ``device`` parameter routes PdhPoller to the correct adapter."""

    def test_unknown_device_raises(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        with pytest.raises(ValueError):
            PdhPoller(device="tpu")

    def test_cpu_device_skips_adapter_discovery(self):
        """device='cpu' must not even attempt NPU/GPU LUID discovery."""
        from winml.modelkit.session.monitor._pdh import PdhPoller

        with patch("winml.modelkit.session.monitor._pdh.resolve_adapter_luid") as mock_resolve:
            poller = PdhPoller(poll_interval_ms=50, device="cpu")
            poller.start()
            poller.stop()

        mock_resolve.assert_not_called()
        assert poller.device_kind is None
        assert poller.adapter_luid is None

    def test_gpu_device_uses_3d_engine_query(self):
        """device='gpu' must resolve a GPU LUID + call build_gpu_query."""
        from winml.modelkit.session.monitor._pdh import PdhPoller

        fake_query = type(
            "Q",
            (),
            {
                "open": lambda self: None,
                "add_counter": lambda self, *a, **k: True,
                "prime": lambda self: None,
                "collect": lambda self, **k: {},
                "_collect_once": lambda self: {},
                "close": lambda self: None,
                "counter_names": [],
            },
        )()

        with (
            patch(
                "winml.modelkit.session.monitor._pdh.resolve_adapter_luid",
                return_value="0x00000000_0xDEADBEEF",
            ) as mock_resolve,
            patch(
                "winml.modelkit.session.monitor._pdh.build_gpu_query",
                return_value=fake_query,
            ) as mock_build_gpu,
            patch(
                "winml.modelkit.session.monitor._pdh.build_npu_query",
            ) as mock_build_npu,
        ):
            poller = PdhPoller(poll_interval_ms=50, device="gpu")
            poller.start()
            poller.stop()

        mock_resolve.assert_called_once_with("gpu", ep_name=None)
        mock_build_gpu.assert_called_once()
        mock_build_npu.assert_not_called()
        assert poller.device_kind == "gpu"
        assert poller.adapter_luid == "0x00000000_0xDEADBEEF"

    def test_auto_prefers_npu_then_gpu(self):
        """device='auto' must probe NPU first, then GPU."""
        from winml.modelkit.session.monitor._pdh import PdhPoller

        fake_query = type(
            "Q",
            (),
            {
                "open": lambda self: None,
                "add_counter": lambda self, *a, **k: True,
                "prime": lambda self: None,
                "collect": lambda self, **k: {},
                "_collect_once": lambda self: {},
                "close": lambda self: None,
                "counter_names": [],
            },
        )()

        # First call (npu) returns None; second call (gpu) returns the LUID.
        with (
            patch(
                "winml.modelkit.session.monitor._pdh.resolve_adapter_luid",
                side_effect=[None, "0x0_0xCAFE"],
            ) as mock_resolve,
            patch(
                "winml.modelkit.session.monitor._pdh.build_gpu_query",
                return_value=fake_query,
            ) as mock_build_gpu,
        ):
            poller = PdhPoller(poll_interval_ms=50, device="auto")
            poller.start()
            poller.stop()

        assert mock_resolve.call_args_list == [
            (("npu",), {"ep_name": None}),
            (("gpu",), {"ep_name": None}),
        ]
        mock_build_gpu.assert_called_once()
        assert poller.device_kind == "gpu"
        assert poller.adapter_luid == "0x0_0xCAFE"

    def test_ep_name_threads_through_to_resolver(self):
        """An ep_name on PdhPoller is forwarded to resolve_adapter_luid."""
        from winml.modelkit.session.monitor._pdh import PdhPoller

        fake_query = type(
            "Q",
            (),
            {
                "open": lambda self: None,
                "add_counter": lambda self, *a, **k: True,
                "prime": lambda self: None,
                "collect": lambda self, **k: {},
                "_collect_once": lambda self: {},
                "close": lambda self: None,
                "counter_names": [],
            },
        )()

        with (
            patch(
                "winml.modelkit.session.monitor._pdh.resolve_adapter_luid",
                return_value="0x0_0xC0FFEE",
            ) as mock_resolve,
            patch(
                "winml.modelkit.session.monitor._pdh.build_gpu_query",
                return_value=fake_query,
            ),
        ):
            poller = PdhPoller(
                poll_interval_ms=50,
                device="gpu",
                ep_name="QNNExecutionProvider",
            )
            poller.start()
            poller.stop()

        mock_resolve.assert_called_once_with("gpu", ep_name="QNNExecutionProvider")

    def test_auto_with_neither_npu_nor_gpu_falls_back_to_cpu_ram(self):
        """When auto can't find any adapter, the poller must still come up
        and only collect CPU/RAM — it should not raise."""
        from winml.modelkit.session.monitor._pdh import PdhPoller

        with patch(
            "winml.modelkit.session.monitor._pdh.resolve_adapter_luid",
            return_value=None,
        ) as mock_resolve:
            poller = PdhPoller(poll_interval_ms=50, device="auto")
            poller.start()
            poller.stop()

        assert mock_resolve.call_args_list == [
            (("npu",), {"ep_name": None}),
            (("gpu",), {"ep_name": None}),
        ]
        assert poller.device_kind is None
        assert poller.adapter_luid is None

    def test_resolved_luid_missing_from_pdh_degrades_gracefully(self):
        """If the resolver returns a LUID but build_*_query then raises
        ValueError (LUID not in PDH enumeration), the poller must fall
        through to CPU/RAM-only rather than propagating the exception."""
        from winml.modelkit.session.monitor._pdh import PdhPoller

        with (
            patch(
                "winml.modelkit.session.monitor._pdh.resolve_adapter_luid",
                return_value="0x00000000_0xDEADBEEF",
            ),
            patch(
                "winml.modelkit.session.monitor._pdh.build_npu_query",
                side_effect=ValueError("LUID not found"),
            ) as mock_build,
        ):
            poller = PdhPoller(poll_interval_ms=50, device="npu")
            poller.start()
            poller.stop()

        mock_build.assert_called_once()
        # Adapter slots cleared after the failed build attempt; CPU/RAM
        # collection still works (no exception).
        assert poller.device_kind is None
        assert poller.adapter_luid is None


class TestHWMonitorDeviceRouting:
    """HWMonitor surfaces the correct adapter block for the requested device."""

    def test_to_dict_keeps_aggregate_gpu_and_selected_gpu_separate(self):
        from winml.modelkit.session import HWMonitor

        hw = HWMonitor(device="gpu")
        hw._pdh = type(
            "FakePoller",
            (),
            {
                "device_kind": "gpu",
                "mean_utilization_pct": 91.23,
                "peak_utilization_pct": 98.76,
                "utilization_sample_count": 5,
                "adapter_luid": "0x0_0xCAFE",
                "mean_cpu_pct": 12.34,
                "peak_cpu_pct": 34.56,
                "cpu_sample_count": 7,
                "ram_used_mb": 1024.56,
                "peak_ram_used_mb": 2048.78,
                "mean_gpu_pct": 4.56,
                "peak_gpu_pct": 7.89,
                "gpu_sample_count": 11,
                "gpu_luids": ["0x0_0xBEEF"],
                "peak_memory_local_mb": 256.78,
                "peak_memory_shared_mb": 128.34,
                "running_time_delta_ns": 123456789,
            },
        )()

        d = hw.to_dict()

        assert d["gpu"] == {
            "mean_pct": 4.56,
            "peak_pct": 7.89,
            "sample_count": 11,
            "luids": ["0x0_0xBEEF"],
        }
        assert d["adapter"] == {
            "mean_pct": 91.23,
            "peak_pct": 98.76,
            "sample_count": 5,
        }
        assert "npu" not in d

    def test_to_dict_emits_gpu_block_when_monitoring_gpu(self):
        from winml.modelkit.session import HWMonitor

        fake_query = type(
            "Q",
            (),
            {
                "open": lambda self: None,
                "add_counter": lambda self, *a, **k: True,
                "prime": lambda self: None,
                "collect": lambda self, **k: {},
                "_collect_once": lambda self: {},
                "close": lambda self: None,
                "counter_names": [],
            },
        )()

        with (
            patch(
                "winml.modelkit.session.monitor._pdh.resolve_adapter_luid",
                return_value="0x0_0xCAFE",
            ),
            patch(
                "winml.modelkit.session.monitor._pdh.build_gpu_query",
                return_value=fake_query,
            ),
            HWMonitor(poll_interval_ms=50, device="gpu") as hw,
        ):
            time.sleep(0.05)

        d = hw.to_dict()
        assert d["device_kind"] == "gpu"
        assert "gpu" in d
        # npu block is omitted when the resolved kind isn't "npu".
        assert "npu" not in d
        assert d["adapter_luid"] == "0x0_0xCAFE"
        # JSON-serializable
        assert isinstance(json.dumps(d), str)

    def test_ep_name_threads_through_to_pdh_poller(self):
        """HWMonitor(ep_name=...) reaches the underlying PdhPoller resolver."""
        from winml.modelkit.session import HWMonitor

        fake_query = type(
            "Q",
            (),
            {
                "open": lambda self: None,
                "add_counter": lambda self, *a, **k: True,
                "prime": lambda self: None,
                "collect": lambda self, **k: {},
                "_collect_once": lambda self: {},
                "close": lambda self: None,
                "counter_names": [],
            },
        )()

        with (
            patch(
                "winml.modelkit.session.monitor._pdh.resolve_adapter_luid",
                return_value="0x0_0xC0FFEE",
            ) as mock_resolve,
            patch(
                "winml.modelkit.session.monitor._pdh.build_npu_query",
                return_value=fake_query,
            ),
            HWMonitor(
                poll_interval_ms=50,
                device="npu",
                ep_name="QNNExecutionProvider",
            ),
        ):
            pass

        mock_resolve.assert_called_once_with("npu", ep_name="QNNExecutionProvider")

    def test_unknown_device_raises(self):
        from winml.modelkit.session import HWMonitor

        with pytest.raises(ValueError):
            HWMonitor(device="tpu")


# ============================================================================
# JSON serializability tests
# ============================================================================


class TestToDictJsonSerializable:
    """Verify to_dict() output is JSON-serializable for all monitors."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_hw_monitor_to_dict_json(self):
        from winml.modelkit.session import HWMonitor

        with HWMonitor(poll_interval_ms=50) as hw:
            time.sleep(0.1)
        d = hw.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_vitisai_monitor_to_dict_json(self):
        with VitisAIMonitor() as hw:
            time.sleep(0.1)
        d = hw.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_qnn_monitor_result_dict_json(self):
        """v2.4: QNN exposes its data via the typed result accessor.

        ``to_dict()`` was removed from the WinMLEPMonitor contract; consumers
        access ``OpTraceResult`` via ``monitor.result`` and serialize via
        ``result.to_dict()``.
        """
        from winml.modelkit.session import QNNMonitor

        with QNNMonitor() as hw:
            pass
        # Post-exit: the monitor populated _result with a failure-shape
        # OpTraceResult (status="no_data" — no CSV produced in this unit
        # test). The typed accessor returns it; to_dict() on the result
        # must be JSON-serializable.
        assert hw.result is not None
        d = hw.result.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ============================================================================
# Exception safety tests
# ============================================================================


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
class TestMonitorExceptionSafety:
    """Verify monitors clean up properly when exceptions occur."""

    def test_hw_monitor_cleans_up_on_exception(self):
        from winml.modelkit.session import HWMonitor

        monitor = HWMonitor(poll_interval_ms=50)
        with pytest.raises(RuntimeError), monitor:
            raise RuntimeError("simulated error")

        # After exception, thread should be stopped
        assert monitor._pdh.is_active is False

    def test_vitisai_monitor_cleans_up_on_exception(self):
        monitor = VitisAIMonitor()
        with pytest.raises(RuntimeError), monitor:
            raise RuntimeError("simulated error")

        # VitisAI has no background thread — just verify it exited cleanly
        assert monitor.command_submissions == 0


# ============================================================================
# LiveMonitorDisplay tests
# ============================================================================


class TestAvgNow:
    """Contract tests for the ``_avg_now`` helper in ``_live_chart``.

    Pins the ``(avg, now)`` semantics that :meth:`_render_status` relies
    on for the unified ``now%/avg%`` display across NPU / CPU / GPU.
    """

    def test_returns_mean_and_last_for_multi_sample_list(self):
        from winml.modelkit.commands._live_chart import _avg_now

        assert _avg_now([10.0, 20.0]) == (15.0, 20.0)

    def test_returns_single_value_as_both_when_list_has_one_sample(self):
        from winml.modelkit.commands._live_chart import _avg_now

        assert _avg_now([42.5]) == (42.5, 42.5)

    def test_empty_list_returns_fallback_for_both(self):
        from winml.modelkit.commands._live_chart import _avg_now

        assert _avg_now([], fallback_now=42.0) == (42.0, 42.0)

    def test_none_returns_fallback_for_both(self):
        from winml.modelkit.commands._live_chart import _avg_now

        assert _avg_now(None, fallback_now=5.0) == (5.0, 5.0)

    def test_empty_list_defaults_to_zero_fallback(self):
        from winml.modelkit.commands._live_chart import _avg_now

        assert _avg_now([]) == (0.0, 0.0)

    def test_all_zero_samples_return_zero_zero_regardless_of_fallback(self):
        """Non-empty samples override ``fallback_now`` entirely."""
        from winml.modelkit.commands._live_chart import _avg_now

        assert _avg_now([0.0, 0.0, 0.0], fallback_now=99.0) == (0.0, 0.0)


class TestLiveMonitorDisplay:
    """Test LiveMonitorDisplay logic (non-visual)."""

    def test_render_status_warmup_phase(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(total_iterations=110, warmup=10, model_id="test", device="npu")
        status = display._render_status(
            iteration=5,
            latency_ms=1.0,
            util_samples=[50.0],
            memory_local_mb=10.0,
            memory_shared_mb=20.0,
            cpu_pct=5.0,
            ram_mb=8000.0,
        )
        assert "Warmup" in status
        assert "npu" in status.lower() or "Device" in status

    def test_render_status_benchmark_phase(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(total_iterations=110, warmup=10, model_id="test", device="npu")
        status = display._render_status(
            iteration=50,
            latency_ms=2.0,
            util_samples=[80.0, 90.0],
            memory_local_mb=31.0,
            memory_shared_mb=43.0,
            cpu_pct=15.0,
            ram_mb=40000.0,
        )
        assert "Iter" in status
        assert "Throughput" in status
        assert "Latency" in status

    def test_render_status_zero_latency_no_crash(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(total_iterations=10, warmup=0, model_id="test", device="cpu")
        # latency_ms=0 should not cause division by zero
        status = display._render_status(
            iteration=1,
            latency_ms=0.0,
            util_samples=[],
        )
        assert "Throughput" in status

    def test_render_status_empty_samples(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(total_iterations=10, warmup=0, model_id="test", device="cpu")
        status = display._render_status(
            iteration=1,
            latency_ms=1.0,
            util_samples=[],
        )
        assert "0.0%" in status  # NPU should show 0.0%

    def test_update_noop_when_live_is_none(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(total_iterations=10, warmup=0, model_id="test", device="cpu")
        # _live is None (not entered context) — should not crash
        display.update(
            iteration=1,
            latency_ms=1.0,
            util_samples=[50.0],
        )

    def test_print_final_snapshot_is_noop(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(total_iterations=10, warmup=0, model_id="test", device="cpu")
        # Should not crash or print anything
        display.print_final_snapshot(
            util_samples=[50.0],
            memory_mb=10.0,
            latency_ms=1.0,
            hw_dict={},
        )

    def test_render_status_includes_gpu_cell(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(total_iterations=110, warmup=10, model_id="test", device="gpu")
        status = display._render_status(
            iteration=50,
            latency_ms=2.0,
            util_samples=[80.0],
            cpu_pct=15.0,
            gpu_pct=42.5,
        )
        assert "GPU (aggregate):" in status
        assert "42.5" in status

    def test_render_status_uses_resolved_adapter_label_for_auto_gpu(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(
            total_iterations=110,
            warmup=10,
            model_id="test",
            device="auto",
            device_kind="gpu",
        )
        status = display._render_status(
            iteration=50,
            latency_ms=2.0,
            util_samples=[80.0],
            cpu_pct=15.0,
            gpu_pct=42.5,
        )

        assert "Device: GPU" in status
        assert "Device: auto" not in status

    def test_render_status_distinguishes_selected_and_aggregate_gpu_labels(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(total_iterations=110, warmup=10, model_id="test", device="gpu")
        status = display._render_status(
            iteration=50,
            latency_ms=2.0,
            util_samples=[80.0, 90.0],
            cpu_pct=15.0,
            gpu_pct=42.5,
            gpu_samples=[40.0, 45.0],
        )

        assert "GPU (selected): 90.0%/85.0%" in status
        assert "GPU (aggregate): 45.0%/42.5%" in status

    def test_render_status_does_not_label_gpu_adapter_as_npu(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(total_iterations=110, warmup=10, model_id="test", device="gpu")
        status = display._render_status(
            iteration=50,
            latency_ms=2.0,
            util_samples=[80.0, 90.0],
            cpu_pct=15.0,
            gpu_pct=42.5,
            gpu_samples=[42.5],
        )

        assert "NPU:" not in status

    def test_render_chart_does_not_label_gpu_adapter_legend_as_npu(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        fake_plotext = type(
            "FakePlotext",
            (),
            {
                "clf": lambda self: None,
                "theme": lambda self, *args, **kwargs: None,
                "plot": lambda self, *args, **kwargs: None,
                "ylabel": lambda self, *args, **kwargs: None,
                "ylim": lambda self, *args, **kwargs: None,
                "yticks": lambda self, *args, **kwargs: None,
                "xlim": lambda self, *args, **kwargs: None,
                "xlabel": lambda self, *args, **kwargs: None,
                "plotsize": lambda self, *args, **kwargs: None,
                "build": lambda self: "chart",
            },
        )()

        display = LiveMonitorDisplay(total_iterations=110, warmup=10, model_id="test", device="gpu")
        with patch.dict(sys.modules, {"plotext": fake_plotext}):
            renderable = display._render_chart(
                util_samples=[80.0, 90.0],
                cpu_samples=[15.0],
                gpu_samples=[42.5],
            )

        title = renderable.renderables[0].plain
        assert "NPU %" not in title

    def test_render_chart_distinguishes_selected_and_aggregate_gpu_labels(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        fake_plotext = type(
            "FakePlotext",
            (),
            {
                "clf": lambda self: None,
                "theme": lambda self, *args, **kwargs: None,
                "plot": lambda self, *args, **kwargs: None,
                "ylabel": lambda self, *args, **kwargs: None,
                "ylim": lambda self, *args, **kwargs: None,
                "yticks": lambda self, *args, **kwargs: None,
                "xlim": lambda self, *args, **kwargs: None,
                "xlabel": lambda self, *args, **kwargs: None,
                "plotsize": lambda self, *args, **kwargs: None,
                "build": lambda self: "chart",
            },
        )()

        display = LiveMonitorDisplay(total_iterations=110, warmup=10, model_id="test", device="gpu")
        with patch.dict(sys.modules, {"plotext": fake_plotext}):
            renderable = display._render_chart(
                util_samples=[80.0, 90.0],
                cpu_samples=[15.0],
                gpu_samples=[42.5],
            )

        title = renderable.renderables[0].plain
        assert "GPU (selected) %" in title
        assert "GPU (aggregate) %" in title

    def test_render_status_cpu_only_omits_selected_adapter_cell(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(total_iterations=10, warmup=0, model_id="test", device="cpu")
        status = display._render_status(
            iteration=1,
            latency_ms=1.0,
            util_samples=[],
            cpu_pct=12.0,
            cpu_samples=[10.0, 12.0],
            gpu_pct=20.0,
            gpu_samples=[18.0, 20.0],
        )

        assert "Adapter:" not in status
        assert status.splitlines()[1].lstrip().startswith("CPU:")

    def test_update_accepts_gpu_samples_noop_when_live_none(self):
        from winml.modelkit.commands._live_chart import LiveMonitorDisplay

        display = LiveMonitorDisplay(total_iterations=10, warmup=0, model_id="test", device="gpu")
        # _live is None — should accept gpu kwargs without crashing
        display.update(
            iteration=1,
            latency_ms=1.0,
            util_samples=[50.0],
            cpu_samples=[20.0],
            gpu_samples=[33.0],
            gpu_pct=33.0,
        )


# ============================================================================
# GPU utilization aggregation (hardware-independent)
# ============================================================================


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only (_pdh import)")
class TestGpuUtilizationAggregation:
    """Test the pure GPU-engine utilization aggregation helper.

    Matches Task Manager's "busiest engine" semantics: max across engine
    utilization values, capped at 100. No hardware required.
    """

    def test_max_across_engines(self):
        from winml.modelkit.session.monitor._pdh import aggregate_gpu_utilization

        assert aggregate_gpu_utilization([10.0, 80.0, 5.0]) == 80.0

    def test_caps_at_100(self):
        from winml.modelkit.session.monitor._pdh import aggregate_gpu_utilization

        assert aggregate_gpu_utilization([120.0, 50.0]) == 100.0

    def test_ignores_none_values(self):
        from winml.modelkit.session.monitor._pdh import aggregate_gpu_utilization

        assert aggregate_gpu_utilization([None, 30.0, None]) == 30.0

    def test_all_none_returns_none(self):
        from winml.modelkit.session.monitor._pdh import aggregate_gpu_utilization

        assert aggregate_gpu_utilization([None, None]) is None

    def test_empty_returns_none(self):
        from winml.modelkit.session.monitor._pdh import aggregate_gpu_utilization

        assert aggregate_gpu_utilization([]) is None


# ============================================================================
# PdhPoller GPU monitoring
# ============================================================================


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
class TestPdhPollerGpu:
    """Test GPU monitoring in PdhPoller."""

    def test_is_gpu_available_returns_bool(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        assert isinstance(PdhPoller.is_gpu_available(), bool)

    def test_gpu_properties_have_correct_types(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        poller = PdhPoller(poll_interval_ms=50)
        poller.start()
        time.sleep(0.2)
        poller.stop()

        assert isinstance(poller.gpu_samples, list)
        for val in poller.gpu_samples:
            assert isinstance(val, float)
        assert isinstance(poller.mean_gpu_pct, float)
        assert isinstance(poller.peak_gpu_pct, float)
        assert isinstance(poller.gpu_luids, list)

    def test_no_gpu_returns_zero_metrics(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        with patch("winml.modelkit.session.monitor._pdh.discover_gpu_luids", return_value=[]):
            poller = PdhPoller(poll_interval_ms=50)
            poller.start()
            time.sleep(0.2)
            poller.stop()

        assert poller.gpu_samples == []
        assert poller.gpu_luids == []
        assert poller.mean_gpu_pct == 0.0
        assert poller.peak_gpu_pct == 0.0

    def test_no_gpu_still_collects_cpu(self):
        """CPU collection unaffected when no GPU present."""
        from winml.modelkit.session.monitor._pdh import PdhPoller

        with patch("winml.modelkit.session.monitor._pdh.discover_gpu_luids", return_value=[]):
            poller = PdhPoller(poll_interval_ms=50)
            poller.start()
            time.sleep(0.3)
            poller.stop()

        assert poller.cpu_sample_count >= 1


# ============================================================================
# HWMonitor GPU surface
# ============================================================================


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
class TestHWMonitorGpu:
    """Test GPU metrics exposed by HWMonitor."""

    def test_gpu_properties_accessible(self):
        from winml.modelkit.session import HWMonitor

        with HWMonitor(poll_interval_ms=50) as hw:
            time.sleep(0.2)

        assert isinstance(hw.gpu_samples, list)
        assert isinstance(hw.mean_gpu_pct, float)
        assert isinstance(hw.peak_gpu_pct, float)

    def test_to_dict_has_gpu_section(self):
        from winml.modelkit.session import HWMonitor

        with HWMonitor(poll_interval_ms=50) as hw:
            time.sleep(0.1)

        d = hw.to_dict()
        assert "gpu" in d
        assert "mean_pct" in d["gpu"]
        assert "peak_pct" in d["gpu"]
        assert "sample_count" in d["gpu"]
        # Must remain JSON-serializable
        json.dumps(d)
