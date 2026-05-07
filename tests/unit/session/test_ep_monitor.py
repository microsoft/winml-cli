# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for EPMonitor, VitisAIMonitor, and internal helpers."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from winml.modelkit.session import EPMonitor, PerfStats, VitisAIMonitor


# ============================================================================
# EPMonitor (ABC) tests
# ============================================================================


class TestEPMonitor:
    """Test EPMonitor abstract base class."""

    def test_cannot_instantiate(self):
        """EPMonitor is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EPMonitor()

    def test_subclass_must_implement_all_methods(self):
        """Concrete subclass missing methods raises TypeError."""

        class IncompleteMonitor(EPMonitor):
            pass

        with pytest.raises(TypeError):
            IncompleteMonitor()

    def test_concrete_subclass_works(self):
        """A fully-implemented subclass can be instantiated."""

        class DummyMonitor(EPMonitor):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def to_dict(self):
                return {"test": True}

            @classmethod
            def is_available(cls):
                return True

        mon = DummyMonitor()
        assert mon.to_dict() == {"test": True}
        assert DummyMonitor.is_available() is True

    def test_null_ep_monitor(self):
        """NullEPMonitor implements Null Object Pattern correctly."""
        from winml.modelkit.session import NullEPMonitor

        mon = NullEPMonitor()
        assert NullEPMonitor.is_available() is True
        assert mon.to_dict() == {}

        # Context manager works
        with mon as m:
            assert m is mon

        # JSON-serializable
        serialized = json.dumps(mon.to_dict())
        assert serialized == "{}"


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
        assert "utilization_pct" in query.counter_names
        assert "running_time_ns" in query.counter_names
        assert "memory_local_bytes" in query.counter_names
        assert "memory_shared_bytes" in query.counter_names
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

    def test_npu_luid_populated_after_start(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller
        from winml.modelkit.sysinfo.pdh_adapters import discover_npu_luid

        luid = discover_npu_luid()
        if luid is None:
            pytest.skip("No NPU detected")

        poller = PdhPoller(poll_interval_ms=50)
        poller.start()
        poller.stop()
        assert poller.npu_luid == luid

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
        # NPU section
        assert "npu" in d
        assert "mean_pct" in d["npu"]
        assert "peak_pct" in d["npu"]
        assert "sample_count" in d["npu"]
        # Device memory + running time
        assert "device_memory" in d
        assert "local_peak_mb" in d["device_memory"]
        assert "shared_peak_mb" in d["device_memory"]
        assert "running_time_ns" in d
        assert "npu_luid" in d

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
# QNNMonitor tests (placeholder)
# ============================================================================


class TestQNNMonitor:
    """Test QNNMonitor placeholder."""

    def test_is_available_returns_false(self):
        from winml.modelkit.session import QNNMonitor

        assert QNNMonitor.is_available() is False

    def test_context_manager_noop(self):
        from winml.modelkit.session import QNNMonitor

        with QNNMonitor() as hw:
            pass

        assert hw.to_dict()["ep"] == "QNN"

    def test_to_dict_returns_stub(self):
        from winml.modelkit.session import QNNMonitor

        with QNNMonitor() as hw:
            pass

        d = hw.to_dict()
        assert d["ep"] == "QNN"
        assert d["device"] == "NPU"
        assert d["status"] == "not_implemented"


# ============================================================================
# OpenVinoMonitor tests (placeholder)
# ============================================================================


class TestOpenVinoMonitor:
    """Test OpenVinoMonitor placeholder."""

    def test_is_available_returns_false(self):
        from winml.modelkit.session import OpenVinoMonitor

        assert OpenVinoMonitor.is_available() is False

    def test_context_manager_noop(self):
        from winml.modelkit.session import OpenVinoMonitor

        with OpenVinoMonitor() as hw:
            pass

        assert hw.to_dict()["ep"] == "OpenVINO"

    def test_to_dict_returns_stub(self):
        from winml.modelkit.session import OpenVinoMonitor

        with OpenVinoMonitor() as hw:
            pass

        d = hw.to_dict()
        assert d["ep"] == "OpenVINO"
        assert d["device"] == "NPU"
        assert d["status"] == "not_implemented"


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

    def test_import_openvino_monitor_from_submodule(self):
        from winml.modelkit.session import OpenVinoMonitor

        assert OpenVinoMonitor is not None

    def test_import_hw_monitor_from_session(self):
        from winml.modelkit.session import HWMonitor

        assert HWMonitor is not None

    def test_import_qnn_monitor_from_session(self):
        from winml.modelkit.session import QNNMonitor

        assert QNNMonitor is not None

    def test_import_openvino_monitor_from_session(self):
        from winml.modelkit.session import OpenVinoMonitor

        assert OpenVinoMonitor is not None


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
        assert poller.npu_luid is None
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


class TestPollerDeviceRouting:
    """Verify ``device`` parameter routes PdhPoller to the correct adapter."""

    def test_unknown_device_raises(self):
        from winml.modelkit.session.monitor._pdh import PdhPoller

        with pytest.raises(ValueError):
            PdhPoller(device="tpu")

    def test_cpu_device_skips_adapter_discovery(self):
        """device='cpu' must not even attempt NPU/GPU LUID discovery."""
        from winml.modelkit.session.monitor._pdh import PdhPoller

        with (
            patch("winml.modelkit.session.monitor._pdh.discover_npu_luid") as mock_npu,
            patch("winml.modelkit.session.monitor._pdh.discover_gpu_luid") as mock_gpu,
        ):
            poller = PdhPoller(poll_interval_ms=50, device="cpu")
            poller.start()
            poller.stop()

        mock_npu.assert_not_called()
        mock_gpu.assert_not_called()
        assert poller.device_kind is None
        assert poller.adapter_luid is None

    def test_gpu_device_uses_3d_engine_query(self):
        """device='gpu' must call discover_gpu_luid + build_gpu_query."""
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
                "winml.modelkit.session.monitor._pdh.discover_gpu_luid",
                return_value="0x00000000_0xDEADBEEF",
            ),
            patch(
                "winml.modelkit.session.monitor._pdh.build_gpu_query",
                return_value=fake_query,
            ) as mock_build_gpu,
            patch(
                "winml.modelkit.session.monitor._pdh.build_npu_query",
            ) as mock_build_npu,
            patch(
                "winml.modelkit.session.monitor._pdh.discover_npu_luid",
                return_value=None,
            ),
        ):
            poller = PdhPoller(poll_interval_ms=50, device="gpu")
            poller.start()
            poller.stop()

        mock_build_gpu.assert_called_once()
        mock_build_npu.assert_not_called()
        assert poller.device_kind == "gpu"
        assert poller.adapter_luid == "0x00000000_0xDEADBEEF"
        # npu_luid is the back-compat property; should be None for GPU mode.
        assert poller.npu_luid is None

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

        with (
            patch(
                "winml.modelkit.session.monitor._pdh.discover_npu_luid",
                return_value=None,
            ),
            patch(
                "winml.modelkit.session.monitor._pdh.discover_gpu_luid",
                return_value="0x0_0xCAFE",
            ),
            patch(
                "winml.modelkit.session.monitor._pdh.build_gpu_query",
                return_value=fake_query,
            ) as mock_build_gpu,
        ):
            poller = PdhPoller(poll_interval_ms=50, device="auto")
            poller.start()
            poller.stop()

        mock_build_gpu.assert_called_once()
        assert poller.device_kind == "gpu"
        assert poller.adapter_luid == "0x0_0xCAFE"


class TestHWMonitorDeviceRouting:
    """HWMonitor surfaces the correct adapter block for the requested device."""

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
                "winml.modelkit.session.monitor._pdh.discover_gpu_luid",
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
        assert "npu" in d  # back-compat: always present, zeros here
        assert d["npu"]["mean_pct"] == 0.0
        assert d["adapter_luid"] == "0x0_0xCAFE"
        # JSON-serializable
        assert isinstance(json.dumps(d), str)

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

    def test_qnn_monitor_to_dict_json(self):
        from winml.modelkit.session import QNNMonitor

        with QNNMonitor() as hw:
            pass
        d = hw.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_openvino_monitor_to_dict_json(self):
        from winml.modelkit.session import OpenVinoMonitor

        with OpenVinoMonitor() as hw:
            pass
        d = hw.to_dict()
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
