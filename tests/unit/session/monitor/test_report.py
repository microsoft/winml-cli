# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for op-tracing console report and JSON file output."""

import json
from io import StringIO

import pytest
from rich.console import Console

from winml.modelkit.session.monitor.op_metrics import (
    OperatorMetrics,
    OpTraceResult,
)
from winml.modelkit.session.monitor.report import (
    _format_bytes,  # Testing internal implementation
    display_op_trace_report,
    write_op_trace_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_basic_result() -> OpTraceResult:
    """Create a basic-mode OpTraceResult with sample operators."""
    return OpTraceResult(
        model="resnet-50",
        device="npu",
        tracing_level="basic",
        ep="QNNExecutionProvider",
        num_samples=5,
        summary={
            "hvx_threads": 4,
            "accel_execute_us": 1077,
        },
        operators=[
            OperatorMetrics(
                name="Conv2d",
                op_path="/resnet/embedder/conv/Conv",
                duration_us=671.293,
                percent_of_total=17.9,
            ),
            OperatorMetrics(
                name="Transpose",
                op_path="Transpose",
                duration_us=277.418,
                percent_of_total=7.4,
            ),
            OperatorMetrics(
                name="Add",
                op_path="/resnet/layer1/add/Add",
                duration_us=120.5,
                percent_of_total=3.2,
            ),
        ],
    )


def _make_detail_result() -> OpTraceResult:
    """Create a detail-mode OpTraceResult with memory/cache metrics."""
    return OpTraceResult(
        model="resnet-50",
        device="npu",
        tracing_level="detail",
        ep="QNNExecutionProvider",
        tracing_backend="QHAS",
        num_samples=5,
        summary={
            "inference_us": 1343,
            "execute_us": 1132,
            "utilization_pct": 99.6,
            "dram_read_bytes": 25_400_000,
            "dram_write_bytes": 4_096,
            "vtcm_peak_bytes": 6_300_000,
        },
        operators=[
            OperatorMetrics(
                name="Conv2d",
                op_path="/resnet/layer1/conv/Conv",
                duration_us=478.8,
                percent_of_total=17.9,
                dram_read_bytes=18_432,
                vtcm_hit_ratio=0.999,
            ),
            OperatorMetrics(
                name="LayerNorm",
                op_path="/resnet/norm/LayerNorm",
                duration_us=200.0,
                percent_of_total=7.5,
                dram_read_bytes=8_192,
                vtcm_hit_ratio=0.95,
            ),
        ],
    )


def _make_empty_result() -> OpTraceResult:
    """Create an OpTraceResult with no operators."""
    return OpTraceResult(
        model="empty-model",
        device="npu",
        tracing_level="basic",
    )


# ---------------------------------------------------------------------------
# display_op_trace_report — basic mode
# ---------------------------------------------------------------------------


class TestDisplayBasicReport:
    def test_renders_without_error(self):
        """Basic report renders without raising."""
        result = _make_basic_result()
        console = Console(file=StringIO(), width=120)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert len(output) > 0

    def test_contains_tracing_level(self):
        """Output mentions 'basic' tracing level."""
        result = _make_basic_result()
        console = Console(file=StringIO(), width=120)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert "basic" in output.lower()

    def test_contains_operator_names(self):
        """Output includes operator op_path values."""
        result = _make_basic_result()
        console = Console(file=StringIO(), width=120)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert "Conv" in output
        assert "Transpose" in output

    def test_contains_summary_fields(self):
        """Output includes summary metadata like HVX threads and samples."""
        result = _make_basic_result()
        console = Console(file=StringIO(), width=120)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert "4" in output  # hvx_threads
        assert "1,077" in output  # accel_execute_us (comma-formatted)
        assert "5" in output  # num_samples

    def test_top_n_limits_rows(self):
        """Only top_n operators appear in the table."""
        result = _make_basic_result()
        console = Console(file=StringIO(), width=120)
        display_op_trace_report(result, console=console, top_n=2)
        output = console.file.getvalue()
        # Third operator should not appear
        assert "layer1/add" not in output

    def test_default_console_created(self):
        """When console=None, function creates its own Console and runs."""
        result = _make_basic_result()
        # Should not raise
        display_op_trace_report(result, console=None)


# ---------------------------------------------------------------------------
# display_op_trace_report — detail mode
# ---------------------------------------------------------------------------


class TestDisplayDetailReport:
    def test_renders_without_error(self):
        """Detail report renders without raising."""
        result = _make_detail_result()
        console = Console(file=StringIO(), width=140)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert len(output) > 0

    def test_contains_tracing_level(self):
        """Output mentions 'detail' tracing level."""
        result = _make_detail_result()
        console = Console(file=StringIO(), width=140)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert "detail" in output.lower()

    def test_contains_backend_name(self):
        """Output includes tracing backend (e.g., QHAS)."""
        result = _make_detail_result()
        console = Console(file=StringIO(), width=140)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert "QHAS" in output

    def test_contains_detail_columns(self):
        """Detail table includes memory columns like DRAM and VTCM."""
        result = _make_detail_result()
        console = Console(file=StringIO(), width=140)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert "DRAM" in output
        assert "VTCM" in output

    def test_contains_summary_metrics(self):
        """Detail output includes inference/execute times and utilization."""
        result = _make_detail_result()
        console = Console(file=StringIO(), width=140)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert "1,343" in output  # inference_us (comma-formatted)
        assert "1,132" in output  # execute_us (comma-formatted)
        assert "99.6" in output  # utilization_pct

    def test_vtcm_hit_ratio_displayed(self):
        """VTCM hit ratio is rendered as percentage."""
        result = _make_detail_result()
        console = Console(file=StringIO(), width=140)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert "99.9" in output  # 0.999 -> 99.9%


# ---------------------------------------------------------------------------
# display_op_trace_report — empty operators
# ---------------------------------------------------------------------------


class TestDisplayEmptyReport:
    def test_empty_operators_renders(self):
        """Report with no operators renders without error."""
        result = _make_empty_result()
        console = Console(file=StringIO(), width=120)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert "basic" in output.lower()

    def test_empty_operators_shows_no_data_message(self):
        """Report with no operators shows an informational message."""
        result = _make_empty_result()
        console = Console(file=StringIO(), width=120)
        display_op_trace_report(result, console=console)
        output = console.file.getvalue()
        assert "no operator" in output.lower() or "0" in output


# ---------------------------------------------------------------------------
# write_op_trace_json
# ---------------------------------------------------------------------------


class TestWriteOpTraceJson:
    def test_creates_file(self, tmp_path):
        """JSON file is created at the specified path."""
        result = _make_basic_result()
        out = tmp_path / "test_op_trace.json"
        write_op_trace_json(result, out)
        assert out.exists()

    def test_valid_json(self, tmp_path):
        """Output file contains valid JSON."""
        result = _make_basic_result()
        out = tmp_path / "test_op_trace.json"
        write_op_trace_json(result, out)
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

    def test_metadata_preserved(self, tmp_path):
        """Metadata fields are preserved in JSON output."""
        result = _make_basic_result()
        out = tmp_path / "test_op_trace.json"
        write_op_trace_json(result, out)
        data = json.loads(out.read_text())
        assert data["metadata"]["model"] == "resnet-50"
        assert data["metadata"]["device"] == "npu"
        assert data["metadata"]["tracing_level"] == "basic"

    def test_operators_preserved(self, tmp_path):
        """Operator data is preserved in JSON output."""
        result = _make_basic_result()
        out = tmp_path / "test_op_trace.json"
        write_op_trace_json(result, out)
        data = json.loads(out.read_text())
        assert len(data["operators"]) == 3
        assert data["operators"][0]["name"] == "Conv2d"

    def test_creates_parent_directories(self, tmp_path):
        """Parent directories are created if they don't exist."""
        result = _make_basic_result()
        out = tmp_path / "nested" / "dir" / "output.json"
        write_op_trace_json(result, out)
        assert out.exists()

    def test_detail_result_json(self, tmp_path):
        """Detail-mode result serializes correctly."""
        result = _make_detail_result()
        out = tmp_path / "detail.json"
        write_op_trace_json(result, out)
        data = json.loads(out.read_text())
        assert data["metadata"]["tracing_level"] == "detail"
        assert data["operators"][0]["dram_read_bytes"] == 18_432

    def test_empty_result_json(self, tmp_path):
        """Empty result serializes to valid JSON with empty operators."""
        result = _make_empty_result()
        out = tmp_path / "empty.json"
        write_op_trace_json(result, out)
        data = json.loads(out.read_text())
        assert data["operators"] == []


# ---------------------------------------------------------------------------
# _format_bytes
# ---------------------------------------------------------------------------


class TestFormatBytes:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "0"),
            (0, "0"),
            (500, "500 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1_048_576, "1.0 MB"),
            (1_073_741_824, "1.0 GB"),
            (1_099_511_627_776, "1.0 TB"),
        ],
    )
    def test_format_bytes(self, value, expected):
        assert _format_bytes(value) == expected

    def test_small_integer_no_decimal(self):
        """Small integer byte counts should not have decimal points."""
        result = _format_bytes(42)
        assert result == "42 B"
        assert "." not in result
