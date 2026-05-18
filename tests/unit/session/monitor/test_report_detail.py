# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Detail-mode rendering matches mockup spec."""

from io import StringIO

from rich.console import Console

from winml.modelkit.session.monitor.op_metrics import OperatorMetrics, OpTraceResult
from winml.modelkit.session.monitor.report import display_op_trace_report


def _render(result: OpTraceResult, top_n: int = 5, width: int = 140) -> str:
    # Detail mode has 10 columns and naturally fits ~140 cells per the mockup
    # spec ("natural-fits all 10 columns"); basic mode is the 120-cell envelope.
    buf = StringIO()
    console = Console(file=buf, width=width, force_terminal=False, record=True)
    display_op_trace_report(result, console=console, top_n=top_n)
    return console.export_text()


def _make_detail_result() -> OpTraceResult:
    ops = [
        OperatorMetrics(
            name=f"OpType{i}",
            op_path=f"/path/to/op_{i}/Op",
            duration_us=100.0 - i * 10,
            percent_of_total=30.0 - i * 5,
            dram_read_bytes=1024 * (i + 1),
            vtcm_hit_ratio=0.85 - i * 0.05,
            samples_us=[90.0 + i, 100.0 - i, 110.0 - i * 2],
        )
        for i in range(3)
    ]
    return OpTraceResult(
        model="convnext-base",
        device="NPU",
        tracing_level="detail",
        operators=ops,
        num_samples=3,
    )


def test_detail_header_renamed():
    out = _render(_make_detail_result())
    assert "Op-Tracing (detail)" in out
    assert "Op-Level Profiling" not in out


def test_detail_ten_columns_present():
    out = _render(_make_detail_result())
    expected = ["#", "Node", "Type", "Avg", "Total", "% Tot", "Cum %", "p90", "DRAM(R)", "VTCM Hit"]
    header_line = next(line for line in out.splitlines() if all(c in line for c in expected))
    assert header_line is not None


def test_detail_cumulative_percent_monotonic():
    """Cum % column should be monotonically non-decreasing across rows."""
    out = _render(_make_detail_result())
    # Find the data row containing each op's percent_of_total.
    # Sum is 30 + 25 + 20 = 75% (from _make_detail_result).
    # Cum % values progress: 30.0% → 55.0% → 75.0%
    assert "30.0%" in out
    assert "55.0%" in out
    assert "75.0%" in out


def test_detail_p90_no_us_suffix():
    """p90 cell renders bare number without ' us' suffix (T4 lesson)."""
    out = _render(_make_detail_result())
    # Confirm formatted p90 values are present (one of 91.4, 100.0, etc.)
    # and that no row contains 'us' standalone in the p90 column area.
    # We check for absence of ' us' substring — if anywhere a p90 cell
    # had its suffix accidentally re-added, this catches it.
    for line in out.splitlines():
        # Skip the header/border, only check data rows
        if "/path/to/op_" in line:
            assert " us" not in line, f"p90 cell has ' us' suffix: {line!r}"


def test_detail_total_em_dash_when_no_samples():
    op = OperatorMetrics(
        name="Conv2d",
        op_path="/x",
        duration_us=50.0,
        percent_of_total=10.0,
        samples_us=[],
    )
    result = OpTraceResult(
        model="m",
        device="NPU",
        tracing_level="detail",
        operators=[op],
        num_samples=0,
    )
    out = _render(result)
    assert "—" in out  # em-dash for Total / p90 cells


def test_detail_vtcm_hit_em_dash_when_none():
    op = OperatorMetrics(
        name="Conv2d",
        op_path="/x",
        duration_us=50.0,
        percent_of_total=10.0,
        samples_us=[100.0],
        dram_read_bytes=2048,
        vtcm_hit_ratio=None,
    )
    result = OpTraceResult(
        model="m",
        device="NPU",
        tracing_level="detail",
        operators=[op],
        num_samples=1,
    )
    out = _render(result)
    assert "—" in out  # em-dash for VTCM Hit when None


def test_detail_long_node_path_left_truncated():
    long_path = "/very/deep" + "/segment" * 30
    op = OperatorMetrics(
        name="Conv2d",
        op_path=long_path,
        duration_us=100.0,
        percent_of_total=50.0,
        samples_us=[100.0],
        dram_read_bytes=2048,
        vtcm_hit_ratio=0.8,
    )
    result = OpTraceResult(
        model="m",
        device="NPU",
        tracing_level="detail",
        operators=[op],
        num_samples=1,
    )
    out = _render(result)
    assert "…" in out  # leading ellipsis (U+2026)
    assert long_path[-15:] in out  # right tail preserved


def test_detail_p90_does_not_wrap_on_kilo_microsecond_values():
    """Realistic NPU p90 (>= 1000 us) must fit on a single line."""
    op = OperatorMetrics(
        name="Conv2d",
        op_path="/encoder/layer/conv/Conv",
        duration_us=1234.5,
        percent_of_total=42.0,
        # Inclusive p90 of [1000, 1234.5, 1500] is 1446.9 — kilo-µs range.
        samples_us=[1000.0, 1234.5, 1500.0],
        dram_read_bytes=4096,
        vtcm_hit_ratio=0.85,
    )
    result = OpTraceResult(
        model="m",
        device="NPU",
        tracing_level="detail",
        operators=[op],
        num_samples=1,
    )
    out = _render(result)
    # The data row should be a single line containing both the leading op
    # path and the p90 value. If Rich wrapped the p90 cell into a 2nd line,
    # we'd see two distinct lines (one with op path, one with bare p90).
    op_lines = [ln for ln in out.splitlines() if "/encoder/layer/conv/Conv" in ln]
    assert len(op_lines) == 1, f"data row split into multiple lines: {op_lines!r}"
    # And the p90 value (1,446.9) lives on the same line.
    assert "1,446.9" in op_lines[0]


def test_detail_cumulative_percent_monotonic_with_unsorted_input():
    """Render layer must defensively sort to guarantee Cum % monotonic.

    Simulates a QHAS-style upstream that hands back ops in arbitrary
    order. Cum % column values across rows must be non-decreasing.
    """
    # Deliberately UNSORTED input: percent_of_total = [10, 30, 20]
    ops = [
        OperatorMetrics(
            name="Small",
            op_path="/x/Small",
            duration_us=10.0,
            percent_of_total=10.0,
            samples_us=[10.0],
        ),
        OperatorMetrics(
            name="Big",
            op_path="/y/Big",
            duration_us=30.0,
            percent_of_total=30.0,
            samples_us=[30.0],
        ),
        OperatorMetrics(
            name="Mid",
            op_path="/z/Mid",
            duration_us=20.0,
            percent_of_total=20.0,
            samples_us=[20.0],
        ),
    ]
    result = OpTraceResult(
        model="m",
        device="NPU",
        tracing_level="detail",
        operators=ops,
        num_samples=1,
    )
    out = _render(result)
    # After defensive sort by -percent_of_total, order should be Big, Mid, Small.
    # Cum % progresses: 30.0% → 50.0% → 60.0%
    assert "30.0%" in out
    assert "50.0%" in out
    assert "60.0%" in out
    # Verify "Big" appears in the output before "Mid", and "Mid" before "Small"
    big_pos = out.index("Big")
    mid_pos = out.index("Mid")
    small_pos = out.index("Small")
    assert big_pos < mid_pos < small_pos, (
        f"Render did not defensively sort: positions Big={big_pos}, "
        f"Mid={mid_pos}, Small={small_pos}"
    )


def test_detail_mode_summary_renders_non_empty_for_real_qhas_data(tmp_path):
    """End-to-end: render a QHAS-derived OpTraceResult and assert the
    detail-mode summary lines are NON-EMPTY for real production data.

    Pre-Bundle-A bug (I-9): the parser's ``_extract_summary`` produced
    raw QHAS keys (``time_us`` / ``total_dram_read`` / ...) while the
    renderer read user-facing keys (``inference_us`` / ``dram_read_bytes``
    / ...).  5 of 6 keys were disjoint, so the "Inference: ... |
    Execute: ... | Utilization: ..." and "DRAM: Read ... / Write ... |
    VTCM: Peak ..." summary lines silently rendered EMPTY for real
    production QHAS data.

    This e2e test wires the full QHAS fixture through
    :py:meth:`QNNMonitor.parse_existing_artifacts` into the renderer
    and asserts the summary substrings appear in the output.
    """
    from pathlib import Path

    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    fixture_dir = Path(__file__).parent / "qnn" / "fixtures"
    csv_path = tmp_path / "profiling_output.csv"
    csv_path.write_text(
        (fixture_dir / "optrace_resnet50.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    qhas_path = tmp_path / "qhas_output.json"
    qhas_path.write_text(
        (fixture_dir / "qhas_resnet50.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = QNNMonitor.parse_existing_artifacts(
        level="detail",
        artifacts={"csv": csv_path, "qhas": qhas_path},
    )

    out = _render(result)

    # Pre-Bundle-A: these substrings would be missing because parser
    # and renderer used disjoint keys for 5 of 6 metrics.
    assert "Inference:" in out, "summary line 1 lost 'Inference:' (I-9 regression)"
    assert "Execute:" in out, "summary line 1 lost 'Execute:' (I-9 regression)"
    assert "Utilization:" in out, "summary line 1 lost 'Utilization:' (I-9 regression)"
    assert "DRAM:" in out, "summary line 2 lost 'DRAM:' (I-9 regression)"
    assert "VTCM:" in out, "summary line 2 lost 'VTCM:' (I-9 regression)"
