# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Basic-mode rendering matches mockup spec."""

from io import StringIO

from rich.console import Console

from winml.modelkit.session.monitor.op_metrics import OperatorMetrics, OpTraceResult
from winml.modelkit.session.monitor.report import display_op_trace_report


def _render(result: OpTraceResult, top_n: int = 5) -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, record=True)
    display_op_trace_report(result, console=console, top_n=top_n)
    return console.export_text()


def _make_result(num_ops: int = 3) -> OpTraceResult:
    ops = [
        OperatorMetrics(
            name=f"OpType{i}",
            op_path=f"/path/to/op_{i}/Op",
            duration_us=100.0 - i * 10,
            percent_of_total=30.0 - i * 5,
            samples_us=[90.0 + i, 100.0 - i, 110.0 - i * 2],
        )
        for i in range(num_ops)
    ]
    return OpTraceResult(
        model="convnext-base",
        device="NPU",
        tracing_level="basic",
        operators=ops,
        num_samples=3,
    )


def test_basic_header_renamed():
    out = _render(_make_result())
    assert "Op-Tracing (basic)" in out
    assert "Op-Level Profiling" not in out


def test_basic_columns_in_order():
    out = _render(_make_result())
    # header row contains the four column names in order
    header_line = next(line for line in out.splitlines() if "Node" in line and "Type" in line)
    assert header_line.index("Node") < header_line.index("Type")
    assert header_line.index("Type") < header_line.index("p90")
    assert header_line.index("p90") < header_line.index("% Tot")


def test_basic_no_rank_column():
    """Mockup drops the # rank column in basic mode."""
    out = _render(_make_result())
    header_line = next(line for line in out.splitlines() if "Node" in line and "Type" in line)
    # The leading column is Node (no leading "#" digit before it).
    # Allow leading box-drawing / whitespace, then "Node".
    assert header_line.lstrip("│ ").startswith("Node")


def test_basic_long_node_path_left_truncated():
    long_path = "/very/deep" + "/segment" * 30
    op = OperatorMetrics(
        name="Conv2d",
        op_path=long_path,
        duration_us=100.0,
        percent_of_total=50.0,
        samples_us=[100.0],
    )
    result = OpTraceResult(
        model="m",
        device="NPU",
        tracing_level="basic",
        operators=[op],
        num_samples=1,
    )
    out = _render(result)
    # The truncated node line should contain the leading ellipsis and the
    # tail of the path (the rightmost characters preserved).
    assert "…" in out
    assert long_path[-20:] in out


def test_basic_p90_rendered_when_samples_present():
    out = _render(_make_result())
    # Should NOT show "—" for p90 since samples_us is populated
    assert "—" not in out


def test_basic_p90_em_dash_when_no_samples():
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
        tracing_level="basic",
        operators=[op],
        num_samples=0,
    )
    out = _render(result)
    assert "—" in out


def test_basic_p90_does_not_wrap_on_kilo_microsecond_values():
    """Realistic NPU p90 (≥ 1000 µs) must fit on a single terminal line.

    Regression for C-1: width=9 cannot hold "1,234.5 us" (10 chars). With
    a per-cell " us" suffix, Rich would wrap the cell into a second visual
    line containing the orphaned "us". The data row must occupy exactly
    one terminal line — no orphan wrap continuation.
    """
    op = OperatorMetrics(
        name="Conv2d",
        op_path="/encoder/layer/conv/Conv",
        duration_us=1234.5,
        percent_of_total=42.0,
        samples_us=[1234.5],
    )
    result = OpTraceResult(
        model="m",
        device="NPU",
        tracing_level="basic",
        operators=[op],
        num_samples=1,
    )
    out = _render(result)
    lines = out.splitlines()

    # Locate the data row containing the p90 value and the % Tot value.
    p90_lines = [ln for ln in lines if "1,234.5" in ln]
    assert len(p90_lines) == 1, f"expected exactly one row with p90 value, got: {p90_lines!r}"
    p90_line_idx = lines.index(p90_lines[0])

    # The next line must not be a wrap continuation of the data row. A
    # wrap continuation looks like a row with only whitespace and a stray
    # "us" inside the box — i.e., contains "│" but no node path/op type
    # and no closing border ("└").
    if p90_line_idx + 1 < len(lines):
        next_line = lines[p90_line_idx + 1]
        is_wrap = (
            "│" in next_line
            and "us" in next_line
            and "Conv2d" not in next_line
            and "└" not in next_line
        )
        assert not is_wrap, f"p90 cell wrapped to a 2nd line: {next_line!r}"


def test_basic_render_defensively_sorts_unsorted_ops():
    """Render layer presents ops in descending percent_of_total."""
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
    ]
    result = OpTraceResult(
        model="m",
        device="NPU",
        tracing_level="basic",
        operators=ops,
        num_samples=1,
    )
    out = _render(result)
    big_pos = out.index("Big")
    small_pos = out.index("Small")
    assert big_pos < small_pos, (
        f"Render did not defensively sort: Big at {big_pos}, Small at {small_pos}"
    )
