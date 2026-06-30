# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Report helpers — display / write JSON for op-trace results.

Relocated from optracing/report.py as part of the op-tracing refactor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

from .op_metrics import OperatorMetrics, OpTraceResult  # noqa: TC001 (used at runtime)


def display_op_trace_report(
    result: OpTraceResult,
    console: Console | None = None,
    top_n: int = 5,
) -> None:
    """Display op-tracing results as a Rich console report.

    Parameters
    ----------
    result:
        The profiling result to render.
    console:
        Optional Rich Console instance. A default is created if ``None``.
    top_n:
        Maximum number of operators to show in the table. Default ``5``
        matches the mockup spec ``OP_TRACING_TOP_K_DEFAULT`` in
        ``docs/design/perf/console_mockup.py``.
    """
    if console is None:
        console = Console()

    if result.tracing_level == "detail":
        _display_detail_report(result, console, top_n)
    else:
        _display_basic_report(result, console, top_n)


def write_op_trace_json(result: OpTraceResult, output_path: Path | str) -> None:
    """Write op-tracing results to a JSON file.

    Parameters
    ----------
    result:
        The profiling result to serialize.
    output_path:
        Destination file path. Parent directories are created if needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.to_json())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _top_n(
    operators: list[OperatorMetrics],
    n: int,
    key: Callable[[OperatorMetrics], object],
) -> list[OperatorMetrics]:
    """Return the top-``n`` operators by ``key`` with deterministic tie-break.

    Both basic and detail rendering paths want operators presented in
    descending order of ``percent_of_total`` so the ``% Tot`` column scans
    naturally and ``Cum %`` (detail mode) is monotonic. Upstream parsers vary
    (CSV sorts by cycles, QHAS preserves JSON order), so this defensive sort
    runs regardless. ``op_path`` is appended as the tie-breaker so identical
    percentages render in deterministic order.
    """
    return sorted(operators, key=lambda o: (key(o), o.op_path))[:n]


def _format_bytes(n: int | float | None) -> str:
    """Format a byte count to a human-readable string.

    Returns ``"0"`` for ``None`` or zero values.  Integer values below 1024
    are rendered without a decimal point (e.g. ``"42 B"``).
    """
    if n is None or n == 0:
        return "0"
    value: float = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024:
            if unit == "B" and value == int(value):
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _format_number(n: float | int | None) -> str:
    """Format a number with comma separators."""
    if n is None:
        return "-"
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def _truncate_node_name(name: str, max_width: int = 80) -> str:
    """Left-truncate a node path with a leading ellipsis.

    Preserves the right side because the leaf operator name (the
    differentiator) lives at the tail of the path.
    """
    if max_width <= 0:
        return ""
    if len(name) <= max_width:
        return name
    if max_width == 1:
        return "…"
    return "…" + name[-(max_width - 1) :]


def _display_basic_report(result: OpTraceResult, console: Console, top_n: int) -> None:
    """Render a basic-mode op-tracing report (4 columns, width-locked at 120)."""
    console.print()
    console.rule("[bold]Op-Tracing (basic)[/bold]")

    # Summary line (unchanged)
    parts: list[str] = []
    hvx = result.summary.get("hvx_threads")
    if hvx is not None:
        parts.append(f"HVX Threads: {hvx}")
    accel = result.summary.get("accel_execute_us")
    if accel is not None:
        parts.append(f"Accel Execute: {_format_number(accel)} us")
    if result.num_samples:
        parts.append(f"Samples: {result.num_samples}")
    if parts:
        console.print(" | ".join(parts))
    console.print()

    ops = _top_n(result.operators, top_n, key=lambda o: -o.percent_of_total)
    if not ops:
        console.print("[dim]No operator data available.[/dim]")
        return

    table = Table(show_lines=False)
    table.add_column("Node", min_width=80, max_width=80, no_wrap=True, overflow="ellipsis")
    table.add_column("Type", width=12, no_wrap=True)
    table.add_column("p90", justify="right", width=9)
    table.add_column("% Tot", justify="right", width=6)

    for op in ops:
        node_str = _truncate_node_name(op.op_path, max_width=80)
        p90_str = f"{op.p90_us:,.1f}" if op.samples_us else "—"
        table.add_row(
            node_str,
            op.name,
            p90_str,
            f"{op.percent_of_total:.1f}%",
        )

    console.print(table)


def _display_detail_report(result: OpTraceResult, console: Console, top_n: int) -> None:
    """Render a detail-mode op-tracing report (10 columns, width-locked)."""
    # Header rule
    backend_suffix = ""
    if result.tracing_backend:
        backend_suffix = f" -- {result.tracing_backend}"
    console.print()
    console.rule(f"[bold]Op-Tracing (detail){backend_suffix}[/bold]")

    # Summary lines (preserved unchanged)
    summary = result.summary
    line1_parts: list[str] = []
    inf_us = summary.get("inference_us")
    if inf_us is not None:
        line1_parts.append(f"Inference: {_format_number(inf_us)} us")
    exe_us = summary.get("execute_us")
    if exe_us is not None:
        line1_parts.append(f"Execute: {_format_number(exe_us)} us")
    util = summary.get("utilization_pct")
    if util is not None:
        line1_parts.append(f"Utilization: {util}%")
    if line1_parts:
        console.print(" | ".join(line1_parts))

    line2_parts: list[str] = []
    dram_r = summary.get("dram_read_bytes")
    dram_w = summary.get("dram_write_bytes")
    if dram_r is not None or dram_w is not None:
        dr = _format_bytes(dram_r)
        dw = _format_bytes(dram_w)
        line2_parts.append(f"DRAM: Read {dr} / Write {dw}")
    vtcm_peak = summary.get("vtcm_peak_bytes")
    if vtcm_peak is not None:
        line2_parts.append(f"VTCM: Peak {_format_bytes(vtcm_peak)}")
    if line2_parts:
        console.print(" | ".join(line2_parts))
    console.print()

    # Operator table — 10 columns (mockup canon: console_mockup.py:448-465)
    ops = _top_n(result.operators, top_n, key=lambda o: -o.percent_of_total)
    if not ops:
        console.print("[dim]No operator data available.[/dim]")
        return

    table = Table(show_lines=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column(
        "Node",
        min_width=32,
        max_width=80,
        no_wrap=True,
        overflow="ellipsis",
    )
    table.add_column("Type", min_width=9, no_wrap=True)
    table.add_column("Avg", justify="right", width=9)
    table.add_column("Total", justify="right", width=10)
    table.add_column("% Tot", justify="right", width=6)
    table.add_column("Cum %", justify="right", width=6)
    table.add_column("p90", justify="right", width=9)
    table.add_column("DRAM(R)", justify="right", width=8)
    table.add_column("VTCM Hit", justify="right", width=8)

    cum = 0.0
    for i, op in enumerate(ops, 1):
        cum += op.percent_of_total
        node_str = _truncate_node_name(op.op_path, max_width=80)
        avg_str = f"{op.avg_us:,.1f}" if op.samples_us else f"{op.duration_us:,.1f}"
        total_str = f"{op.total_us:,.1f}" if op.samples_us else "—"
        p90_str = f"{op.p90_us:,.1f}" if op.samples_us else "—"
        vtcm_str = f"{op.vtcm_hit_ratio * 100:.1f}%" if op.vtcm_hit_ratio is not None else "—"
        table.add_row(
            str(i),
            node_str,
            op.name,
            avg_str,
            total_str,
            f"{op.percent_of_total:.1f}%",
            f"{cum:.1f}%",
            p90_str,
            _format_bytes(op.dram_read_bytes),
            vtcm_str,
        )

    console.print(table)
