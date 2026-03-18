# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Console report and JSON file output for op-tracing results."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from .result import OpTraceResult  # noqa: TC001 (used at runtime)


def display_op_trace_report(
    result: OpTraceResult,
    console: Console | None = None,
    top_n: int = 15,
) -> None:
    """Display op-tracing results as a Rich console report.

    Parameters
    ----------
    result:
        The profiling result to render.
    console:
        Optional Rich Console instance. A default is created if ``None``.
    top_n:
        Maximum number of operators to show in the table.
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


def _display_basic_report(
    result: OpTraceResult, console: Console, top_n: int
) -> None:
    """Render a basic-mode report with operator name, avg cycles, and %."""
    # Header
    console.print()
    console.rule("[bold]Op-Level Profiling (basic)[/bold]")

    # Summary line
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

    # Operator table
    ops = result.operators[:top_n]
    if not ops:
        console.print("[dim]No operator data available.[/dim]")
        return

    table = Table(title="Top Operators by Duration", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Operator", min_width=30)
    table.add_column("Avg Cyc", justify="right", min_width=10)
    table.add_column("% Tot", justify="right", min_width=7)

    for i, op in enumerate(ops, 1):
        table.add_row(
            str(i),
            op.op_path,
            _format_number(op.duration_us),
            f"{op.percent_of_total:.1f}%",
        )

    console.print(table)


def _display_detail_report(
    result: OpTraceResult, console: Console, top_n: int
) -> None:
    """Render a detail-mode report with memory and cache columns."""
    # Header
    backend_suffix = ""
    if result.tracing_backend:
        backend_suffix = f" -- {result.tracing_backend}"
    console.print()
    console.rule(
        f"[bold]Op-Level Profiling (detail){backend_suffix}[/bold]"
    )

    # Summary lines
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

    # Operator table
    ops = result.operators[:top_n]
    if not ops:
        console.print("[dim]No operator data available.[/dim]")
        return

    table = Table(title="Top Operators by Duration", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Operator", min_width=25)
    table.add_column("Type", min_width=10)
    table.add_column("Dur(us)", justify="right", min_width=9)
    table.add_column("% Tot", justify="right", min_width=7)
    table.add_column("DRAM(R)", justify="right", min_width=9)
    table.add_column("VTCM Hit", justify="right", min_width=9)

    for i, op in enumerate(ops, 1):
        vtcm_str = (
            f"{op.vtcm_hit_ratio * 100:.1f}%"
            if op.vtcm_hit_ratio is not None
            else "-"
        )
        table.add_row(
            str(i),
            op.op_path,
            op.name,
            _format_number(op.duration_us),
            f"{op.percent_of_total:.1f}%",
            _format_bytes(op.dram_read_bytes),
            vtcm_str,
        )

    console.print(table)
