# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# ruff: noqa: D103

"""Console writer mockup — full analyze command output.

Demonstrates:
1. Per-EP stacked bar tables with progressive animation
2. Per-EP pattern support (detected globally, checked per-EP)
3. Overall analysis summary with detailed op/pattern listings

Run: uv run python docs/design/static_analyzer/console_mockup.py

Data Structure Convention (shared with modelkit/commands/analyze.py)
====================================================================

All operator and pattern data uses SupportLevel enum value strings as keys:

    Operator instance counts:
        dict[str, dict[str, int]]
        e.g., {"Conv": {"white": 53, "gray": 0, "black": 0, "unknown": 0}}

    Keys:
        "white"   → Supported     (green,  🟢)
        "gray"    → Partial        (yellow, 🟡)
        "black"   → Unsupported    (red,    🔴)
        "unknown" → Unknown        (dim,    🔵)

    Pattern support:
        dict[str, dict]
        e.g., {"SUBGRAPH/GELU_Erf": {"count": 8, "status": "gray"}}
        status uses the same SupportLevel value strings.

    Model-level operator counts:
        dict[str, int]
        e.g., {"Conv": 53, "Relu": 49, ...}
        Keys are plain op names (not prefixed).
"""

from __future__ import annotations

import time

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


# ── Shared constants (same as modelkit/commands/analyze.py) ───────────────

COLORS = {
    "white": "green",
    "gray": "yellow",
    "black": "red",
    "unknown": "bright_black",
}

STATUS_ICONS = {"white": "🟢", "gray": "🟡", "black": "🔴", "unknown": "🔵"}

PATTERN_STATUS_LABELS = {
    "white": "supported",
    "gray": "partial",
    "black": "unsupported",
    "unknown": "unknown",
}

MAX_BAR_WIDTH = 40


# ── Fake data ─────────────────────────────────────────────────────────────

# Model-level operator counts (same for all EPs)
ALL_OPS: dict[str, int] = {
    "Conv": 53,
    "Relu": 53,
    "BatchNormalization": 45,
    "MatMul": 25,
    "Add": 18,
    "Reshape": 10,
    "Transpose": 6,
    "Erf": 8,
    "Resize": 3,
    "Softmax": 4,
    "Gather": 3,
    "LayerNorm": 3,
}

# Per-EP operator instance counts
EP_DATA: dict[str, dict[str, dict[str, int]]] = {
    "QNNExecutionProvider": {
        "Conv": {"white": 53, "gray": 0, "black": 0, "unknown": 0},
        "Relu": {"white": 53, "gray": 0, "black": 0, "unknown": 0},
        "BatchNormalization": {"white": 45, "gray": 0, "black": 0, "unknown": 0},
        "MatMul": {"white": 20, "gray": 5, "black": 0, "unknown": 0},
        "Add": {"white": 12, "gray": 5, "black": 1, "unknown": 0},
        "Reshape": {"white": 8, "gray": 0, "black": 0, "unknown": 2},
        "Transpose": {"white": 6, "gray": 0, "black": 0, "unknown": 0},
        "Erf": {"white": 0, "gray": 8, "black": 0, "unknown": 0},
        "Resize": {"white": 0, "gray": 0, "black": 3, "unknown": 0},
        "Softmax": {"white": 4, "gray": 0, "black": 0, "unknown": 0},
        "Gather": {"white": 0, "gray": 0, "black": 0, "unknown": 3},
        "LayerNorm": {"white": 2, "gray": 1, "black": 0, "unknown": 0},
    },
    "OpenVINOExecutionProvider": {
        "Conv": {"white": 53, "gray": 0, "black": 0, "unknown": 0},
        "Relu": {"white": 53, "gray": 0, "black": 0, "unknown": 0},
        "BatchNormalization": {"white": 45, "gray": 0, "black": 0, "unknown": 0},
        "MatMul": {"white": 25, "gray": 0, "black": 0, "unknown": 0},
        "Add": {"white": 18, "gray": 0, "black": 0, "unknown": 0},
        "Reshape": {"white": 10, "gray": 0, "black": 0, "unknown": 0},
        "Transpose": {"white": 6, "gray": 0, "black": 0, "unknown": 0},
        "Erf": {"white": 8, "gray": 0, "black": 0, "unknown": 0},
        "Resize": {"white": 3, "gray": 0, "black": 0, "unknown": 0},
        "Softmax": {"white": 4, "gray": 0, "black": 0, "unknown": 0},
        "Gather": {"white": 3, "gray": 0, "black": 0, "unknown": 0},
        "LayerNorm": {"white": 3, "gray": 0, "black": 0, "unknown": 0},
    },
    "VitisAIExecutionProvider": {
        "Conv": {"white": 0, "gray": 0, "black": 0, "unknown": 53},
        "Relu": {"white": 0, "gray": 0, "black": 0, "unknown": 53},
        "BatchNormalization": {"white": 0, "gray": 0, "black": 0, "unknown": 45},
        "MatMul": {"white": 0, "gray": 0, "black": 0, "unknown": 25},
        "Add": {"white": 0, "gray": 0, "black": 0, "unknown": 18},
        "Reshape": {"white": 0, "gray": 0, "black": 0, "unknown": 10},
        "Transpose": {"white": 0, "gray": 0, "black": 0, "unknown": 6},
        "Erf": {"white": 0, "gray": 0, "black": 0, "unknown": 8},
        "Resize": {"white": 0, "gray": 0, "black": 0, "unknown": 3},
        "Softmax": {"white": 0, "gray": 0, "black": 0, "unknown": 4},
        "Gather": {"white": 0, "gray": 0, "black": 0, "unknown": 3},
        "LayerNorm": {"white": 0, "gray": 0, "black": 0, "unknown": 3},
    },
}

# Per-EP pattern support
EP_PATTERNS: dict[str, dict[str, dict]] = {
    "QNNExecutionProvider": {
        "SUBGRAPH/GELU_Erf": {"count": 8, "status": "gray"},
        "SUBGRAPH/LayerNorm": {"count": 4, "status": "white"},
        "SUBGRAPH/Attention": {"count": 2, "status": "white"},
    },
    "OpenVINOExecutionProvider": {
        "SUBGRAPH/GELU_Erf": {"count": 8, "status": "white"},
        "SUBGRAPH/LayerNorm": {"count": 4, "status": "white"},
        "SUBGRAPH/Attention": {"count": 2, "status": "white"},
    },
    "VitisAIExecutionProvider": {
        "SUBGRAPH/GELU_Erf": {"count": 8, "status": "unknown"},
        "SUBGRAPH/LayerNorm": {"count": 4, "status": "unknown"},
        "SUBGRAPH/Attention": {"count": 2, "status": "unknown"},
    },
}


# ── Rendering helpers ─────────────────────────────────────────────────────


def build_stacked_bar(counts: dict[str, int], max_count: int) -> Text:
    total = sum(counts.values())
    if total == 0:
        return Text()
    bar_width = max(1, round(total / max_count * MAX_BAR_WIDTH))
    nonzero = sum(1 for v in counts.values() if v > 0)
    bar_width = max(bar_width, nonzero)
    bar = Text()
    chars_used = 0
    for level in ("white", "gray", "black", "unknown"):
        count = counts.get(level, 0)
        if count == 0:
            continue
        width = max(1, round(count / total * bar_width))
        width = min(width, bar_width - chars_used)
        bar.append("█" * width, style=COLORS[level])
        chars_used += width
    return bar


def worst_level_icon(counts: dict[str, int]) -> str:
    if counts.get("black", 0) > 0:
        return "🔴"
    if counts.get("gray", 0) > 0:
        return "🟡"
    if counts.get("unknown", 0) > 0:
        return "🔵"
    return "🟢"


def build_spu_text(counts: dict[str, int]) -> Text:
    w = counts.get("white", 0)
    g = counts.get("gray", 0)
    b = counts.get("black", 0)
    u = counts.get("unknown", 0)
    text = Text()
    text.append(str(w), style="bold green")
    text.append("/", style="dim")
    text.append(str(g), style="bold yellow" if g > 0 else "dim")
    text.append("/", style="dim")
    text.append(str(b), style="bold red" if b > 0 else "dim")
    if u > 0:
        text.append("/", style="dim")
        text.append(str(u), style="bold bright_black")
    return text


def build_table(
    data: dict[str, dict[str, int]],
    ep_name: str = "",
    complete: bool = False,
    all_ops: dict[str, int] | None = None,
) -> Table:
    """Build per-EP analysis table.

    Shows incremental progress: ops with data show colored bars (partial or
    complete), ops without data show dim pending rows with placeholder bars.
    """
    if all_ops:
        display_order = sorted(all_ops, key=lambda x: all_ops[x], reverse=True)
    else:
        display_order = sorted(data, key=lambda x: sum(data[x].values()), reverse=True)

    # Stable max_count anchored to all_ops (no shifting during animation)
    if all_ops:
        max_count = max(all_ops.values())
    else:
        vals = [data[op] for op in display_order if data.get(op)]
        max_count = max((sum(c.values()) for c in vals), default=1)

    title = "📊 ONNX Static Analysis"
    if ep_name:
        title += f" — [bold cyan]{ep_name}[/bold cyan]"
    if complete:
        title += "  [bold green]✅ Complete[/bold green]"

    table = Table(
        title=title,
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 1),
        expand=False,
    )
    table.add_column("Op Type", width=28, no_wrap=True)
    table.add_column("S/P/U", width=14, no_wrap=True)
    table.add_column("", no_wrap=True)

    agg: dict[str, int] = {"white": 0, "gray": 0, "black": 0, "unknown": 0}

    for op_type in display_order:
        total = all_ops.get(op_type, 0) if all_ops else sum(data.get(op_type, {}).values())
        counts = data.get(op_type)

        if not counts:
            # Pending — no data yet
            bar_width = max(1, round(total / max_count * MAX_BAR_WIDTH))
            table.add_row(
                Text(f"   {op_type} ({total})", style="dim"),
                Text("...", style="dim"),
                Text("░" * bar_width, style="dim"),
            )
        else:
            # Has data — show progress
            analyzed = sum(counts.values())
            for level in agg:
                agg[level] += counts.get(level, 0)

            icon = worst_level_icon(counts)
            op_label = Text()
            op_label.append(f"{icon} ")
            op_label.append(op_type, style="cyan")
            if analyzed < total:
                op_label.append(f" ({analyzed}/{total})", style="dim")
            else:
                op_label.append(f" ({total})", style="dim")

            # Colored portion + dim remainder
            bar = build_stacked_bar(counts, max_count)
            remaining = total - analyzed
            if remaining > 0:
                remaining_width = max(1, round(remaining / max_count * MAX_BAR_WIDTH))
                bar.append("░" * remaining_width, style="dim")

            table.add_row(op_label, build_spu_text(counts), bar)

    # TOTAL row
    table.add_section()
    total_ops = sum(all_ops.values()) if all_ops else sum(agg.values())
    analyzed_count = sum(agg.values())
    total_label = Text()
    total_label.append("TOTAL", style="bold")
    if analyzed_count < total_ops:
        total_label.append(f" ({analyzed_count}/{total_ops})", style="dim")
    else:
        total_label.append(f" ({total_ops})", style="dim")

    # TOTAL bar: colored portion + dim remainder (same as per-op)
    total_bar = build_stacked_bar(agg, max(total_ops, 1))
    total_remaining = total_ops - analyzed_count
    if total_remaining > 0:
        total_remaining_width = max(1, round(total_remaining / max(total_ops, 1) * MAX_BAR_WIDTH))
        total_bar.append("░" * total_remaining_width, style="dim")

    table.add_row(total_label, build_spu_text(agg), total_bar)

    return table


# ── Demo ──────────────────────────────────────────────────────────────────


def demo_full() -> None:
    console = Console(width=95)

    # ── Model Info Header ──
    console.print()
    console.print("═" * 80)
    console.print("📊 [bold]OP CHECK[/bold]")
    console.print("═" * 80)
    console.print("   📦 Model: [bold cyan]convnext-tiny-224.onnx[/bold cyan]")
    console.print("   🔧 Opset: [green]17[/green]  Producer: [green]pytorch v2.1.0[/green]")
    console.print(
        f"   📋 Operators: [cyan]{sum(ALL_OPS.values())}[/cyan] total, "
        f"[cyan]{len(ALL_OPS)}[/cyan] unique types"
    )
    console.print()

    # ── Per-EP tables with Live animation ──
    for ep_idx, (ep_name, op_data) in enumerate(EP_DATA.items()):
        sorted_op_names = sorted(ALL_OPS, key=lambda x: ALL_OPS[x], reverse=True)

        ep_num = ep_idx + 1
        total_eps = len(EP_DATA)
        console.print("─" * 80)
        console.print(f"💻 [bold]EP {ep_num}/{total_eps}[/bold]: [bold cyan]{ep_name}[/bold cyan]")
        console.print("─" * 80)

        # Simulate incremental per-node analysis
        results: dict[str, dict[str, int]] = {}

        with Live(
            build_table(results, ep_name=ep_name, all_ops=ALL_OPS),
            console=console,
            refresh_per_second=8,
        ) as live:
            for op_type in sorted_op_names:
                results[op_type] = op_data[op_type]
                live.update(build_table(results, ep_name=ep_name, all_ops=ALL_OPS))
                time.sleep(0.15)

            # Final complete
            live.update(
                build_table(
                    results,
                    ep_name=ep_name,
                    all_ops=ALL_OPS,
                    complete=True,
                )
            )

        console.print()

    # ── Pattern Matching ──
    console.print("═" * 80)
    console.print("🔍 [bold]PATTERN MATCHING[/bold]")
    console.print("═" * 80)

    for ep_name in EP_DATA:
        patterns = EP_PATTERNS.get(ep_name, {})
        if not patterns:
            continue

        # EP sub-header
        console.print(f"   💻 [bold cyan]{ep_name}[/bold cyan]")

        for pat_id, pat_info in sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True):
            status = pat_info["status"]
            count = pat_info["count"]
            icon = STATUS_ICONS.get(status, "❓")
            label = PATTERN_STATUS_LABELS.get(status, "unknown")
            console.print(
                f"      {icon} [cyan]{pat_id}[/cyan] [dim]({count} instances)[/dim] — {label}"
            )

        console.print()

    # ── Analysis Summary ──
    console.print("═" * 80)
    console.print("📈 [bold]ANALYSIS SUMMARY[/bold]")
    console.print("═" * 80)

    for ep_name, op_data in EP_DATA.items():
        patterns = EP_PATTERNS.get(ep_name, {})

        agg: dict[str, int] = {"white": 0, "gray": 0, "black": 0, "unknown": 0}
        for counts in op_data.values():
            for level in agg:
                agg[level] += counts.get(level, 0)

        icon = worst_level_icon(agg)
        if agg["black"] > 0:
            ep_style = "bold red"
        elif agg["gray"] > 0:
            ep_style = "bold yellow"
        elif agg["unknown"] > 0 and agg["white"] == 0:
            ep_style = "bold bright_black"
        else:
            ep_style = "bold green"

        spu = build_spu_text(agg)
        console.print(f"   {icon} [{ep_style}]{ep_name}[/{ep_style}]: ", end="")
        console.print(spu)

        # List ops with issues (worst level takes priority)
        black_ops = [op for op, c in op_data.items() if c.get("black", 0) > 0]
        gray_ops = [
            op for op, c in op_data.items() if c.get("gray", 0) > 0 and c.get("black", 0) == 0
        ]
        unknown_ops = [
            op
            for op, c in op_data.items()
            if c.get("unknown", 0) > 0 and c.get("black", 0) == 0 and c.get("gray", 0) == 0
        ]

        if black_ops:
            console.print("      [red]⛔ Unsupported:[/red]")
            for op in black_ops:
                console.print(f"         • [dim]OP/ai.onnx/{op}[/dim]")
        if gray_ops:
            console.print("      [yellow]⚠️  Partial:[/yellow]")
            for op in gray_ops:
                console.print(f"         • [dim]OP/ai.onnx/{op}[/dim]")
        if unknown_ops:
            console.print("      [bright_black]❓ Unknown:[/bright_black]")
            for op in unknown_ops:
                console.print(f"         • [dim]OP/ai.onnx/{op}[/dim]")

        bad_patterns = {pid: p for pid, p in patterns.items() if p["status"] != "white"}
        if bad_patterns:
            console.print("      [dim]Patterns:[/dim]")
            for pid, p in sorted(bad_patterns.items(), key=lambda x: x[1]["count"], reverse=True):
                status = p["status"]
                icon_p = STATUS_ICONS.get(status, "❓")
                label = PATTERN_STATUS_LABELS.get(status, "unknown")
                console.print(
                    f"         {icon_p} [dim]{pid}[/dim] ({p['count']} instances, {label})"
                )

        has_issues = black_ops or gray_ops or unknown_ops or bad_patterns
        if not has_issues:
            console.print("      [green]Ready to deploy[/green]")

        console.print()

    # ── Legend ──
    console.print(
        "  [dim]S/P/U = Supported/Partial/Unsupported[/dim]"
        "  [green]██[/green] supported"
        "  [yellow]██[/yellow] partial"
        "  [red]██[/red] unsupported"
        "  [bright_black]██[/bright_black] unknown"
    )
    console.print()


if __name__ == "__main__":
    demo_full()
