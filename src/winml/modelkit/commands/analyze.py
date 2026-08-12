# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Analyze command for winml CLI.

Analyzes ONNX models for runtime support with Rich Live stacked bar
visualization, showing real-time per-node progress display.

Usage:
    winml analyze --model MODEL [--ep EP] [--device DEVICE] [OPTIONS]
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import click
from rich.cells import cell_len
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from ..session import DEVICE_TYPE_TO_DEVICE
from ..utils import cli as cli_utils
from ..utils.constants import (
    DEVICE_PRIORITY,
    EP_SUPPORTED_DEVICES,
    SUPPORTED_DEVICES,
    SUPPORTED_EPS,
    EPName,
    EPNameOrAlias,
    normalize_ep_name,
)
from ..utils.logging import configure_logging


if TYPE_CHECKING:
    from ..analyze.models.runtime_checks import PatternRuntime
    from ..analyze.optim_output import OptimizationOutputSupport


logger = logging.getLogger(__name__)

# ── Rich visualization helpers ────────────────────────────────────────────

MAX_BAR_WIDTH = 40

_COLORS = {
    "supported": "green",
    "partial": "yellow",
    "unsupported": "red",
    "unknown": "bright_black",
}


_TRAILING_PAREN_RE = re.compile(r" \([^()]*\)$")
_RUNTIME_DEBUG_LEVELS = ("unsupported", "partial", "supported")
_SUPPORT_LEVEL_KEYS = ("supported", "partial", "unsupported", "unknown")
_SKIP_NO_RULE_DATA_SUFFIX = "  Skipped - no rule data"
_SKIP_TABLE_MIN_WIDTH = 80


def _skip_table_width(section_name: str, ep_device_pair_display_name: str | None) -> int:
    """Return a stable width for skip tables based on rendered title length."""
    title = f"📊 {section_name}"
    if ep_device_pair_display_name:
        title += f" — {ep_device_pair_display_name}"
    title += _SKIP_NO_RULE_DATA_SUFFIX

    # Keep a small margin for table padding and terminal glyph width variance.
    return max(_SKIP_TABLE_MIN_WIDTH, cell_len(title) + 2)


def _display_name(pattern_id: str) -> str:
    """Extract operator display name from pattern_id.

    Examples::

        'OP/ai.onnx/Conv'              -> 'Conv'
        'OP/ai.onnx/Conv (QDQ)'        -> 'Conv'
        'OP/com.microsoft/EPContext (QNN)' -> 'EPContext'

    Strips any trailing ``" (xxx)"`` annotation (QDQ marker, EP-prefix
    suffix produced by EPContextNodeChecker, etc.).
    """
    name = pattern_id.split("/")[-1]
    return _TRAILING_PAREN_RE.sub("", name)


_LEVEL_ICONS = [
    ("unsupported", "🔴"),
    ("partial", "🟡"),
    ("unknown", "🔵"),
]


def _worst_level_icon(counts: dict[str, int]) -> str:
    """Return icon for the worst support level present (lower bound)."""
    for level, icon in _LEVEL_ICONS:
        if counts.get(level, 0) > 0:
            return icon
    return "🟢"


def _build_stacked_bar(counts: dict[str, int], max_count: int) -> Text:
    """Build a stacked bar where total width is proportional to max_count."""
    total = sum(counts.get(level, 0) for level in _SUPPORT_LEVEL_KEYS)
    if total == 0:
        return Text()

    bar_width = max(1, round(total / max_count * MAX_BAR_WIDTH))
    # Ensure bar can fit all non-zero segments
    nonzero = sum(1 for level in _SUPPORT_LEVEL_KEYS if counts.get(level, 0) > 0)
    bar_width = max(bar_width, nonzero)

    bar = Text()
    chars_used = 0

    for level in _SUPPORT_LEVEL_KEYS:
        count = counts.get(level, 0)
        if count == 0:
            continue
        width = max(1, round(count / total * bar_width))
        width = min(width, bar_width - chars_used)
        bar.append("█" * width, style=_COLORS[level])
        chars_used += width

    return bar


def _build_support_text(counts: dict[str, int]) -> Text:
    """Build 'S/P/U/Unk' format with per-level colors."""
    supported_count = counts.get("supported", 0)
    partial_count = counts.get("partial", 0)
    unsupported_count = counts.get("unsupported", 0)
    unknown_count = counts.get("unknown", 0)

    text = Text()
    text.append(str(supported_count), style="bold green")
    text.append("/", style="dim")
    text.append(str(partial_count), style="bold yellow" if partial_count > 0 else "dim")
    text.append("/", style="dim")
    text.append(str(unsupported_count), style="bold red" if unsupported_count > 0 else "dim")
    text.append("/", style="dim")
    text.append(str(unknown_count), style="bold bright_black" if unknown_count > 0 else "dim")
    return text


def _format_count_breakdown(
    *,
    counts_by_item: dict[str, int],
    max_items: int = 8,
) -> str:
    """Build compact breakdown text like A(1)+B(2)+..."""
    ranked_items = sorted(
        ((name, int(count)) for name, count in counts_by_item.items() if int(count) > 0),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked_items:
        return ""

    displayed_items = ranked_items[:max_items]
    tokens = [
        f"{name.split('/')[-1]}({count})"
        for name, count in displayed_items
    ]
    if len(ranked_items) > max_items:
        tokens.append("...")

    return "+".join(tokens)


def _build_pattern_coverage_op_line(ep_payload: dict[str, Any]) -> str:
    """Build one-line internal-op coverage summary for PATTERN CHECK."""
    op_counts: dict[str, int] = {}

    pattern_items = ep_payload.get("patterns", []) if isinstance(ep_payload, dict) else []
    for pattern_item in pattern_items:
        node_breakdown = pattern_item.get("node_breakdown", [])
        if not isinstance(node_breakdown, list):
            continue
        for breakdown_item in node_breakdown:
            if not isinstance(breakdown_item, dict):
                continue
            op_type = str(breakdown_item.get("op_type", "")).strip()
            total_count = int(breakdown_item.get("total_count", 0))
            if not op_type or total_count <= 0:
                continue
            op_counts[op_type] = op_counts.get(op_type, 0) + total_count

    breakdown = _format_count_breakdown(counts_by_item=op_counts)
    total_op_count = sum(op_counts.values())
    if not breakdown:
        return "Coverage OP(0)=(none)"
    return f"Coverage OP({total_op_count})={breakdown}"


def _build_analysis_table(
    data: dict[str, dict[str, int]],
    ep_device_pair_display_name: str | None = None,
    complete: bool = False,
    all_ops: dict[str, int] | None = None,
    op_check_skipped: bool = False,
) -> Table:
    """Build the analysis table with variable-width stacked bars.

    Args:
        data: Per-op instance counts (filled in as analysis progresses).
              Ops with data show colored bars (partial or complete).
              Ops in all_ops but not in data show dim pending rows.
          ep_device_pair_display_name: EP/device display label for title
        complete: Show complete marker
        all_ops: All op types with total counts (for showing pending rows)
        op_check_skipped: If True, render a title-only table (no rows/columns)
    """
    title = "📊 OP CHECK"
    if ep_device_pair_display_name:
        title += f" — [bold cyan]{ep_device_pair_display_name}[/bold cyan]"

    if op_check_skipped:
        title += _SKIP_NO_RULE_DATA_SUFFIX
        skip_width = _skip_table_width("OP CHECK", ep_device_pair_display_name)
        table = Table(
            title=title,
            show_header=False,
            header_style="bold",
            box=None,
            padding=(0, 1),
            expand=False,
            width=skip_width,
        )
        # add_column is required even though no rows are added — without it the
        # empty table doesn't render the centered title.
        table.add_column("")
        return table
    if complete:
        title += "  [bold green]✅ Complete[/bold green]"

    # Build display order: all_ops sorted by count, or just data if no all_ops
    if all_ops:
        display_order = sorted(all_ops, key=lambda x: all_ops[x], reverse=True)
    else:
        display_order = sorted(data, key=lambda x: sum(data[x].values()), reverse=True)

    # Max count for bar width scaling (anchored to all_ops for stable bars during animation)
    if all_ops:
        max_count = max(all_ops.values(), default=1)
    else:
        max_count = max(
            (sum(v.get(level, 0) for level in _SUPPORT_LEVEL_KEYS) for v in data.values()),
            default=1,
        )

    table = Table(
        title=title,
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 1),
        expand=False,
    )

    table.add_column("Op Type", width=28, no_wrap=True)
    table.add_column("S/P/U/Unk", width=20, no_wrap=True)
    table.add_column("", no_wrap=False)

    agg: dict[str, int] = {
        "supported": 0,
        "partial": 0,
        "unsupported": 0,
        "unknown": 0,
    }

    for op_type in display_order:
        total = all_ops.get(op_type, 0) if all_ops else sum(data.get(op_type, {}).values())
        counts = data.get(op_type)

        if not counts:
            # No data yet — fully pending
            bar_width = max(1, round(total / max_count * MAX_BAR_WIDTH)) if max_count else 1
            table.add_row(
                Text(f"   {op_type} ({total})", style="dim"),
                Text("...", style="dim"),
                Text("░" * bar_width, style="dim"),
            )
        else:
            # Has data — show progress (partial or complete)
            analyzed_for_op = sum(counts.get(level, 0) for level in _SUPPORT_LEVEL_KEYS)
            for level in agg:
                agg[level] += counts.get(level, 0)

            icon = _worst_level_icon(counts)
            op_label = Text()
            op_label.append(f"{icon} ")
            op_label.append(op_type, style="cyan")
            if analyzed_for_op < total:
                op_label.append(f" ({analyzed_for_op}/{total})", style="dim")
            else:
                op_label.append(f" ({total})", style="dim")

            # Build bar: colored portion (analyzed) + dim portion (remaining)
            bar = _build_stacked_bar(counts, max_count)
            remaining = total - analyzed_for_op
            if remaining > 0:
                remaining_width = max(1, round(remaining / max_count * MAX_BAR_WIDTH))
                bar.append("░" * remaining_width, style="dim")

            table.add_row(op_label, _build_support_text(counts), bar)

    # Summary row
    table.add_section()
    total_ops = (
        sum(all_ops.values())
        if all_ops
        else sum(agg.get(level, 0) for level in _SUPPORT_LEVEL_KEYS)
    )
    analyzed_count = sum(agg.get(level, 0) for level in _SUPPORT_LEVEL_KEYS)
    total_label = Text()
    total_label.append("TOTAL", style="bold")
    if analyzed_count < total_ops:
        total_label.append(f" ({analyzed_count}/{total_ops})", style="dim")
    else:
        total_label.append(f" ({total_ops})", style="dim")

    # TOTAL bar: colored portion + dim remainder
    total_bar = _build_stacked_bar(agg, max(total_ops, 1))
    total_remaining = total_ops - analyzed_count
    if total_remaining > 0:
        total_remaining_width = max(1, round(total_remaining / max(total_ops, 1) * MAX_BAR_WIDTH))
        total_bar.append("░" * total_remaining_width, style="dim")

    table.add_row(
        total_label,
        _build_support_text(agg),
        total_bar,
    )

    return table


def _build_pattern_query_table(
    data: dict[str, dict[str, int]],
    ep_device_pair_display_name: str | None = None,
    complete: bool = False,
    all_patterns: dict[str, int] | None = None,
    pattern_check_skipped: bool = False,
) -> Table:
    """Build pattern query progress table with S/P/U/Unk counts."""
    title = "📊 PATTERN CHECK"
    if ep_device_pair_display_name:
        title += f" — [bold cyan]{ep_device_pair_display_name}[/bold cyan]"
    if pattern_check_skipped:
        title += _SKIP_NO_RULE_DATA_SUFFIX
        skip_width = _skip_table_width("PATTERN CHECK", ep_device_pair_display_name)
        table = Table(
            title=title,
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 1),
            expand=False,
            width=skip_width,
        )
        table.add_column("Pattern", width=60, no_wrap=True)

        if all_patterns:
            display_order = sorted(all_patterns, key=lambda x: all_patterns[x], reverse=True)
            for pattern_id in display_order:
                total = int(all_patterns.get(pattern_id, 0))
                table.add_row(Text(f"   {pattern_id} ({total})", style="dim"))
        else:
            table.add_row(Text("   (none)", style="dim"))
        return table

    if complete:
        title += "  [bold green]✅ Complete[/bold green]"

    if all_patterns:
        display_order = sorted(all_patterns, key=lambda x: all_patterns[x], reverse=True)
        max_count = max(all_patterns.values(), default=1)
    else:
        display_order = sorted(data, key=lambda x: sum(data[x].values()), reverse=True)
        max_count = max(
            (sum(v.get(level, 0) for level in _SUPPORT_LEVEL_KEYS) for v in data.values()),
            default=1,
        )

    table = Table(
        title=title,
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 1),
        expand=False,
    )

    table.add_column("Pattern", width=36, no_wrap=True)
    table.add_column("S/P/U/Unk", width=20, no_wrap=True)
    table.add_column("", no_wrap=False)

    agg: dict[str, int] = {
        "supported": 0,
        "partial": 0,
        "unsupported": 0,
        "unknown": 0,
    }

    for pattern_id in display_order:
        total = (
            all_patterns.get(pattern_id, 0)
            if all_patterns
            else sum(data.get(pattern_id, {}).values())
        )
        counts = data.get(pattern_id)

        if not counts:
            bar_width = max(1, round(total / max_count * MAX_BAR_WIDTH)) if max_count else 1
            table.add_row(
                Text(f"   {pattern_id} ({total})", style="dim"),
                Text("...", style="dim"),
                Text("░" * bar_width, style="dim"),
            )
            continue

        analyzed_for_pattern = sum(counts.get(level, 0) for level in _SUPPORT_LEVEL_KEYS)
        for level in agg:
            agg[level] += counts.get(level, 0)

        icon = _worst_level_icon(counts)
        pattern_label = Text()
        pattern_label.append(f"{icon} ")
        pattern_label.append(pattern_id, style="cyan")
        if analyzed_for_pattern < total:
            pattern_label.append(f" ({analyzed_for_pattern}/{total})", style="dim")
        else:
            pattern_label.append(f" ({total})", style="dim")

        bar = _build_stacked_bar(counts, max_count)
        remaining = total - analyzed_for_pattern
        if remaining > 0:
            remaining_width = max(1, round(remaining / max_count * MAX_BAR_WIDTH))
            bar.append("░" * remaining_width, style="dim")

        table.add_row(pattern_label, _build_support_text(counts), bar)

    table.add_section()
    total_patterns = (
        sum(all_patterns.values())
        if all_patterns
        else sum(agg.get(level, 0) for level in _SUPPORT_LEVEL_KEYS)
    )
    analyzed_count = sum(agg.get(level, 0) for level in _SUPPORT_LEVEL_KEYS)

    total_label = Text()
    total_label.append("TOTAL", style="bold")
    if analyzed_count < total_patterns:
        total_label.append(f" ({analyzed_count}/{total_patterns})", style="dim")
    else:
        total_label.append(f" ({total_patterns})", style="dim")

    total_bar = _build_stacked_bar(agg, max(total_patterns, 1))
    total_remaining = total_patterns - analyzed_count
    if total_remaining > 0:
        total_remaining_width = max(
            1,
            round(total_remaining / max(total_patterns, 1) * MAX_BAR_WIDTH),
        )
        total_bar.append("░" * total_remaining_width, style="dim")

    table.add_row(
        total_label,
        _build_support_text(agg),
        total_bar,
    )

    return table


_PATTERN_STATUS_ICONS = {
    "supported": "🟢",
    "partial": "🟡",
    "unsupported": "🔴",
    "unknown": "🔵",
}


def _pattern_status_view_for_summary(
    ep_patterns: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Normalize pattern payload into {ep: {pattern_id: {count,status}}} view."""
    if not ep_patterns or not isinstance(ep_patterns, dict):
        return {}

    summary_view: dict[str, dict[str, dict[str, Any]]] = {}
    for ep_name, payload in ep_patterns.items():
        pattern_items = payload.get("patterns", []) if isinstance(payload, dict) else []
        summary_view[ep_name] = {
            str(item.get("pattern_id", "")): {
                "count": int(item.get("instances", 0)),
                # Keep backward compatibility for legacy payload values.
                "status": (
                    "unknown"
                    if str(item.get("status", "unknown")).strip().lower() == "unknow"
                    else str(item.get("status", "unknown")).strip().lower()
                ),
            }
            for item in pattern_items
            if str(item.get("pattern_id", ""))
        }

    return summary_view


def _render_analysis_summary(
    console: Console,
    results: list,
    ep_instance_counts: dict[tuple[str, str], dict[str, dict[str, int]]],
    ep_patterns: dict[str, dict[str, Any]] | None = None,
    *,
    ep: EPNameOrAlias | Literal["all", "auto"] | None = None,
    device: str | None = None,
    no_data_eps: set[tuple[str, str]] | None = None,
    op_check_skipped: bool = False,
    analyze_elapsed_ms: int | None = None,
) -> None:
    """Render the Analysis Summary section after pattern detection.

    Args:
        console: Rich console for output.
        results: List of EPSupport objects from AnalysisOutput.
        ep_instance_counts: Per-EP instance counts accumulated during analysis,
            keyed by ``(ep_name, device)``, then op name, then support level.
        ep_patterns: Per-EP subgraph pattern support extracted from results.
        ep: Requested EP name (for display when no results).
        device: Requested device (for display when no results).
        op_check_skipped: True when op check was skipped (no rule data, no
            unknown-op probing). When True, the per-op classification list is
            suppressed — every op would land in "unknown" with no actionable
            information.
        analyze_elapsed_ms: End-to-end analyze call duration for the current
            EP/device run. Rendered as a dim annotation beside the heading.
    """
    from ..analyze.models.support_level import SupportLevel

    console.print("═" * 80)
    summary_title = "\U0001f4c8 [bold]ANALYSIS SUMMARY[/bold]"
    if analyze_elapsed_ms is not None:
        if ep is not None and device:
            ep_display = _ep_name_device_display_name(str(ep), str(device))
        elif ep is not None:
            ep_display = str(ep)
        elif results:
            first_ep = results[0]
            first_ep_name = str(getattr(first_ep, "ep_type", ""))
            first_device = str(getattr(first_ep, "device_type", "")).upper()
            ep_display = (
                _ep_name_device_display_name(first_ep_name, first_device)
                if first_ep_name and first_device
                else first_ep_name or "current EP"
            )
        else:
            ep_display = "current EP"

        elapsed_seconds = max(0.0, analyze_elapsed_ms / 1000.0)
        summary_title += (
            f" [dim](Analyze total: {ep_display}, {elapsed_seconds:.2f}s)[/dim]"
        )

    console.print(summary_title)
    console.print("═" * 80)

    pattern_status_view = _pattern_status_view_for_summary(ep_patterns)

    if not results:
        ep_label: str = ep or "all EPs"
        if device:
            msg = (
                f"   [dim]No runtime check results for [bold]{ep_label}[/bold] "
                f"on [bold]{device}[/bold] — no rule data available.[/dim]"
            )
        else:
            msg = (
                f"   [dim]No runtime check results for [bold]{ep_label}[/bold] "
                f"— no rule data available.[/dim]"
            )
        console.print(msg)
        console.print()
        return

    for ep_support in results:
        ep_name = ep_support.ep_type
        device_name = (ep_support.device_type or device or "").upper()
        ep_device_pair = (ep_name, device_name)
        ep_label = (
            ep_name if not device_name else _ep_name_device_display_name(ep_name, device_name)
        )

        # Aggregate instance counts for this EP.
        ep_data = ep_instance_counts.get(ep_device_pair)
        if ep_data is None:
            ep_data = {}
        has_instance_data = any(
            sum(
                counts.get(level, 0)
                for level in _SUPPORT_LEVEL_KEYS
            )
            > 0
            for counts in ep_data.values()
        )

        # For EPs with no rule data, skip op-level rows — only show patterns.
        # Always render at least a header so the EP is visible in the summary.
        if no_data_eps and ep_device_pair in no_data_eps and not has_instance_data:
            patterns = pattern_status_view.get(ep_name, {})
            console.print(f"   🔵 [bold bright_black]{ep_label}[/bold bright_black]:")
            if patterns:
                console.print("      [dim]Op check skipped — no rule data[/dim]")
                for pid, p in sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True):
                    status = p["status"]
                    icon_p = _PATTERN_STATUS_ICONS.get(status, "❓")
                    label = status
                    console.print(
                        f"      {icon_p} [dim]{pid}[/dim] ({p['count']} instances, {label})"
                    )
            else:
                console.print("      [dim]Op check skipped — no rule data, no patterns[/dim]")
            console.print()
            continue

        agg: dict[str, int] = {
            "supported": 0,
            "partial": 0,
            "unsupported": 0,
            "unknown": 0,
        }
        for counts in ep_data.values():
            for level in agg:
                agg[level] += counts.get(level, 0)

        icon = _worst_level_icon(agg)

        # EP name style based on worst level
        if agg.get("unsupported", 0) > 0:
            ep_style = "bold red"
        elif agg.get("partial", 0) > 0:
            ep_style = "bold yellow"
        elif (
            agg.get("unknown", 0) > 0
            and agg.get("supported", 0) == 0
        ):
            ep_style = "bold bright_black"
        else:
            ep_style = "bold green"

        analyzed = _build_support_text(agg)
        console.print(f"   {icon} [{ep_style}]{ep_label}[/{ep_style}]: ", end="")
        console.print(analyzed)

        # List ops by non-white support level (skip when op check was skipped \u2014
        # the classification would be all-unknown with no useful detail).
        _issue_sections = [
            (SupportLevel.UNSUPPORTED, "red", "\u26d4 Unsupported"),
            (SupportLevel.PARTIAL, "yellow", "\u26a0\ufe0f  Partial"),
            (SupportLevel.UNKNOWN, "bright_black", "\u2753 Unknown"),
        ]
        classification = ep_support.classification
        visible_op_names = set(ep_data)
        if not op_check_skipped:
            for level, color, heading in _issue_sections:
                ops = [
                    op
                    for op in classification.get(level, [])
                    if _display_name(op) in visible_op_names
                ]
                if ops:
                    console.print(f"      [{color}]{heading}:[/{color}]")
                    for op in sorted(ops):
                        console.print(f"         \u2022 [dim]{op}[/dim]")

        # List non-supported patterns for this EP
        patterns = pattern_status_view.get(ep_name, {})
        bad_patterns = {pid: p for pid, p in patterns.items() if p["status"] != "supported"}
        if bad_patterns:
            console.print("      [dim]Patterns:[/dim]")
            for pid, p in sorted(bad_patterns.items(), key=lambda x: x[1]["count"], reverse=True):
                status = p["status"]
                icon_p = _PATTERN_STATUS_ICONS.get(status, "\u2753")
                label = status
                console.print(
                    f"         {icon_p} [dim]{pid}[/dim] ({p['count']} instances, {label})"
                )

        # "Ready to deploy" requires actual op-check data; suppress when skipped.
        if not op_check_skipped:
            has_issues = (
                any(
                    _display_name(op) in visible_op_names
                    for lvl, _, _ in _issue_sections
                    for op in classification.get(lvl, [])
                )
                or bad_patterns
            )
            if not has_issues:
                console.print("      [green]Ready to deploy[/green]")

        console.print()


def _render_optim_output_support(
    console: Console,
    results: list[OptimizationOutputSupport],
    target_label: str,
    *,
    verbose: bool = False,
) -> None:
    """Render produced-operator target support for applicable optimizations.

    For each optimization that would change the model, this shows whether the
    operators it *introduces* (added/modified nodes) are supported on the
    resolved EP/device, using the same runtime-support rule data as the op
    check above.

    Args:
        console: Rich console for output.
        results: Per-optimization support results for one target, in pipeline
            order (from :func:`check_optimization_output_support`).
        target_label: Human-readable EP/device label for the section header.
        verbose: When True, always show the per-operator reason string.
    """
    console.print("\u2550" * 80)
    console.print(f"\U0001f9e9 [bold]OPTIMIZATION OUTPUT SUPPORT[/bold] \u2014 {target_label}")
    console.print("\u2550" * 80)

    from ..analyze.models.support_level import SupportLevel

    if not results:
        console.print(
            "   [dim]No registered optimization would change this model \u2014 "
            "nothing to check.[/dim]"
        )
        console.print()
        return

    for opt in results:
        worst_color = _COLORS.get(opt.worst_support.value, "white")
        console.print(
            f"[bold green]{escape(opt.enable_flag)}[/bold green]  "
            f"[dim]({escape(opt.category)})[/dim]  "
            f"[{worst_color}]{opt.worst_support.value}[/{worst_color}]"
        )
        console.print(f"  [dim]{escape(opt.description)}[/dim]")

        if opt.error:
            console.print(
                f"  [yellow]Could not check produced operators: {escape(opt.error)}[/yellow]"
            )
            console.print()
            continue

        if not opt.operators:
            console.print("  [dim]No new operators produced.[/dim]")
            console.print()
            continue

        for op in opt.operators:
            color = _COLORS.get(op.support.value, "white")
            marker = "+" if op.change == "added" else "~"
            line = (
                f"    [{color}]\u25cf[/{color}] {marker} {escape(op.label)} "
                f"[{color}]{op.support.value}[/{color}]"
            )
            if op.reason and (verbose or op.support is not SupportLevel.SUPPORTED):
                line += f" [dim]({escape(op.reason)})[/dim]"
            console.print(line)

        console.print()

    console.print(
        "  [dim]Shows whether operators an optimization would introduce are supported "
        "on the target. Enable one with its --enable-* flag (dependencies auto-enabled).[/dim]"
    )
    console.print()


def _resolve_run_unknown_op(
    ep: EPName,
    device: str,
    run_unknown_op: bool,
    local_pairs: set[tuple[EPName, str]],
) -> bool:
    """Resolve whether to run unknown operators for a given (EP, device) pair.

    Some execution providers (e.g., VitisAI) do not have sufficient runtime
    data to support unknown operator checks, so --run-unknown-op is disabled
    for them regardless of the user's flag. Unknown-op probing also requires
    the pair to be available locally — probing a non-local pair would just
    fail at session creation.

    Args:
        ep: Execution provider name (e.g., "VitisAIExecutionProvider")
        device: Device name (e.g., "NPU")
        run_unknown_op: User-requested flag value
        local_pairs: Set of (ep, device) pairs available on the local machine

    Returns:
        Effective run_unknown_op value for this (ep, device) pair
    """
    if run_unknown_op and ep == "VitisAIExecutionProvider":
        logger.info(
            "Disabling --run-unknown-op for VitisAIExecutionProvider: "
            "AMD op runtime results are not available yet"
        )
        return False
    if run_unknown_op and (ep, device) not in local_pairs:
        logger.warning(
            "Disabling --run-unknown-op for %s: pair is not available on the local machine",
            _ep_name_device_display_name(ep, device),
        )
        return False
    return run_unknown_op


def _get_local_ep_device_pairs() -> list[tuple[EPName, str]]:
    """Return locally available (EP, device) pairs from ORT autoEP API.

    Registers WinML EP libraries first, then queries ``ort.get_ep_devices()``.
    Any ``.AUTO`` EP aliases are filtered out (e.g. OpenVINOExecutionProvider.AUTO).
    """
    pairs: set[tuple[EPName, str]] = set()

    from .. import winml

    for registered_ep_device in winml.get_registered_ep_devices():
        ep_name_raw = str(getattr(registered_ep_device, "ep_name", ""))
        if not ep_name_raw or ep_name_raw.endswith(".AUTO"):
            continue

        # ep_name_raw is an arbitrary attribute string from ORT; cast lets
        # normalize_ep_name (typed for EPNameOrAlias | None) accept it.
        # Unknown values return None and get filtered below.
        ep_name = normalize_ep_name(cast("EPNameOrAlias", ep_name_raw))
        if ep_name is None or ep_name not in SUPPORTED_EPS:
            continue

        device_obj = getattr(registered_ep_device, "device", None)
        device_type = getattr(device_obj, "type", None)
        device_name = DEVICE_TYPE_TO_DEVICE.get(device_type)
        if device_name is None:
            continue

        pairs.add((ep_name, device_name.upper()))

    return _sort_ep_device_pairs(pairs)


def _sort_ep_device_pairs(
    pairs: set[tuple[EPName, str]] | list[tuple[EPName, str]],
) -> list[tuple[EPName, str]]:
    """Sort EP/device pairs using ``EP_SUPPORTED_DEVICES`` declaration order.

    Priority is derived from a single source of truth:
    - EP priority: insertion order of keys in ``EP_SUPPORTED_DEVICES``
    - Device priority: per-EP device tuple order in ``EP_SUPPORTED_DEVICES``
    """
    ep_priority = {ep_name: idx for idx, ep_name in enumerate(EP_SUPPORTED_DEVICES)}
    device_priority_by_ep = {
        ep_name: {device_name.upper(): idx for idx, device_name in enumerate(device_names)}
        for ep_name, device_names in EP_SUPPORTED_DEVICES.items()
    }

    def _pair_sort_key(pair: tuple[EPName, str]) -> tuple[int, int, str, str]:
        ep_name, device_name = pair
        ep_rank = ep_priority.get(ep_name, len(ep_priority))
        device_rank = device_priority_by_ep.get(ep_name, {}).get(
            device_name.upper(),
            len(device_priority_by_ep.get(ep_name, {})),
        )
        return ep_rank, device_rank, ep_name, device_name

    return sorted(
        set(pairs),
        key=_pair_sort_key,
    )


def _filter_supported_local_ep_device_pairs(
    pairs: list[tuple[EPName, str]] | set[tuple[EPName, str]],
) -> list[tuple[EPName, str]]:
    """Keep only local EP/device pairs supported by the legacy matrix."""
    return [
        (ep_name, device_name)
        for ep_name, device_name in pairs
        if device_name.lower() in EP_SUPPORTED_DEVICES.get(ep_name, ())
    ]


def _select_best_exact_local_pair_for_device(
    device_name: str,
    supported_local_pairs: list[tuple[EPName, str]],
    ranked_eps_for_device: list[str],
) -> tuple[EPName, str] | None:
    """Pick the best exact supported local pair for one device.

    ``ranked_eps_for_device`` is treated as a preference list only. The chosen
    pair must come from ``supported_local_pairs`` so analyze never fabricates an
    EP/device combination that is not available locally. When the ranking omits
    local candidates, fall back to the existing deterministic local pair order.
    """
    target_device = str(device_name).upper()
    local_candidates = [
        (candidate_ep, candidate_device)
        for candidate_ep, candidate_device in supported_local_pairs
        if candidate_device == target_device
    ]
    if not local_candidates:
        return None

    candidate_by_ep = {
        candidate_ep: (candidate_ep, candidate_device)
        for candidate_ep, candidate_device in local_candidates
    }
    for ranked_ep in ranked_eps_for_device:
        canonical_ep = normalize_ep_name(cast("EPNameOrAlias", ranked_ep))
        ranked_pair = candidate_by_ep.get(canonical_ep)
        if ranked_pair is not None:
            return ranked_pair

    return local_candidates[0]


def _select_best_auto_local_pair(
    supported_local_pairs: list[tuple[EPName, str]],
) -> tuple[EPName, str] | None:
    """Pick the best default target from exact local bindings."""
    from ..session import available_eps_for_device

    for device_name in DEVICE_PRIORITY:
        best_local_pair = _select_best_exact_local_pair_for_device(
            device_name.upper(),
            supported_local_pairs,
            available_eps_for_device(device_name),
        )
        if best_local_pair is not None:
            return best_local_pair
    return None


def _ep_name_device_display_name(ep_name: str, device_name: str) -> str:
    """Return EP/device label for table and summary display."""
    return f"{ep_name} ({device_name.upper()})"


def _empty_runtime_debug_summary_payload() -> dict[str, Any]:
    """Create empty runtime debug summary payload with fixed top-level keys.

    ``unknown`` is intentionally the first key and holds a list of node keys
    (no case_indices); the remaining levels map node keys to detail entries.
    """
    payload: dict[str, Any] = {"unknown": []}
    payload.update({level: {} for level in _RUNTIME_DEBUG_LEVELS})
    return payload


def _normalize_runtime_debug_summary_payload(
    summary_payload: object,
) -> dict[str, Any]:
    """Normalize runtime debug summary payload to fixed JSON schema."""
    normalized = _empty_runtime_debug_summary_payload()
    if not isinstance(summary_payload, dict):
        return normalized

    raw_unknown = summary_payload.get("unknown")
    if isinstance(raw_unknown, list):
        normalized["unknown"] = [str(node) for node in raw_unknown]
    elif isinstance(raw_unknown, dict):
        # Tolerate a dict-shaped unknown payload by keeping node keys only.
        normalized["unknown"] = [str(node) for node in raw_unknown]

    for level in _RUNTIME_DEBUG_LEVELS:
        raw_level_entries = summary_payload.get(level)
        if not isinstance(raw_level_entries, dict):
            continue

        level_entries: dict[str, dict[str, object | None]] = {}
        for node_stable_key, raw_entry in raw_level_entries.items():
            if not isinstance(raw_entry, dict):
                continue

            level_entries[str(node_stable_key)] = {
                "case_indices": raw_entry.get("case_indices"),
                "table_path": raw_entry.get("table_path"),
                "table_file": raw_entry.get("table_file"),
                "match_status": raw_entry.get("match_status", "op_match"),
            }

        normalized[level] = level_entries

    return normalized


def _extract_runtime_debug_summary_payload_for_pair(
    run_result: Any,
    ep_name: str,
    device_name: str,
) -> dict[str, Any]:
    """Extract runtime debug summary payload for one EP/device pair."""
    empty_payload = _empty_runtime_debug_summary_payload()

    try:
        result_json = json.loads(run_result.to_json())
    except Exception:
        logger.debug("Failed to deserialize run_result for runtime debug payload", exc_info=True)
        return empty_payload

    raw_results = result_json.get("results")
    if not isinstance(raw_results, list):
        return empty_payload

    target_device = str(device_name).upper()

    # Prefer exact EP/device match first.
    for ep_result in raw_results:
        if not isinstance(ep_result, dict):
            continue
        if ep_result.get("ep_type") != ep_name:
            continue

        result_device = str(ep_result.get("device_type") or "").upper()
        if result_device and result_device != target_device:
            continue

        return _normalize_runtime_debug_summary_payload(
            ep_result.get("runtime_debug_details_summary")
        )

    # Fallback: first available runtime_debug_details_summary in this run.
    for ep_result in raw_results:
        if not isinstance(ep_result, dict):
            continue
        if "runtime_debug_details_summary" in ep_result:
            return _normalize_runtime_debug_summary_payload(
                ep_result.get("runtime_debug_details_summary")
            )

    return empty_payload


def _build_runtime_debug_output_path(model_path: Path, ep_name: str, device_name: str) -> Path:
    """Build debug summary output path near the model file."""
    filename = f"{model_path.stem}.analyze.{ep_name}.{str(device_name).upper()}.debug.json"
    return model_path.parent / filename


# ── Click command ─────────────────────────────────────────────────────────


@click.command(name="analyze")
@cli_utils.model_path_option(required=True)
@cli_utils.ep_option(
    required=False,
    default="auto",
    include_auto=True,
    include_all=True,
    optional_message=(
        "all = evaluate all rule-data-backed EPs; "
        "auto = infer a single best target from local availability"
    ),
)
@cli_utils.device_option(
    required=False,
    default="auto",
    include_auto=True,
    include_all=True,
    optional_message=(
        "all = all rule-data-backed devices; "
        "auto = infer a single best target from local availability"
    ),
)
@cli_utils.verbosity_options()
@cli_utils.no_color_option()
@cli_utils.build_config_option()
@cli_utils.output_option("Save JSON output to file")
@cli_utils.overwrite_option(optional_message="Applies to both --output and --optim-config.")
@click.option(
    "--information/--no-information",
    default=True,
    help="Include detailed recommendations (default: enabled)",
)
@cli_utils.format_option()
@click.option(
    "--run-unknown-op/--no-run-unknown-op",
    default=False,
    help="Run unknown operators on local machine if possible (default: disabled)",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help=(
        "Enable runtime debug mode. Requires WINMLCLI_RULES_DIR_FOR_DEBUG "
        "to point to a rules_debug directory containing */*.parquet files."
    ),
)
@click.option(
    "--save-node",
    multiple=True,
    type=click.Choice(["partial", "unsupported"], case_sensitive=False),
    help="Save specific node types for further analysis. Can be specified multiple times "
    "(e.g., --save-node partial --save-node unsupported).",
)
@click.option(
    "--optim-config",
    type=click.Path(path_type=Path),
    default=None,
    help="Save auto-discovered optimization config to JSON file",
)
@click.option(
    "--check-optim/--no-check-optim",
    default=False,
    help=(
        "For each optimization that would change the model, check whether the "
        "operators it introduces are supported on the resolved EP/device "
        "(default: disabled)"
    ),
)
@click.pass_context
def analyze(
    ctx: click.Context,
    model: Path,
    ep: EPNameOrAlias | Literal["all", "auto"] | None,
    device: str | None,
    output: Path | None,
    overwrite: bool,
    information: bool,
    output_format: cli_utils.OutputFormat,
    verbose: int,
    quiet: bool,
    config_file: Path | None,
    run_unknown_op: bool,
    debug: bool,
    save_node: tuple[str, ...],
    optim_config: Path | None,
    check_optim: bool,
) -> None:
    r"""Analyze ONNX model for runtime support with live progress.

    Performs static analysis to detect patterns and check operator
    compatibility, showing real-time per-operator results.

    Exit Codes:

        0: Model fully supported

        1: Partial support — some unsupported operators

        2: Error — invalid input or analysis failure

    Examples:
    \b
        winml analyze --model model.onnx --ep qnn
        winml analyze --model model.onnx --ep openvino --device gpu
        winml analyze --model model.onnx --output results.json
    """
    # Apply build config defaults (CLI explicit options take precedence).
    # Read raw JSON so missing keys are distinguishable from dataclass defaults.
    if config_file is not None:
        _, raw_cfg = cli_utils.load_build_config(config_file)
        cc = raw_cfg.get("compile") or {}
        if not cli_utils.is_cli_provided(ctx, "ep") and "execution_provider" in cc:
            ep = cc["execution_provider"]

    # Configure logging — merge with top-level group so `winml -v analyze …`
    # and `winml analyze -v …` are equivalent.
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)
    configure_logging(verbosity=verbose, quiet=quiet)

    # Refuse to clobber existing outputs unless the user opted in — fail fast
    # before analysis runs. Guards both result JSON and the optim-config dump.
    cli_utils.guard_output(output, overwrite)
    cli_utils.guard_output(optim_config, overwrite, label="Optimization config")

    try:
        from ..analyze import ONNXStaticAnalyzer

        # Validate model
        if not model.exists():
            raise click.UsageError(f"ONNX model file not found: {model}")

        from ..analyze.utils.ep_utils import (
            has_any_rule_data,
            has_rule_data_for_ep,
        )
        from ..analyze.utils.rule_loader import (
            WINMLCLI_RULES_DIR_FOR_DEBUG_ENV,
            get_runtime_rules_debug_search_dirs,
            get_runtime_rules_search_dirs,
        )

        for_debug = debug
        if for_debug:
            debug_search_dirs = get_runtime_rules_debug_search_dirs()
            has_debug_parquet = any(
                debug_dir.is_dir() and any(debug_dir.glob("*/*.parquet"))
                for debug_dir in debug_search_dirs
            )
            if not has_debug_parquet:
                configured_debug_dir = os.environ.get(WINMLCLI_RULES_DIR_FOR_DEBUG_ENV, "").strip()
                logger.error(
                    "--debug requires %s to be configured and point to a rules_debug "
                    "directory containing */*.parquet files; otherwise --debug cannot "
                    "take effect.",
                    WINMLCLI_RULES_DIR_FOR_DEBUG_ENV,
                )
                logger.error(
                    "%s configured: %s",
                    WINMLCLI_RULES_DIR_FOR_DEBUG_ENV,
                    "yes" if configured_debug_dir else "no",
                )
                if configured_debug_dir:
                    logger.error(
                        "Configured %s raw value: %s",
                        WINMLCLI_RULES_DIR_FOR_DEBUG_ENV,
                        configured_debug_dir,
                    )
                    if debug_search_dirs:
                        logger.error(
                            "Resolved absolute path(s) from %s:",
                            WINMLCLI_RULES_DIR_FOR_DEBUG_ENV,
                        )
                        for resolved_debug_dir in debug_search_dirs:
                            logger.error("  - %s", resolved_debug_dir)
                    else:
                        logger.error(
                            "Resolved absolute path(s) from %s: (none)",
                            WINMLCLI_RULES_DIR_FOR_DEBUG_ENV,
                        )
                raise click.UsageError("--debug rules directory not configured.")

        search_dirs = get_runtime_rules_search_dirs()
        if not has_any_rule_data():
            searched = ", ".join(str(p) for p in search_dirs) if search_dirs else "(none)"
            logger.error("Please reinstall winml-cli, or manually download rule parquet files.")
            logger.error("Searched directories: %s", searched)
            raise click.UsageError("No runtime rule parquet files were found.")

        # Resolve the EP/device selection. `all` keeps the full rule-data-backed
        # set (fan-out, unchanged). `auto` uses exact local OrtEpDevice bindings
        # when the request needs them (concrete EP + auto device, or auto EP +
        # all devices); the remaining auto cases keep using the shared session
        # helpers from build/run/perf.
        from ..session import EPDeviceTarget, available_eps_for_device, resolve_device

        # Only a pinned (concrete) EP can constrain device auto-resolution.
        # ``ep`` is a concrete EP/alias here unless it is the "auto"/"all"
        # sentinel; the cast drops those sentinels from the type for resolve_*.
        ep_hint: EPNameOrAlias | None = (
            None if ep in ("auto", "all") or ep is None else cast("EPNameOrAlias", ep)
        )
        concrete_requested_ep = None if ep_hint is None else normalize_ep_name(ep_hint)
        needs_local_inventory = ep in (None, "auto") or device in (None, "auto") or run_unknown_op
        local_pair_list = (
            _sort_ep_device_pairs(_get_local_ep_device_pairs()) if needs_local_inventory else []
        )
        supported_local_pair_list = _sort_ep_device_pairs(
            _filter_supported_local_ep_device_pairs(local_pair_list)
        )
        local_pairs = set(supported_local_pair_list)
        default_auto_pair = (
            _select_best_auto_local_pair(supported_local_pair_list)
            if ep in (None, "auto") and device in (None, "auto")
            else None
        )

        devices: list[str]
        if device == "all":
            devices = list(SUPPORTED_DEVICES)
        elif device == "auto":
            if concrete_requested_ep is not None:
                matching_local_pairs = [
                    (candidate_ep, candidate_device)
                    for candidate_ep, candidate_device in supported_local_pair_list
                    if candidate_ep == concrete_requested_ep
                ]
                if not matching_local_pairs:
                    raise click.UsageError(
                        "Could not auto-select a device: "
                        f"{ep} has no supported local binding available on this system."
                    )
                devices = [matching_local_pairs[0][1]]
            elif default_auto_pair is not None:
                devices = [default_auto_pair[1]]
            else:
                try:
                    resolved_device = resolve_device(
                        EPDeviceTarget(ep=ep_hint or "auto", device="auto")
                    ).device
                except (ValueError, RuntimeError) as e:
                    raise click.UsageError(f"Could not auto-select a device: {e}") from e
                devices = [resolved_device]
        elif device is not None:
            devices = [device]
        else:
            devices = []
        devices = sorted(d.upper() for d in devices)

        execution_pairs: list[tuple[EPName, str]]
        if ep == "auto" and device == "all":
            # auto + all: resolve the best exact local binding per represented
            # device using the same per-device ranking as auto + concrete
            # device, while keeping full fan-out unchanged.
            best_local_pairs: list[tuple[EPName, str]] = []
            for target_device in devices:
                best_local_pair = _select_best_exact_local_pair_for_device(
                    target_device,
                    supported_local_pair_list,
                    available_eps_for_device(target_device),
                )
                if best_local_pair is not None:
                    best_local_pairs.append(best_local_pair)
            execution_pairs = _sort_ep_device_pairs(best_local_pairs)
        else:
            if ep == "all":
                eps: list[EPName | None] = list(SUPPORTED_EPS)
            elif ep == "auto":
                # Single highest-priority exact local EP available on the target
                # device. device == "all" is handled above, so a concrete device
                # context exists here -- but guard against an empty device list
                # (e.g. a programmatic ``device=None`` call) so we exit cleanly
                # instead of raising an unguarded IndexError on ``devices[0]``.
                ref_device = default_auto_pair[1] if default_auto_pair is not None else None
                ref_device = ref_device or (devices[0] if devices else None)
                if default_auto_pair is not None:
                    best_local_pair = default_auto_pair
                elif ref_device:
                    best_local_pair = _select_best_exact_local_pair_for_device(
                        ref_device,
                        supported_local_pair_list,
                        available_eps_for_device(ref_device),
                    )
                else:
                    best_local_pair = None
                if not ref_device:
                    raise click.UsageError("No device context available for EP auto-resolution.")
                if best_local_pair is None:
                    raise click.UsageError(
                        f"No execution provider is available for device '{ref_device}'."
                    )
                execution_pairs = _sort_ep_device_pairs([best_local_pair])
                eps = []
            else:
                # ep is a specific EP or alias
                eps = [normalize_ep_name(ep)]

            if ep != "auto":
                # Build with a for-loop rather than a single nested comprehension so
                # the `candidate_ep is not None and ... in EP_SUPPORTED_DEVICES`
                # narrowing carries through to the appended tuple's type (EPName,
                # not str). The inner generator stays a comprehension to satisfy
                # ruff PERF401.
                execution_pairs = []
                for candidate_ep in eps:
                    if candidate_ep is None or candidate_ep not in EP_SUPPORTED_DEVICES:
                        continue
                    execution_pairs.extend(
                        (candidate_ep, candidate_device)
                        for candidate_device in devices
                        if candidate_device.lower() in EP_SUPPORTED_DEVICES[candidate_ep]
                    )
                execution_pairs = _sort_ep_device_pairs(execution_pairs)

        if not execution_pairs:
            raise click.UsageError("No EP/device combination matched the current selection.")

        logger.info("Analyzing model: %s", model)
        if needs_local_inventory:
            logger.info(
                "Local targets: %s",
                ", ".join(
                    _ep_name_device_display_name(candidate_ep, candidate_device)
                    for candidate_ep, candidate_device in local_pair_list
                ),
            )
        logger.info(
            "Execution targets: %s",
            ", ".join(
                _ep_name_device_display_name(target_ep, target_device)
                for target_ep, target_device in execution_pairs
            ),
        )

        analyzer = ONNXStaticAnalyzer()

        # Console for Rich output (stderr so stdout stays clean for JSON)
        console = Console(stderr=True)

        # Optionally probe which optimizations would change the model and what
        # operators they would introduce. This probe is target-independent, so
        # materialize it once here and only re-run the (cheap) support lookup per
        # EP/device below. Quiet mode disables rendering, not data collection.
        optim_outputs: list[tuple[Any, Any]] = []
        optim_probe_error: str | None = None
        if check_optim:
            try:
                import onnx

                from ..optim import get_all_capabilities, iter_optimization_outputs

                if not quiet:
                    console.print(
                        "[dim]Probing optimization outputs "
                        "(this can take a while on large models)…[/dim]"
                    )
                optim_proto = onnx.load(str(model))
                optim_outputs = list(iter_optimization_outputs(optim_proto, get_all_capabilities()))
                # Each entry retains a full produced-model clone so the (cheap)
                # per-target support lookup below can reuse them across every
                # resolved EP/device without re-running the probe. The trade-off
                # is peak memory ~ (#applicable optimizations x model size); log
                # the count so that cost is visible for large models.
                logger.info(
                    "Probed %d applicable optimization(s) for output support checking",
                    len(optim_outputs),
                )
            except Exception as exc:
                logger.warning("Could not probe optimization outputs: %s", exc)
                optim_probe_error = str(exc)
                optim_outputs = []

        # Model info header
        if not quiet:
            console.print()
            console.print("═" * 80)
            console.print("📊 [bold]ANALYSIS PROGRESS[/bold]")
            console.print("═" * 80)
            console.print(f"   📦 Model: [bold cyan]{model.name}[/bold cyan]")

            # Load model metadata for header
            try:
                import onnx

                _proto = onnx.load(str(model), load_external_data=False)
                _opset = _proto.opset_import[0].version if _proto.opset_import else "?"
                _producer = _proto.producer_name or "unknown"
                if _proto.producer_version:
                    _producer += f" v{_proto.producer_version}"
                _total_ops = len(_proto.graph.node)
                _unique_ops = len({n.op_type for n in _proto.graph.node})
                console.print(
                    f"   🔧 Opset: [green]{_opset}[/green]  Producer: [green]{_producer}[/green]"
                )
                console.print(
                    f"   📋 Operators: [cyan]{_total_ops}[/cyan] total, "
                    f"[cyan]{_unique_ops}[/cyan] unique types"
                )
                if len(execution_pairs) > 1:
                    execution_labels = ", ".join(
                        _ep_name_device_display_name(target_ep, target_device)
                        for target_ep, target_device in execution_pairs
                    )
                    console.print(f"   🎯 Analysis targets: [cyan]{execution_labels}[/cyan]")
                console.print()
                del _proto  # free memory
            except Exception:
                logger.debug("Could not load model metadata for header display")

        # Per-EP state for Live display
        current_ep_device_pair: tuple[str, str] | None = None
        current_device = execution_pairs[0][1]
        all_op_counts: dict[str, int] = {}
        instance_counts: dict[str, dict[str, int]] = {}
        all_pattern_counts: dict[str, int] = {}
        pattern_instance_counts: dict[str, dict[str, int]] = {}
        ep_instance_counts: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
        live: Live | None = None
        pattern_progress: Progress | None = None
        pattern_progress_task_id: TaskID | None = None
        pattern_progress_total = 0
        unknown_op_progress: Progress | None = None
        unknown_op_task_id: TaskID | None = None
        unknown_op_total_nodes = 0
        ep_counter = 0
        ep_header_rendered = False
        _no_data_eps: set[tuple[str, str]] = set()  # EP/device pairs with no op rule data
        analysis_results: list = []
        optimization_support_payloads: list[dict[str, object] | None] = []
        current_run_unknown_op = False
        current_op_check_skipped = False
        current_pattern_check_skipped = False
        pattern_check_active = False

        def _collect_optimization_support(
            target_ep: EPName, target_device: str
        ) -> tuple[list[Any], dict[str, object]]:
            """Check one target and return renderable results plus JSON data."""
            from ..analyze.optim_output import check_optimization_output_support

            support_error: str | None = None
            optim_support = []
            if optim_probe_error is None:
                try:
                    optim_support = check_optimization_output_support(
                        optim_outputs,
                        ep=target_ep,
                        device=target_device,
                        model_path=str(model),
                    )
                except Exception as exc:
                    support_error = str(exc)
                    logger.warning("Could not check optimization output support: %s", exc)
            payload: dict[str, object] = {
                "ep_type": target_ep,
                "device_type": target_device,
                "probe_error": optim_probe_error,
                "support_error": support_error,
                "optimizations": [item.to_dict() for item in optim_support],
            }
            return optim_support, payload

        def _current_ep_device_pair_display_name() -> str:
            """Return current EP/device display label, or empty when unset."""
            if current_ep_device_pair is None:
                return ""
            return _ep_name_device_display_name(*current_ep_device_pair)

        def _finalize_unknown_op_progress() -> None:
            """Stop active unknown-op progress bar for no-rule-data probing."""
            nonlocal unknown_op_progress, unknown_op_task_id, unknown_op_total_nodes
            if unknown_op_progress is None:
                return
            try:
                if unknown_op_task_id is not None and unknown_op_total_nodes > 0:
                    unknown_op_progress.update(
                        unknown_op_task_id,
                        completed=unknown_op_total_nodes,
                    )
            except Exception:
                logger.debug("Failed to finalize unknown-op progress", exc_info=True)
            finally:
                unknown_op_progress.stop()

                # Persist and render per-op compile/run snapshot after probing completes.
                if current_ep_device_pair is not None and instance_counts:
                    ep_instance_counts[current_ep_device_pair] = {
                        k: dict(v) for k, v in instance_counts.items()
                    }
                    try:
                        console.print(
                            _build_analysis_table(
                                instance_counts,
                                ep_device_pair_display_name=_current_ep_device_pair_display_name(),
                                complete=True,
                                all_ops=all_op_counts,
                                op_check_skipped=current_op_check_skipped,
                            )
                        )
                    except Exception:
                        logger.debug("Failed to render unknown-op final table", exc_info=True)

                unknown_op_progress = None
                unknown_op_task_id = None
                unknown_op_total_nodes = 0

        def _finalize_pattern_live(mark_complete: bool = True) -> None:
            """Render the completed pattern-query table once."""
            nonlocal current_pattern_check_skipped, pattern_check_active
            nonlocal pattern_progress, pattern_progress_task_id, pattern_progress_total
            if not pattern_check_active:
                return
            if pattern_progress is not None:
                try:
                    if (
                        mark_complete
                        and pattern_progress_task_id is not None
                        and pattern_progress_total > 0
                    ):
                        pattern_progress.update(
                            pattern_progress_task_id,
                            completed=pattern_progress_total,
                        )
                except Exception:
                    logger.debug("Failed to finalize pattern progress", exc_info=True)
                finally:
                    pattern_progress.stop()
                    pattern_progress = None
                    pattern_progress_task_id = None
                    pattern_progress_total = 0
            final_table = _build_pattern_query_table(
                pattern_instance_counts,
                ep_device_pair_display_name=_current_ep_device_pair_display_name(),
                complete=mark_complete and not current_pattern_check_skipped,
                all_patterns=all_pattern_counts,
                pattern_check_skipped=current_pattern_check_skipped,
            )
            console.print(final_table)
            pattern_check_active = False

        def _finalize_live(mark_complete: bool = True) -> None:
            """Stop the active Live display, optionally marking it complete."""
            nonlocal live
            if live is None:
                return
            try:
                if mark_complete and current_ep_device_pair is not None:
                    ep_instance_counts[current_ep_device_pair] = {
                        k: dict(v) for k, v in instance_counts.items()
                    }
                    live.update(
                        _build_analysis_table(
                            instance_counts,
                            ep_device_pair_display_name=_current_ep_device_pair_display_name(),
                            complete=True,
                            all_ops=all_op_counts,
                            op_check_skipped=current_op_check_skipped,
                        )
                    )
            except Exception:
                logger.debug("Failed to render final table", exc_info=True)
            finally:
                live.stop()
                live = None

        def on_pattern_query_start(
            ep_name: EPName,
            pattern_counts: dict[str, int],
            pattern_lookup_supported: bool = True,
        ) -> None:
            """Called when pattern query stage starts for one EP."""
            nonlocal current_ep_device_pair
            nonlocal pattern_instance_counts, all_pattern_counts, ep_counter
            nonlocal ep_header_rendered, current_pattern_check_skipped, pattern_check_active
            nonlocal pattern_progress, pattern_progress_task_id, pattern_progress_total

            # Safety: finalize any stale displays.
            _finalize_pattern_live()
            _finalize_live()
            _finalize_unknown_op_progress()

            current_ep_device_pair = (ep_name, current_device)
            all_pattern_counts = {
                str(pattern_id): int(total)
                for pattern_id, total in pattern_counts.items()
                if int(total) > 0
            }
            pattern_instance_counts = {}
            current_pattern_check_skipped = not pattern_lookup_supported
            pattern_check_active = True

            ep_counter += 1
            console.print("─" * 80)
            console.print(
                f"💻 [bold]EP {ep_counter}[/bold]: [bold cyan]{ep_name}[/bold cyan] "
                f"on [bold]{current_device}[/bold]"
            )
            console.print("─" * 80)
            ep_header_rendered = True

            pattern_total = sum(all_pattern_counts.values())
            if pattern_total > 0 and not current_pattern_check_skipped:
                pattern_progress_total = pattern_total
                pattern_progress = Progress(
                    TextColumn("   [cyan]Pattern progress[/cyan]"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                pattern_progress.start()
                pattern_progress_task_id = pattern_progress.add_task(
                    "pattern",
                    total=pattern_progress_total,
                )

        def on_pattern_query_result(ep_name: EPName, pattern_id: str, support_status: str) -> None:
            """Called when one pattern instance gets a query status."""
            if current_ep_device_pair is None:
                return
            if ep_name != current_ep_device_pair[0]:
                return

            status = str(support_status).strip().lower()
            if status == "unknow":
                status = "unknown"
            if status not in _SUPPORT_LEVEL_KEYS:
                status = "unknown"

            counts = pattern_instance_counts.setdefault(str(pattern_id), {})
            counts[status] = counts.get(status, 0) + 1

            if pattern_progress is not None and pattern_progress_task_id is not None:
                pattern_progress.advance(pattern_progress_task_id, 1)

        def on_pattern_summary_ready(ep_name: EPName, ep_payload: dict[str, Any]) -> None:
            """Finalize pattern progress display before OP CHECK starts."""
            _ = ep_name
            _finalize_pattern_live(mark_complete=not current_pattern_check_skipped)
            console.print()
            console.print(_build_pattern_coverage_op_line(ep_payload), soft_wrap=True)

        def on_ep_start(
            ep_name: EPName,
            operator_counts: dict[str, int],
            skip_runtime_checks: bool = False,
        ) -> None:
            """Called when OP CHECK stage starts for a new EP."""
            nonlocal current_ep_device_pair
            nonlocal instance_counts, all_op_counts, live
            nonlocal unknown_op_progress, unknown_op_task_id, unknown_op_total_nodes
            nonlocal current_run_unknown_op, current_op_check_skipped
            nonlocal ep_counter, ep_header_rendered

            _finalize_pattern_live()
            _finalize_live()
            _finalize_unknown_op_progress()

            if not ep_header_rendered:
                ep_counter += 1
                console.print("─" * 80)
                console.print(
                    f"💻 [bold]EP {ep_counter}[/bold]: [bold cyan]{ep_name}[/bold cyan] "
                    f"on [bold]{current_device}[/bold]"
                )
                console.print("─" * 80)
                ep_header_rendered = True

            # Reset for new EP (normalize keys to display names)
            current_ep_device_pair = (ep_name, current_device)
            all_op_counts = {
                _display_name(k): int(v)
                for k, v in operator_counts.items()
                if int(v) > 0
            }
            instance_counts = {}

            if skip_runtime_checks:
                current_op_check_skipped = True
                _no_data_eps.add((ep_name, current_device))
                console.print()
                console.print(
                    _build_analysis_table(
                        instance_counts,
                        ep_device_pair_display_name=_current_ep_device_pair_display_name(),
                        op_check_skipped=True,
                    )
                )
                return

            has_rule_data = has_rule_data_for_ep(ep_name, current_device)
            current_op_check_skipped = not has_rule_data and not current_run_unknown_op

            # Skip OP CHECK display for EPs with no rule data —
            # op results would all be 0/0/0 (unknown). Pattern detection
            # still runs; results appear in the ANALYSIS SUMMARY.
            if not has_rule_data:
                _no_data_eps.add((ep_name, current_device))

                if current_run_unknown_op:
                    total_nodes = sum(operator_counts.values())
                    unknown_op_total_nodes = max(0, total_nodes)

                    if unknown_op_total_nodes == 0:
                        console.print(
                            "   [green]All operators are covered by pattern matching; "
                            "no OP CHECK nodes remain.[/green]"
                        )
                        return

                    console.print(
                        "   [yellow]No rule data detected; probing unknown ops "
                        "one by one...[/yellow]"
                    )

                    unknown_op_progress = Progress(
                        TextColumn("   [cyan]Unknown-op progress[/cyan]"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                        console=console,
                        redirect_stdout=False,
                        redirect_stderr=False,
                    )
                    unknown_op_progress.start()
                    unknown_op_task_id = unknown_op_progress.add_task(
                        "unknown-op",
                        total=max(1, unknown_op_total_nodes),
                    )
                    return

                console.print()
                console.print(
                    _build_analysis_table(
                        instance_counts,
                        ep_device_pair_display_name=_current_ep_device_pair_display_name(),
                        op_check_skipped=True,
                    )
                )
                return

            console.print()

            # Start new Live display — all ops shown as pending
            live = Live(
                _build_analysis_table(
                    instance_counts,
                    ep_device_pair_display_name=_current_ep_device_pair_display_name(),
                    all_ops=all_op_counts,
                    op_check_skipped=current_op_check_skipped,
                ),
                console=console,
                refresh_per_second=30,
            )
            live.start()

        def on_node_result(pattern_runtime: PatternRuntime) -> None:
            """Callback invoked per-node during analysis."""
            if pattern_runtime.result.reason == "pattern_matched":
                # Pattern-matched nodes are excluded from OP CHECK totals and rows.
                return

            op = _display_name(pattern_runtime.pattern_id)
            level = pattern_runtime.result.classification.value
            op_counts = instance_counts.setdefault(op, {})
            op_counts[level] = op_counts.get(level, 0) + 1

            if live is not None:
                live.update(
                    _build_analysis_table(
                        instance_counts,
                        ep_device_pair_display_name=_current_ep_device_pair_display_name(),
                        all_ops=all_op_counts,
                        op_check_skipped=current_op_check_skipped,
                    )
                )

            if unknown_op_progress is not None and unknown_op_task_id is not None:
                unknown_op_progress.advance(unknown_op_task_id, 1)

        save_node_types = set(save_node)

        if not quiet:
            # Redirect logging through Rich console so log messages render
            # above the Live table instead of breaking it
            root_logger = logging.getLogger()
            old_handlers = root_logger.handlers[:]
            rich_handler = RichHandler(
                console=console,
                show_path=False,
                show_time=True,
                rich_tracebacks=False,
            )
            rich_handler.setLevel(root_logger.level)
            root_logger.handlers = [rich_handler]

            try:
                for target_ep, target_device in execution_pairs:
                    current_device = target_device
                    current_ep_device_pair = None
                    ep_header_rendered = False

                    run_unknown_op_for_ep = _resolve_run_unknown_op(
                        target_ep, target_device, run_unknown_op, local_pairs
                    )

                    current_run_unknown_op = run_unknown_op_for_ep

                    analyze_start = time.perf_counter()
                    result = analyzer.analyze(
                        model_path=str(model),
                        ep=target_ep,
                        device=target_device,
                        enable_information=information,
                        for_debug=for_debug,
                        run_unknown_op=run_unknown_op_for_ep,
                        save_node_types=save_node_types,
                        on_node_result=on_node_result,
                        on_ep_start=on_ep_start,
                        on_pattern_query_start=on_pattern_query_start,
                        on_pattern_query_result=on_pattern_query_result,
                        on_pattern_summary_ready=on_pattern_summary_ready,
                    )
                    analyze_elapsed_ms = int((time.perf_counter() - analyze_start) * 1000)
                    analysis_results.append(result)

                    ep_patterns = result.pattern_matching_by_ep

                    # Finalize last EP's Live display
                    _finalize_live()
                    _finalize_unknown_op_progress()

                    console.print()

                    # Analysis Summary section
                    _render_analysis_summary(
                        console,
                        result.output.results,
                        ep_instance_counts,
                        ep_patterns=ep_patterns,
                        ep=target_ep,
                        device=target_device,
                        no_data_eps=_no_data_eps,
                        op_check_skipped=current_op_check_skipped,
                        analyze_elapsed_ms=analyze_elapsed_ms,
                    )

                    # Legend (at the very bottom, only when there are EP results)
                    if result.output.results:
                        console.print(
                            "  [dim]S/P/U/Unk = Supported/Partial/Unsupported/Unknown[/dim]"
                            "  [green]██[/green] supported"
                            "  [yellow]██[/yellow] partial"
                            "  [red]██[/red] unsupported"
                            "  [bright_black]██[/bright_black] unknown"
                        )
                        console.print()

                    # Optimization output support section (per-EP), when opted in.
                    if check_optim:
                        optim_support, payload = _collect_optimization_support(
                            target_ep, target_device
                        )
                        optimization_support_payloads.append(payload)
                        _render_optim_output_support(
                            console,
                            optim_support,
                            _ep_name_device_display_name(target_ep, target_device),
                            verbose=verbose > 0,
                        )
            finally:
                # Safety: stop Live if still running (e.g. on exception)
                _finalize_pattern_live(mark_complete=False)
                _finalize_live(mark_complete=False)
                _finalize_unknown_op_progress()
                root_logger.handlers = old_handlers
        else:
            # Quiet mode — no live display
            for target_ep, target_device in execution_pairs:
                run_unknown_op_for_ep = _resolve_run_unknown_op(
                    target_ep, target_device, run_unknown_op, local_pairs
                )

                result = analyzer.analyze(
                    model_path=str(model),
                    ep=target_ep,
                    device=target_device,
                    enable_information=information,
                    for_debug=for_debug,
                    run_unknown_op=run_unknown_op_for_ep,
                    save_node_types=save_node_types,
                )
                analysis_results.append(result)

                if check_optim:
                    _, payload = _collect_optimization_support(target_ep, target_device)
                    optimization_support_payloads.append(payload)

        result = analysis_results[-1]

        serialized_results: list[dict[str, object]] = []
        json_mode = output_format == "json"
        if output or json_mode:
            for index, run_result in enumerate(analysis_results):
                payload = json.loads(run_result.to_json())
                if check_optim:
                    payload["optimization_output_support"] = optimization_support_payloads[index]
                serialized_results.append(payload)

        # Save JSON if requested
        if output:
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                if len(analysis_results) == 1:
                    output.write_text(json.dumps(serialized_results[0], indent=2), encoding="utf-8")
                else:
                    output.write_text(json.dumps(serialized_results), encoding="utf-8")
                logger.info("JSON results saved to: %s", output)
            except OSError as e:
                logger.error("Failed to write JSON output to %s: %s", output, e)
            except Exception as e:
                logger.error("Failed to serialize results to JSON: %s", e)
                logger.debug("JSON serialization traceback:", exc_info=True)

        # Save runtime debug summary JSON next to model when debug mode is enabled.
        if for_debug:
            try:
                for (target_ep, target_device), run_result in zip(
                    execution_pairs, analysis_results, strict=True
                ):
                    debug_payload = _extract_runtime_debug_summary_payload_for_pair(
                        run_result=run_result,
                        ep_name=target_ep,
                        device_name=target_device,
                    )
                    debug_output = _build_runtime_debug_output_path(
                        model_path=model,
                        ep_name=target_ep,
                        device_name=target_device,
                    )
                    debug_output.write_text(
                        json.dumps(debug_payload, indent=2),
                        encoding="utf-8",
                    )
                    logger.info("Runtime debug summary saved to: %s", debug_output)
            except OSError as e:
                logger.error("Failed to write runtime debug summary file: %s", e)
            except Exception as e:
                logger.error("Failed to prepare runtime debug summary JSON: %s", e)
                logger.debug("Runtime debug summary traceback:", exc_info=True)

        # Save optimization config if requested
        if optim_config:
            try:
                # Merge optimization configs from all execution pairs; warn on conflicts.

                per_pair_values: dict[str, list[tuple[tuple[str, str], object]]] = {}
                for (target_ep, target_device), run_result in zip(
                    execution_pairs, analysis_results, strict=True
                ):
                    pair_config = run_result.get_optimization_config(ep=target_ep).to_dict()
                    for key, value in pair_config.items():
                        per_pair_values.setdefault(key, []).append(
                            ((target_ep, target_device), value)
                        )

                merged: dict[str, object] = {}
                for key, entries in per_pair_values.items():
                    merged[key] = entries[0][1]
                    distinct = {value for _, value in entries}
                    if len(distinct) == 1:
                        continue
                    detail = ", ".join(
                        f"{_ep_name_device_display_name(pair[0], pair[1])}={value!r}"
                        for pair, value in entries
                    )
                    logger.warning(
                        "Conflicting optimization setting %r across analysis pairs: %s "
                        "(using %r from first pair in merged config)",
                        key,
                        detail,
                        merged[key],
                    )

                merged = dict(sorted(merged.items()))
                optim_config.parent.mkdir(parents=True, exist_ok=True)
                optim_config.write_text(json.dumps(merged, indent=2), encoding="utf-8")
                logger.info("Optimization config saved to: %s", optim_config)
            except OSError as e:
                logger.error("Failed to write config to %s: %s", optim_config, e)
            except Exception as e:
                logger.error("Failed to generate optimization config: %s", e)
                logger.debug("Config generation traceback:", exc_info=True)

        # Emit JSON to stdout if requested
        if json_mode:
            if len(analysis_results) == 1:
                click.echo(json.dumps(serialized_results[0], indent=2))
            else:
                click.echo(json.dumps(serialized_results, indent=2))

        # Exit code: 0 = fully supported, 1 = partial support
        overall_supported = all(run_result.is_fully_supported() for run_result in analysis_results)
        if not overall_supported:
            raise cli_utils.PartialSupportError

    except FileNotFoundError as e:
        raise click.UsageError(f"File not found: {e}") from e
    except (click.exceptions.Exit, click.ClickException):
        # Exit/click exceptions are intentional control flow; re-raise so the
        # catch-all below doesn't relabel them as "Analysis failed".
        raise
    except Exception as e:
        if verbose:
            logger.exception("Full traceback:")
        raise click.UsageError(f"Analysis failed: {e}") from e


__all__ = ["analyze"]
