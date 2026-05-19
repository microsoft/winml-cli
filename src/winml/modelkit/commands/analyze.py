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

import logging
import re
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text

from ..utils import cli as cli_utils
from ..utils.constants import EPName, EPNameOrAlias, normalize_ep_name
from ..utils.logging import configure_logging


logger = logging.getLogger(__name__)

# ── Rich visualization helpers ────────────────────────────────────────────

MAX_BAR_WIDTH = 40

_COLORS = {
    "supported": "green",
    "partial": "yellow",
    "unsupported": "red",
    "unknown": "bright_black",
}


def _discover_runtime_rule_parquet_files() -> tuple[list[Path], list[Path]]:
    """Return runtime-rule search directories and discovered parquet files.

    The runtime checker supports both flat and one-level nested layouts.
    """
    from ..analyze.utils.rule_loader import get_runtime_rules_search_dirs

    search_dirs = get_runtime_rules_search_dirs()
    parquet_files: list[Path] = []

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        parquet_files.extend(sorted(search_dir.glob("*.parquet")))
        parquet_files.extend(sorted(search_dir.glob("*/*.parquet")))

    return search_dirs, parquet_files


_TRAILING_PAREN_RE = re.compile(r" \([^()]*\)$")


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
    total = sum(counts.values())
    if total == 0:
        return Text()

    bar_width = max(1, round(total / max_count * MAX_BAR_WIDTH))
    # Ensure bar can fit all non-zero segments
    nonzero = sum(1 for v in counts.values() if v > 0)
    bar_width = max(bar_width, nonzero)

    bar = Text()
    chars_used = 0

    for level in ("supported", "partial", "unsupported", "unknown"):
        count = counts.get(level, 0)
        if count == 0:
            continue
        width = max(1, round(count / total * bar_width))
        width = min(width, bar_width - chars_used)
        bar.append("█" * width, style=_COLORS[level])
        chars_used += width

    return bar


def _build_analyzed_text(counts: dict[str, int]) -> Text:
    """Build 'W/G/B' format like '53/0/0' or '12/5/1' with colors."""
    w = counts.get("supported", 0)
    g = counts.get("partial", 0)
    b = counts.get("unsupported", 0)
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


def _build_analysis_table(
    data: dict[str, dict[str, int]],
    ep_name: EPName | None = None,
    complete: bool = False,
    all_ops: dict[str, int] | None = None,
) -> Table:
    """Build the analysis table with variable-width stacked bars.

    Args:
        data: Per-op instance counts (filled in as analysis progresses).
              Ops with data show colored bars (partial or complete).
              Ops in all_ops but not in data show dim pending rows.
        ep_name: EP name for title
        complete: Show complete marker
        all_ops: All op types with total counts (for showing pending rows)
    """
    # Build display order: all_ops sorted by count, or just data if no all_ops
    if all_ops:
        display_order = sorted(all_ops, key=lambda x: all_ops[x], reverse=True)
    else:
        display_order = sorted(data, key=lambda x: sum(data[x].values()), reverse=True)

    # Max count for bar width scaling (anchored to all_ops for stable bars during animation)
    if all_ops:
        max_count = max(all_ops.values(), default=1)
    else:
        max_count = max((sum(v.values()) for v in data.values()), default=1)

    title = "📊 OP CHECK"
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

    agg: dict[str, int] = {"supported": 0, "partial": 0, "unsupported": 0, "unknown": 0}

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
            analyzed_for_op = sum(counts.values())
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

            table.add_row(op_label, _build_analyzed_text(counts), bar)

    # Summary row
    table.add_section()
    total_ops = sum(all_ops.values()) if all_ops else sum(agg.values())
    analyzed_count = sum(agg.values())
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
        _build_analyzed_text(agg),
        total_bar,
    )

    return table


_STATUS_ICONS = {"s": "🟢", "p": "🟡", "u": "🔴", "uk": "🔵"}
_PATTERN_STATUS_LABELS = {"s": "supported", "p": "partial", "u": "unsupported", "uk": "unknown"}
_SUPPORT_LEVEL_TO_SHORT = {
    "supported": "s",
    "partial": "p",
    "unsupported": "u",
    "unknown": "uk",
}


_PAT_COLORS = {"s": "green", "p": "yellow", "u": "red", "uk": "bright_black"}


def _render_pattern_matching(
    console: Console,
    ep_patterns: dict[str, dict[str, dict]],
) -> None:
    """Render the PATTERN MATCHING section — per-EP pattern support."""
    if not any(ep_patterns.values()):
        return

    console.print("═" * 80)
    console.print("🔍 [bold]PATTERN MATCHING[/bold]")
    console.print("═" * 80)

    for ep_name, patterns in ep_patterns.items():
        if not patterns:
            continue

        console.print(f"   💻 [bold cyan]{ep_name}[/bold cyan]")

        for pat_id, pat_info in sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True):
            status = pat_info["status"]
            count = pat_info["count"]
            icon = _STATUS_ICONS.get(status, "❓")
            label = _PATTERN_STATUS_LABELS.get(status, "unknown")
            console.print(
                f"      {icon} [cyan]{pat_id}[/cyan] [dim]({count} instances)[/dim]"
                f" — [{_PAT_COLORS.get(status, 'dim')}]{label}[/{_PAT_COLORS.get(status, 'dim')}]"
            )

        console.print()


def _extract_ep_patterns(
    results: list,
) -> dict[str, dict[str, dict]]:
    """Extract per-EP subgraph pattern support from analysis results.

    Args:
        results: List of EPSupport objects from AnalysisOutput.

    Returns:
        Dict keyed by EP name, containing dicts of pattern_id to
        ``{"count": int, "status": str}`` where status is one of
        ``"s"`` (supported), ``"p"`` (partial), ``"u"`` (unsupported),
        ``"uk"`` (unknown).
    """
    ep_patterns: dict[str, dict[str, dict]] = {}
    for ep_support in results:
        patterns: dict[str, dict] = {}
        for info in ep_support.information:
            if info.pattern_id and info.pattern_id.startswith("SUBGRAPH/"):
                status = (
                    _SUPPORT_LEVEL_TO_SHORT.get(info.status.value, "uk") if info.status else "uk"
                )
                patterns[info.pattern_id] = {
                    "count": len(info.pattern_node_list),
                    "status": status,
                }
        ep_patterns[ep_support.ep_type] = patterns
    return ep_patterns


def _render_analysis_summary(
    console: Console,
    results: list,
    ep_instance_counts: dict[str, dict[str, dict[str, int]]],
    ep_patterns: dict[str, dict[str, dict]] | None = None,
    *,
    ep: EPNameOrAlias | None = None,
    device: str | None = None,
    no_data_eps: set[str] | None = None,
) -> None:
    """Render the Analysis Summary section after pattern detection.

    Args:
        console: Rich console for output.
        results: List of EPSupport objects from AnalysisOutput.
        ep_instance_counts: Per-EP instance counts accumulated during analysis,
            keyed by EP name, then op name, then support level.
        ep_patterns: Per-EP subgraph pattern support extracted from results.
        ep: Requested EP name (for display when no results).
        device: Requested device (for display when no results).
    """
    from ..analyze.models.support_level import SupportLevel

    console.print("═" * 80)
    console.print("\U0001f4c8 [bold]ANALYSIS SUMMARY[/bold]")
    console.print("═" * 80)

    if not results:
        ep_label = ep or "all EPs"
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

        # For EPs with no rule data, skip op-level rows — only show patterns.
        # Always render at least a header so the EP is visible in the summary.
        if no_data_eps and ep_name in no_data_eps:
            patterns = (ep_patterns or {}).get(ep_name, {})
            console.print(f"   🔵 [bold bright_black]{ep_name}[/bold bright_black]:")
            if patterns:
                console.print("      [dim]Op check skipped — no rule data[/dim]")
                for pid, p in sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True):
                    status = p["status"]
                    icon_p = _STATUS_ICONS.get(status, "❓")
                    label = _PATTERN_STATUS_LABELS.get(status, "unknown")
                    console.print(
                        f"      {icon_p} [dim]{pid}[/dim] ({p['count']} instances, {label})"
                    )
            else:
                console.print("      [dim]Op check skipped — no rule data, no patterns[/dim]")
            console.print()
            continue

        # Aggregate instance counts for this EP
        ep_data = ep_instance_counts.get(ep_name, {})
        agg: dict[str, int] = {"supported": 0, "partial": 0, "unsupported": 0, "unknown": 0}
        for counts in ep_data.values():
            for level in agg:
                agg[level] += counts.get(level, 0)

        icon = _worst_level_icon(agg)

        # EP name style based on worst level
        if agg.get("unsupported", 0) > 0:
            ep_style = "bold red"
        elif agg.get("partial", 0) > 0:
            ep_style = "bold yellow"
        elif agg.get("unknown", 0) > 0 and agg.get("supported", 0) == 0:
            ep_style = "bold bright_black"
        else:
            ep_style = "bold green"

        analyzed = _build_analyzed_text(agg)
        console.print(f"   {icon} [{ep_style}]{ep_name}[/{ep_style}]: ", end="")
        console.print(analyzed)

        # List ops by non-white support level
        classification = ep_support.classification
        _issue_sections = [
            (SupportLevel.UNSUPPORTED, "red", "\u26d4 Unsupported"),
            (SupportLevel.PARTIAL, "yellow", "\u26a0\ufe0f  Partial"),
            (SupportLevel.UNKNOWN, "bright_black", "\u2753 Unknown"),
        ]
        for level, color, heading in _issue_sections:
            ops = classification.get(level, [])
            if ops:
                console.print(f"      [{color}]{heading}:[/{color}]")
                for op in sorted(ops):
                    console.print(f"         \u2022 [dim]{op}[/dim]")

        # List non-supported patterns for this EP
        patterns = (ep_patterns or {}).get(ep_name, {})
        bad_patterns = {pid: p for pid, p in patterns.items() if p["status"] != "s"}
        if bad_patterns:
            console.print("      [dim]Patterns:[/dim]")
            for pid, p in sorted(bad_patterns.items(), key=lambda x: x[1]["count"], reverse=True):
                status = p["status"]
                icon_p = _STATUS_ICONS.get(status, "\u2753")
                label = _PATTERN_STATUS_LABELS.get(status, "unknown")
                console.print(
                    f"         {icon_p} [dim]{pid}[/dim] ({p['count']} instances, {label})"
                )

        has_issues = any(classification.get(lvl) for lvl, _, _ in _issue_sections) or bad_patterns
        if not has_issues:
            console.print("      [green]Ready to deploy[/green]")

        console.print()


# ── Click command ─────────────────────────────────────────────────────────


@click.command(name="analyze")
@cli_utils.model_path_option(required=True)
@cli_utils.ep_option(
    required=False, optional_message="If not specified, analyzes all supported EPs"
)
@cli_utils.device_option(
    required=False, optional_message="If not specified, uses NPU as default", default="NPU"
)
@cli_utils.verbosity_options
@cli_utils.build_config_option
@cli_utils.output_option("Save JSON output to file")
@click.option(
    "--information/--no-information",
    default=True,
    help="Include detailed recommendations (default: enabled)",
)
@click.option(
    "--htp-metadata",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to HTP metadata JSON file for enhanced pattern extraction",
)
@click.option(
    "--run-unknown-op/--no-run-unknown-op",
    default=False,
    help="Run unknown operators on local machine if possible (default: disabled)",
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
@click.pass_context
def analyze(
    ctx: click.Context,
    model: Path,
    ep: EPNameOrAlias | None,
    device: str | None,
    output: Path | None,
    information: bool,
    verbose: int,
    quiet: bool,
    config_file: Path | None,
    htp_metadata: Path | None,
    run_unknown_op: bool,
    save_node: tuple[str, ...],
    optim_config: Path | None,
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
        winml analyze --model model.onnx --ep ov --device GPU
        winml analyze --model model.onnx --output results.json
    """
    # Apply build config defaults (CLI explicit options take precedence)
    if config_file is not None:
        build_cfg = cli_utils.load_build_config(config_file)
        if build_cfg.compile and not cli_utils.is_cli_provided(ctx, "ep"):
            ep = build_cfg.compile.ep_config.provider

    # Configure logging
    configure_logging(verbosity=verbose, quiet=quiet)

    try:
        from ..analyze import ONNXStaticAnalyzer

        # Validate model
        if not model.exists():
            logger.error("ONNX model file not found: %s", model)
            sys.exit(2)

        search_dirs, parquet_files = _discover_runtime_rule_parquet_files()
        if not parquet_files:
            searched = ", ".join(str(p) for p in search_dirs) if search_dirs else "(none)"
            logger.error("No runtime rule parquet files were found.")
            logger.error("Please reinstall winml-cli, or manually download rule parquet files.")
            logger.error("Searched directories: %s", searched)
            sys.exit(2)

        from ..analyze.utils.ep_utils import (
            get_devices_with_rule_data,
            has_rule_data_for_ep,
        )

        ep_normalized = normalize_ep_name(ep)

        # Validate only when the user explicitly specified --device
        if (
            cli_utils.is_cli_provided(ctx, "device")
            and ep_normalized
            and device
            and not has_rule_data_for_ep(ep_normalized, device)
        ):
            available = get_devices_with_rule_data(ep_normalized)
            if available and device.upper() not in [a.upper() for a in available]:
                # Device is not supported by this EP at all.
                logger.error(
                    "%s only supports %s.",
                    ep_normalized,
                    ", ".join(available),
                )
                sys.exit(2)
            elif not available:
                # EP is completely unknown — no rule data and not in the EP map.
                logger.error(
                    "%s has no rule data for %s.",
                    ep_normalized,
                    device,
                )
                sys.exit(2)
            # else: device is valid for this EP but no rule data exists —
            # proceed; RuntimeChecker will return no_data results gracefully.

        ep_label = ep_normalized or "all EPs"
        device_label = device or "NPU"
        logger.info("Analyzing model: %s", model)
        logger.info("Target: %s on %s", ep_label, device_label)

        analyzer = ONNXStaticAnalyzer()

        # Console for Rich output (stderr so stdout stays clean for JSON)
        console = Console(stderr=True)

        # Model info header
        if not quiet:
            console.print()
            console.print("═" * 80)
            console.print("📊 [bold]OP CHECK[/bold]")
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
                console.print(
                    f"   🎯 Target: [bold]{ep_label}[/bold] on [bold]{device_label}[/bold]"
                )
                console.print()
                del _proto  # free memory
            except Exception:
                logger.debug("Could not load model metadata for header display")

        # Per-EP state for Live display
        current_ep_name = ""
        all_op_counts: dict[str, int] = {}
        instance_counts: dict[str, dict[str, int]] = {}
        ep_instance_counts: dict[str, dict[str, dict[str, int]]] = {}
        live: Live | None = None
        ep_counter = 0
        _no_data_eps: set[str] = set()  # EPs with no op rule data

        run_unknown_op_for_ep = run_unknown_op
        if ep == "VitisAIExecutionProvider":
            run_unknown_op_for_ep = False
            logger.info(
                "Disabling --run-unknown-op for VitisAIExecutionProvider: "
                "AMD op runtime results are not available yet"
            )

        def _finalize_live(mark_complete: bool = True) -> None:
            """Stop the active Live display, optionally marking it complete."""
            nonlocal live
            if live is None:
                return
            try:
                if mark_complete and current_ep_name:
                    ep_instance_counts[current_ep_name] = {
                        k: dict(v) for k, v in instance_counts.items()
                    }
                    live.update(
                        _build_analysis_table(
                            instance_counts,
                            ep_name=current_ep_name,
                            complete=True,
                            all_ops=all_op_counts,
                        )
                    )
            except Exception:
                logger.debug("Failed to render final table", exc_info=True)
            finally:
                live.stop()
                live = None

        def on_ep_start(ep_name, operator_counts):
            """Called when analysis starts for a new EP."""
            nonlocal current_ep_name, instance_counts, all_op_counts, ep_counter, live

            # Finalize previous EP's Live display
            if current_ep_name:
                _finalize_live()
                console.print()  # blank line between EP tables

            # Reset for new EP (normalize keys to display names)
            current_ep_name = ep_name
            all_op_counts = {_display_name(k): v for k, v in operator_counts.items()}
            instance_counts = {}

            # Skip OP CHECK display for EPs with no rule data —
            # op results would all be 0/0/0 (unknown). Pattern detection
            # still runs; results appear in the ANALYSIS SUMMARY.
            if not has_rule_data_for_ep(ep_name, device or ""):
                _no_data_eps.add(ep_name)
                return

            ep_counter += 1

            # EP section header
            console.print("─" * 80)
            console.print(f"💻 [bold]EP {ep_counter}[/bold]: [bold cyan]{ep_name}[/bold cyan]")
            console.print("─" * 80)

            # Start new Live display — all ops shown as pending
            live = Live(
                _build_analysis_table(
                    instance_counts,
                    ep_name=ep_name,
                    all_ops=all_op_counts,
                ),
                console=console,
                refresh_per_second=30,
            )
            live.start()

        def on_node_result(pattern_runtime):
            """Callback invoked per-node during analysis."""
            op = _display_name(pattern_runtime.pattern_id)
            level = pattern_runtime.result.classification.value
            op_counts = instance_counts.setdefault(op, {})
            op_counts[level] = op_counts.get(level, 0) + 1

            if live is not None:
                live.update(
                    _build_analysis_table(
                        instance_counts,
                        ep_name=current_ep_name,
                        all_ops=all_op_counts,
                    )
                )

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
                save_node_types = set(save_node)
                result = analyzer.analyze(
                    model_path=str(model),
                    ep=ep_normalized,
                    device=device,
                    enable_information=information,
                    htp_metadata_path=str(htp_metadata) if htp_metadata else None,
                    run_unknown_op=run_unknown_op_for_ep,
                    save_node_types=save_node_types,
                    on_node_result=on_node_result,
                    on_ep_start=on_ep_start,
                )

                # Extract per-EP pattern support (available now)
                ep_patterns = _extract_ep_patterns(result.output.results)

                # Finalize last EP's Live display
                _finalize_live()
            finally:
                # Safety: stop Live if still running (e.g. on exception)
                _finalize_live(mark_complete=False)
                root_logger.handlers = old_handlers

            console.print()

            # Pattern Matching section (per-EP)
            _render_pattern_matching(console, ep_patterns)

            # Analysis Summary section
            _render_analysis_summary(
                console,
                result.output.results,
                ep_instance_counts,
                ep_patterns=ep_patterns,
                ep=ep_normalized,
                device=device,
                no_data_eps=_no_data_eps,
            )

            # Legend (at the very bottom, only when there are EP results)
            if result.output.results:
                console.print(
                    "  [dim]S/P/U = Supported/Partial/Unsupported[/dim]"
                    "  [green]██[/green] supported"
                    "  [yellow]██[/yellow] partial"
                    "  [red]██[/red] unsupported"
                    "  [bright_black]██[/bright_black] unknown"
                )
                console.print()
        else:
            # Quiet mode — no live display
            save_node_types = set(save_node)
            result = analyzer.analyze(
                model_path=str(model),
                ep=ep_normalized,
                device=device,
                enable_information=information,
                htp_metadata_path=str(htp_metadata) if htp_metadata else None,
                run_unknown_op=run_unknown_op_for_ep,
                save_node_types=save_node_types,
            )

        # Save JSON if requested
        if output:
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(result.to_json(), encoding="utf-8")
                logger.info("JSON results saved to: %s", output)
            except OSError as e:
                logger.error("Failed to write JSON output to %s: %s", output, e)
            except Exception as e:
                logger.error("Failed to serialize results to JSON: %s", e)
                logger.debug("JSON serialization traceback:", exc_info=True)

        # Save optimization config if requested
        if optim_config:
            import json

            try:
                config = result.get_optimization_config(ep=ep_normalized)
                optim_config.parent.mkdir(parents=True, exist_ok=True)
                optim_config.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
                logger.info("Optimization config saved to: %s", optim_config)
            except OSError as e:
                logger.error("Failed to write config to %s: %s", optim_config, e)
            except Exception as e:
                logger.error("Failed to generate optimization config: %s", e)
                logger.debug("Config generation traceback:", exc_info=True)

        # Exit code: 0 = fully supported, 1 = partial support
        sys.exit(0 if result.is_fully_supported() else 1)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(2)
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(2)


__all__ = ["analyze"]
