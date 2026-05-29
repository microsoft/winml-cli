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
from typing import Literal

import click
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
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

from ..utils import cli as cli_utils
from ..utils.constants import (
    ALL_EP_NAMES,
    DEVICE_TYPE_TO_DEVICE,
    EP_SUPPORTED_DEVICES,
    SUPPORTED_DEVICES,
    SUPPORTED_EPS,
    EPName,
    EPNameOrAlias,
    normalize_ep_name,
)
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
    """Build 'S/P/U/Unk' format like '53/0/0/0' or '12/5/1/3' with colors."""
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
    text.append("/", style="dim")
    text.append(str(u), style="bold bright_black" if u > 0 else "dim")
    return text


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
        title += "  Skipped - no rule data"
        table = Table(
            title=title,
            show_header=False,
            header_style="bold",
            box=None,
            padding=(0, 1),
            expand=False,
            width=80,
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
        max_count = max((sum(v.values()) for v in data.values()), default=1)

    table = Table(
        title=title,
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 1),
        expand=False,
    )

    table.add_column("Op Type", width=28, no_wrap=True)
    table.add_column("S/P/U/Unk", width=16, no_wrap=True)
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
    ep_instance_counts: dict[tuple[str, str], dict[str, dict[str, int]]],
    ep_patterns: dict[str, dict[str, dict]] | None = None,
    *,
    ep: EPNameOrAlias | Literal["all", "auto"] | None = None,
    device: str | None = None,
    no_data_eps: set[tuple[str, str]] | None = None,
    op_check_skipped: bool = False,
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
                counts.get(level, 0) for level in ("supported", "partial", "unsupported", "unknown")
            )
            > 0
            for counts in ep_data.values()
        )

        # For EPs with no rule data, skip op-level rows — only show patterns.
        # Always render at least a header so the EP is visible in the summary.
        if no_data_eps and ep_device_pair in no_data_eps and not has_instance_data:
            patterns = (ep_patterns or {}).get(ep_name, {})
            console.print(f"   🔵 [bold bright_black]{ep_label}[/bold bright_black]:")
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
        if not op_check_skipped:
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

        # "Ready to deploy" requires actual op-check data; suppress when skipped.
        if not op_check_skipped:
            has_issues = (
                any(classification.get(lvl) for lvl, _, _ in _issue_sections) or bad_patterns
            )
            if not has_issues:
                console.print("      [green]Ready to deploy[/green]")

        console.print()


def _resolve_run_unknown_op(
    ep: str,
    device: str,
    run_unknown_op: bool,
    local_pairs: set[tuple[str, str]],
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

    try:
        from .. import winml

        for registered_ep_device in winml.get_registered_ep_devices():
            ep_name_raw = str(getattr(registered_ep_device, "ep_name", ""))
            if not ep_name_raw or ep_name_raw.endswith(".AUTO"):
                continue

            ep_name = normalize_ep_name(ep_name_raw)
            if ep_name is None or ep_name not in SUPPORTED_EPS:
                continue

            device_obj = getattr(registered_ep_device, "device", None)
            device_type = getattr(device_obj, "type", None)
            device_name = DEVICE_TYPE_TO_DEVICE.get(device_type)
            if device_name is None:
                continue

            pairs.add((ep_name, device_name))
    except Exception:
        logger.debug(
            "Failed to query local EP/device pairs via ort.get_ep_devices()",
            exc_info=True,
        )

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


def _ep_name_device_display_name(ep_name: str, device_name: str) -> str:
    """Return EP/device label for table and summary display."""
    return f"{ep_name} ({device_name.upper()})"


# ── Click command ─────────────────────────────────────────────────────────


@click.command(name="analyze")
@cli_utils.model_path_option(required=True)
@click.option(
    "--ep",
    "--execution-provider",
    required=False,
    default="auto",
    show_default=True,
    type=click.Choice([*ALL_EP_NAMES, "all", "auto"], case_sensitive=False),
    help=(
        "Target execution provider. Supports canonical names, aliases, and all/auto. "
        "all = evaluate all rule-data-backed EPs; auto = infer from local availability"
    ),
)
@click.option(
    "--device",
    required=False,
    default="auto",
    show_default=True,
    type=click.Choice([*SUPPORTED_DEVICES, "all", "auto"], case_sensitive=False),
    help=(
        "Target device type. Supports CPU/GPU/NPU and all/auto. "
        "all = all rule-data-backed devices; auto = infer from local availability"
    ),
)
@cli_utils.verbosity_options
@cli_utils.build_config_option()
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
    ep: EPNameOrAlias | Literal["all", "auto"] | None,
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
            has_rule_data_for_ep,
        )

        if device == "auto":
            from ..sysinfo.device import _get_available_devices

            devices = _get_available_devices()
        elif device == "all":
            devices = SUPPORTED_DEVICES
        else:
            devices = [device]
        devices = sorted(d.upper() for d in devices)

        if ep == "auto":
            from ..sysinfo.device import _get_available_eps

            eps = _get_available_eps()
        elif ep == "all":
            eps = SUPPORTED_EPS
        else:
            # ep is a specific EP or alias
            eps = [normalize_ep_name(ep)]

        execution_pairs = [
            (candidate_ep, candidate_device)
            for candidate_ep in eps
            for candidate_device in devices
            if candidate_ep in EP_SUPPORTED_DEVICES
            and candidate_device.lower() in EP_SUPPORTED_DEVICES[candidate_ep]
        ]
        execution_pairs = _sort_ep_device_pairs(execution_pairs)

        local_pairs = set(_get_local_ep_device_pairs())

        if device == "auto" and ep == "auto":
            execution_pairs = [pair for pair in execution_pairs if pair in local_pairs]
        elif device == "auto":
            unsupported_pairs = [pair for pair in execution_pairs if pair not in local_pairs]
            if unsupported_pairs:
                logger.warning(
                    "--device auto resolves from local availability, but --ep is pinned;"
                    " the following pairs are not available on this machine: %s",
                    ", ".join(_ep_name_device_display_name(e, d) for e, d in unsupported_pairs),
                )
        elif ep == "auto":
            unsupported_pairs = [pair for pair in execution_pairs if pair not in local_pairs]
            if unsupported_pairs:
                logger.warning(
                    "--ep auto resolves from local availability, but --device is pinned;"
                    " the following pairs are not available on this machine: %s",
                    ", ".join(_ep_name_device_display_name(e, d) for e, d in unsupported_pairs),
                )

        if not execution_pairs:
            logger.error("No EP/device combination matched the current selection.")
            sys.exit(2)

        logger.info("Analyzing model: %s", model)
        logger.info(
            "Local targets: %s",
            ", ".join(
                _ep_name_device_display_name(candidate_ep, candidate_device)
                for candidate_ep, candidate_device in local_pairs
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
        ep_instance_counts: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
        live: Live | None = None
        unknown_op_progress: Progress | None = None
        unknown_op_task_id: TaskID | None = None
        unknown_op_total_nodes = 0
        ep_counter = 0
        _no_data_eps: set[tuple[str, str]] = set()  # EP/device pairs with no op rule data
        analysis_results: list = []
        current_run_unknown_op = False
        current_op_check_skipped = False

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

        def on_ep_start(ep_name, operator_counts):
            """Called when analysis starts for a new EP."""
            nonlocal current_ep_device_pair
            nonlocal instance_counts, all_op_counts, ep_counter, live
            nonlocal unknown_op_progress, unknown_op_task_id, unknown_op_total_nodes
            nonlocal current_run_unknown_op, current_op_check_skipped

            # Finalize previous EP's Live display
            if current_ep_device_pair is not None:
                _finalize_live()
                _finalize_unknown_op_progress()
                console.print()  # blank line between EP tables

            # Reset for new EP (normalize keys to display names)
            current_ep_device_pair = (ep_name, current_device)
            all_op_counts = {_display_name(k): v for k, v in operator_counts.items()}
            instance_counts = {}

            has_rule_data = has_rule_data_for_ep(ep_name, current_device)
            current_op_check_skipped = not has_rule_data and not current_run_unknown_op

            # Skip OP CHECK display for EPs with no rule data —
            # op results would all be 0/0/0 (unknown). Pattern detection
            # still runs; results appear in the ANALYSIS SUMMARY.
            if not has_rule_data:
                _no_data_eps.add((ep_name, current_device))

                if current_run_unknown_op:
                    ep_counter += 1
                    total_nodes = sum(operator_counts.values())
                    unknown_op_total_nodes = max(0, total_nodes)

                    console.print("─" * 80)
                    console.print(
                        f"💻 [bold]EP {ep_counter}[/bold]: [bold cyan]{ep_name}[/bold cyan] "
                        f"on [bold]{current_device}[/bold]"
                    )
                    console.print("─" * 80)
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
                    )
                    unknown_op_progress.start()
                    unknown_op_task_id = unknown_op_progress.add_task(
                        "unknown-op",
                        total=max(1, unknown_op_total_nodes),
                    )
                    return

            ep_counter += 1

            # EP section header
            console.print("─" * 80)
            console.print(
                f"💻 [bold]EP {ep_counter}[/bold]: [bold cyan]{ep_name}[/bold cyan] "
                f"on [bold]{current_device}[/bold]"
            )
            console.print("─" * 80)

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

                    run_unknown_op_for_ep = _resolve_run_unknown_op(
                        target_ep, target_device, run_unknown_op, local_pairs
                    )

                    current_run_unknown_op = run_unknown_op_for_ep

                    result = analyzer.analyze(
                        model_path=str(model),
                        ep=target_ep,
                        device=target_device,
                        enable_information=information,
                        htp_metadata_path=str(htp_metadata) if htp_metadata else None,
                        run_unknown_op=run_unknown_op_for_ep,
                        save_node_types=save_node_types,
                        on_node_result=on_node_result,
                        on_ep_start=on_ep_start,
                    )
                    analysis_results.append(result)

                    # Extract per-EP pattern support (available now)
                    ep_patterns = _extract_ep_patterns(result.output.results)

                    # Finalize last EP's Live display
                    _finalize_live()
                    _finalize_unknown_op_progress()

                    console.print()

                    # Pattern Matching section (per-EP)
                    _render_pattern_matching(console, ep_patterns)

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
            finally:
                # Safety: stop Live if still running (e.g. on exception)
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
                    htp_metadata_path=str(htp_metadata) if htp_metadata else None,
                    run_unknown_op=run_unknown_op_for_ep,
                    save_node_types=save_node_types,
                )
                analysis_results.append(result)

        result = analysis_results[-1]

        # Save JSON if requested
        if output:
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                if len(analysis_results) == 1:
                    output.write_text(result.to_json(), encoding="utf-8")
                else:
                    import json

                    output.write_text(
                        json.dumps(
                            [json.loads(run_result.to_json()) for run_result in analysis_results]
                        ),
                        encoding="utf-8",
                    )
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

        # Exit code: 0 = fully supported, 1 = partial support
        overall_supported = all(run_result.is_fully_supported() for run_result in analysis_results)
        sys.exit(0 if overall_supported else 1)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(2)
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(2)


__all__ = ["analyze"]
