# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Console writer for static analyzer results.

This module provides real-time console output using Rich library,
displaying analysis results in a user-friendly format.
"""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .models.output import AnalysisOutput
from .models.support_level import SupportLevel


if TYPE_CHECKING:
    from .models.information import Information
    from .models.output import EPSupport


class StaticAnalyzerConsoleWriter:
    """Console writer for displaying static analyzer results."""

    # Configuration constants
    SEPARATOR_LENGTH = 80
    CONSOLE_WIDTH = 80  # Match typical terminal width

    def __init__(self, console: Console | None = None, verbose: bool = True) -> None:
        """Initialize console writer.

        Args:
            console: Rich console instance (optional)
            verbose: Whether to output verbose information
        """
        self.verbose = verbose
        self.console = console or Console(
            width=self.CONSOLE_WIDTH,
            force_terminal=True,
            legacy_windows=False,
            highlight=False,
        )
        # Ensure we have a reliable width value (Rich may return None in some cases)
        self.width = self.CONSOLE_WIDTH if not self.console.width else self.console.width

    def _bright_cyan(self, text: str | int | float) -> str:
        """Format text in bright cyan."""
        return f"[bold cyan]{text}[/bold cyan]"

    def _bright_green(self, text: str) -> str:
        """Format text in bright green."""
        return f"[bold green]{text}[/bold green]"

    def _bright_red(self, text: str) -> str:
        """Format text in bright red."""
        return f"[bold red]{text}[/bold red]"

    def _bright_yellow(self, text: str) -> str:
        """Format text in bright yellow."""
        return f"[bold yellow]{text}[/bold yellow]"

    def _bold(self, text: str) -> str:
        """Format text in bold."""
        return f"[bold]{text}[/bold]"

    def _dim(self, text: str) -> str:
        """Format text in dim style."""
        return f"[dim]{text}[/dim]"

    def write_analysis_results(self, analysis: AnalysisOutput) -> None:
        """Write complete analysis results to console.

        Args:
            analysis: AnalysisOutput containing all analysis results
        """
        self._write_header(analysis)
        self._write_model_info(analysis)
        self._write_operator_summary(analysis)

        if analysis.metadata.detected_pattern_count:
            self._write_pattern_summary(analysis)

        self._write_ihv_results(analysis)
        self._write_footer(analysis)

    def _write_header(self, analysis: AnalysisOutput) -> None:
        """Write analysis header."""
        self.console.print()
        self.console.print("=" * self.SEPARATOR_LENGTH)
        self.console.print(f"📊 {self._bold('ONNX MODEL STATIC ANALYSIS REPORT')}")
        self.console.print("=" * self.SEPARATOR_LENGTH)

        timestamp = analysis.analysis_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self.console.print(f"🕒 Analysis Time: {self._dim(timestamp)}")
        self.console.print(f"📦 Model: {self._bright_cyan(analysis.metadata.model_path)}")
        self.console.print()

    def _write_model_info(self, analysis: AnalysisOutput) -> None:
        """Write model information section."""
        self.console.print(f"📋 {self._bold('MODEL INFORMATION')}")
        self.console.print("-" * self.SEPARATOR_LENGTH)

        metadata = analysis.metadata
        self.console.print(f"   • ONNX Opset: {self._bright_green(metadata.opset_version)}")

        if metadata.producer_name:
            producer = f"{metadata.producer_name}"
            if metadata.producer_version:
                producer += f" v{metadata.producer_version}"
            self.console.print(f"   • Producer: {self._bright_green(producer)}")

        self.console.print(
            f"   • Total Operators: {self._bright_cyan(f'{metadata.total_operators:,}')}"
        )
        self.console.print(
            f"   • Unique Types: {self._bright_cyan(metadata.unique_operator_types)}"
        )

        if metadata.detected_pattern_count:
            total_patterns = sum(metadata.detected_pattern_count.values())
            self.console.print(
                f"   • Detected Patterns: {self._bright_cyan(total_patterns)} instances "
                f"({self._bright_cyan(len(metadata.detected_pattern_count))} types)"
            )
        self.console.print()

    def _write_operator_summary(self, analysis: AnalysisOutput) -> None:
        """Write operator summary with top operators."""
        self.console.print(f"🔧 {self._bold('OPERATOR ANALYSIS')}")
        self.console.print("-" * self.SEPARATOR_LENGTH)

        metadata = analysis.metadata
        sorted_ops = sorted(metadata.operator_counts.items(), key=lambda x: x[1], reverse=True)

        # Create table
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Operator Type", style="green")
        table.add_column("Count", justify="right", style="cyan")
        table.add_column("Percentage", justify="right", style="yellow")

        for rank, (op_type, count) in enumerate(sorted_ops, 1):
            percentage = (
                (count / metadata.total_operators * 100) if metadata.total_operators > 0 else 0
            )
            table.add_row(f"{rank}.", op_type, f"{count:,}", f"{percentage:.1f}%")

        self.console.print(table)
        self.console.print()

    def _write_pattern_summary(self, analysis: AnalysisOutput) -> None:
        """Write pattern detection summary."""
        self.console.print(f"🔍 {self._bold('PATTERN DETECTION')}")
        self.console.print("-" * self.SEPARATOR_LENGTH)

        detected = analysis.metadata.detected_pattern_count
        total_instances = sum(detected.values())

        self.console.print(
            f"   Detected {self._bright_cyan(total_instances)} pattern instances "
            f"across {self._bright_cyan(len(detected))} pattern types"
        )
        self.console.print()

        # Show pattern details
        sorted_patterns = sorted(detected.items(), key=lambda x: x[1], reverse=True)
        for pattern_id, count in sorted_patterns:
            pattern_type = "🔸 Subgraph" if pattern_id.startswith("SUBGRAPH/") else "🔹 Operator"
            self.console.print(
                f"   {pattern_type}: {self._bright_green(pattern_id)} "
                f"({self._bright_cyan(count)} instances)"
            )
        self.console.print()

    def _write_ihv_results(self, analysis: AnalysisOutput) -> None:
        """Write IHV support analysis results."""
        self.console.print(f"💻 {self._bold('IHV PLATFORM SUPPORT ANALYSIS')}")
        self.console.print("=" * self.SEPARATOR_LENGTH)

        for ihv_result in analysis.results:
            self._write_single_ihv_result(
                ihv_result,
                total_operators=analysis.metadata.total_operators,
                unique_operator_types=analysis.metadata.unique_operator_types,
            )
            self.console.print()

    def _write_single_ihv_result(
        self, ihv_result: EPSupport, total_operators: int, unique_operator_types: int
    ) -> None:
        """Write support analysis for a single IHV platform.

        Args:
            ihv_result: IHV support result
            total_operators: Total number of operators in model
            unique_operator_types: Total number of unique operator types
        """
        # Header with support status
        status_icon = "✅" if ihv_result.runtime_support else "❌"
        status_text = (
            self._bright_green("SUPPORTED")
            if ihv_result.runtime_support
            else self._bright_red("NOT SUPPORTED")
        )

        self.console.print(
            f"\n{status_icon} {self._bold(ihv_result.ihv_type.value)} - {status_text}"
        )
        self.console.print("-" * 60)

        # Version info
        if ihv_result.ep_version:
            self.console.print(f"   EP Version: {self._dim(ihv_result.ep_version)}")
        if ihv_result.driver_version:
            self.console.print(f"   Driver Version: {self._dim(ihv_result.driver_version)}")

        # Show EP Type and Device
        self.console.print(f"   EP: {self._bright_cyan(ihv_result.ep_type)}")
        if ihv_result.device_type:
            self.console.print(f"   Device: {self._bright_cyan(ihv_result.device_type)}")

        # Support classification
        self.console.print(f"\n   {self._bold('Support Classification:')}")

        classification_info = [
            (SupportLevel.SUPPORTED, "✅", "Fully Supported", "green"),
            (SupportLevel.PARTIAL, "⚠️ ", "Partial Support", "yellow"),
            (SupportLevel.UNKNOWN, "❓", "Unknown Support", "blue"),
            (SupportLevel.UNSUPPORTED, "⛔", "Not Supported", "red"),
        ]

        for level, icon, label, color in classification_info:
            if level in ihv_result.classification:
                operators = ihv_result.classification[level]
                count = len(operators)
                # Calculate percentage based on unique operator types, not total instances
                percentage = (
                    (count / unique_operator_types * 100) if unique_operator_types > 0 else 0
                )

                count_str = f"[{color}]{count:3d}[/{color}]"
                pct_str = f"[{color}]{percentage:5.1f}%[/{color}]"

                self.console.print(f"   {icon} {label:20s}: {count_str} operator types ({pct_str})")

                # Show operators - expand SUPPORTED level, show samples for others
                if count > 0:
                    if level == SupportLevel.SUPPORTED:
                        # Expand all fully supported operators
                        for op in operators:
                            self.console.print(f"      • {self._bright_green(op)}")
                    else:
                        # Show all operators for non-supported levels
                        for op in operators:
                            self.console.print(f"      • {self._dim(op)}")

        # Information summary
        info_count = len(ihv_result.information)
        if info_count > 0:
            self.console.print(
                f"\n   💡 {self._bright_yellow('Actionable Information')}: "
                f"{self._bright_cyan(info_count)} items"
            )

            self._write_information_items(ihv_result.information, ihv_result.ihv_type.value)

    def _format_wrapped_text(
        self,
        text: str,
        indent: str = "      ",
        first_line_indent: str = "",
        width: int | None = None,
    ) -> list[str]:
        """Wrap long text with proper indentation.

        Args:
            text: Text to wrap
            indent: Indentation string for continuation lines
            first_line_indent: Indentation for first line (default: empty)
            width: Maximum width of each line (default: console width - padding)

        Returns:
            List of formatted lines with proper indentation
        """
        # Use console width if not specified, with padding for safety
        if width is None:
            width = self.width - 10

        # Wrap the text
        wrapper = textwrap.TextWrapper(
            width=width,
            initial_indent=first_line_indent,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )

        return wrapper.wrap(text)

    def _write_information_items(self, information_list: list[Information], ihv_name: str) -> None:
        """Write detailed information items.

        Args:
            information_list: List of Information objects
            ihv_name: IHV platform name
        """
        for idx, info in enumerate(information_list, 1):
            self.console.print()

            # Format issue header with explanation
            issue_text = f"Issue #{idx}:"
            full_text = f"{issue_text} {info.explanation}"

            wrapped_lines = self._format_wrapped_text(
                full_text,
                indent="   ",
                first_line_indent="   ",
            )

            # Print wrapped lines with formatting
            if wrapped_lines:
                # Format first line with Issue header as bold
                first_line = wrapped_lines[0].strip()
                issue_end = first_line.find(":") + 1
                if issue_end > 0:
                    issue_part = first_line[:issue_end]
                    rest_part = first_line[issue_end:].lstrip()
                    self.console.print(f"   {self._bold(issue_part)} {rest_part}")
                else:
                    self.console.print(f"   {first_line}")

                # Print remaining lines
                for line in wrapped_lines[1:]:
                    self.console.print(line)

            self.console.print()

            # Pattern info
            if info.pattern_id:
                self.console.print(
                    f"      {self._bold('Pattern:')} {self._bright_cyan(info.pattern_id)}"
                )

            # Show affected nodes with details
            if info.pattern_node_list:
                instance_count = len(info.pattern_node_list)
                total_nodes = sum(len(nodes) for nodes in info.pattern_node_list)
                self.console.print(
                    f"      {self._bold('Affected:')} {self._bright_cyan(instance_count)} pattern instances, "
                    f"{self._bright_cyan(total_nodes)} total nodes"
                )

                # Show first 3 pattern instances with node lists
                max_patterns_to_show = 3
                for pattern_idx, node_list in enumerate(
                    info.pattern_node_list[:max_patterns_to_show], 1
                ):
                    if node_list:
                        self.console.print(f"         Instance {pattern_idx}:")
                        # Show all nodes for readability
                        for i, node in enumerate(node_list):
                            node_escaped = escape(node)
                            self.console.print(
                                f"            {self._dim(f'{i + 1}.')} {self._bright_green(node_escaped)}"
                            )

                if instance_count > max_patterns_to_show:
                    self.console.print(
                        f"         {self._dim(f'... and {instance_count - max_patterns_to_show} more pattern instances')}"
                    )
                self.console.print()

            # Show actions with transformation details
            if info.actions:
                self.console.print(f"      {self._bold('Recommended Actions:')}")
                for action_idx, action in enumerate(info.actions, 1):
                    # Show transformation: from_pattern -> to_pattern
                    transformation = (
                        f"{self._bright_yellow(action.pattern_from_id)} → "
                        f"{self._bright_green(action.pattern_to_id)}"
                    )
                    priority_str = (
                        f" {self._bright_red(f'[{action.level.value.upper()}]')}"
                        if action.level
                        else ""
                    )

                    self.console.print(
                        f"         {action_idx}. {self._bold('Transform:')} {transformation}{priority_str}"
                    )

                    # Show expected status after transformation
                    if action.status:
                        status_icon = {
                            "WHITE": "✅",
                            "GRAY": "⚠️",
                            "UNKNOWN": "❓",
                            "BLACK": "⛔",
                        }.get(action.status.value, "•")
                        self.console.print(
                            f"            {self._bold('Expected Result:')} {status_icon} {self._bright_green(action.status.value)}"
                        )

                    # Show action details
                    if action.details:
                        self.console.print(f"            {self._bold('Details:')}")

                        # Try to parse as JSON for better formatting
                        try:
                            details_obj = json.loads(action.details)

                            # If it's a list or dict, display as formatted JSON
                            if isinstance(details_obj, (list, dict)):
                                json_str = json.dumps(details_obj, indent=2, ensure_ascii=False)
                                # Display formatted JSON with proper indentation
                                indent_prefix = "               "
                                for line in json_str.split("\n"):
                                    self.console.print(f"{indent_prefix}{line}")
                            else:
                                # If it's a string value, just display it
                                wrapped_lines = self._format_wrapped_text(
                                    str(details_obj),
                                    indent="               ",
                                    first_line_indent="               ",
                                )
                                for line in wrapped_lines:
                                    self.console.print(self._dim(line))
                        except (json.JSONDecodeError, ValueError):
                            # If not valid JSON, treat as plain text
                            wrapped_lines = self._format_wrapped_text(
                                action.details,
                                indent="               ",
                                first_line_indent="               ",
                            )
                            for line in wrapped_lines:
                                self.console.print(self._dim(line))

                    # Show action items (transformations/optimizations)
                    if action.action_items:
                        self.console.print(f"            {self._bold('Steps:')}")
                        for item in action.action_items:
                            opt_str = ""
                            if item.optimization_options:
                                # Show all options
                                opts = ", ".join(
                                    f"{k}={v}" for k, v in item.optimization_options.items()
                                )
                                opt_str = f" {self._dim(f'({opts})')}"
                            self.console.print(
                                f"               • {self._bright_cyan(item.type)}{opt_str}"
                            )
                    self.console.print()

    def _write_footer(self, analysis: AnalysisOutput) -> None:
        """Write analysis footer with summary."""
        self.console.print("=" * self.SEPARATOR_LENGTH)
        self.console.print(f"📈 {self._bold('ANALYSIS SUMMARY')}")
        self.console.print("-" * self.SEPARATOR_LENGTH)

        # Overall support status
        supported_platforms = sum(1 for r in analysis.results if r.runtime_support)
        total_platforms = len(analysis.results)

        # Check if unsupported platforms only have unknown nodes (no unsupported/partial)
        unsupported_results = [r for r in analysis.results if not r.runtime_support]
        has_only_unknown = False
        if unsupported_results:
            # Check if there are NO unsupported or partial issues (only unknown/supported)
            # Use .get() to handle missing keys and check for non-empty lists
            has_only_unknown = all(
                not r.classification.get(SupportLevel.UNSUPPORTED, [])
                and not r.classification.get(SupportLevel.PARTIAL, [])
                for r in unsupported_results
            )

        if supported_platforms == total_platforms:
            status_msg = self._bright_green(
                f"✅ Model is supported on all {total_platforms} platform(s)"
            )
        elif supported_platforms > 0 and not has_only_unknown:
            status_msg = self._bright_yellow(
                f"⚠️ Model is supported on {supported_platforms}/{total_platforms} platform(s)"
            )
        elif supported_platforms > 0 and has_only_unknown:
            status_msg = self._bright_yellow(
                f"⚠️ Model is supported on {supported_platforms}/{total_platforms} platform(s), "
                f"unknown nodes found on some of platforms"
            )
        elif has_only_unknown:
            status_msg = self._bright_yellow("⚠️  Model has unknown nodes")
        else:
            status_msg = self._bright_red("❌ Model is not supported on any platform")

        self.console.print(f"   {status_msg}")

        # Show platform-specific summaries
        for ep_result in analysis.results:
            platform_name = ep_result.ep_type
            if ep_result.runtime_support:
                self.console.print(f"   • {self._bright_green(platform_name)}: Ready to deploy")
            else:
                # Count issues
                issue_counts = {
                    level: len(ops)
                    for level, ops in ep_result.classification.items()
                    if level != SupportLevel.SUPPORTED
                }

                if any(issue_counts.values()):
                    # Check if only unknown nodes (no black or gray)
                    has_only_unknown = all(
                        level == SupportLevel.UNKNOWN
                        for level in issue_counts
                    )

                    issue_summary = ", ".join(
                        f"{count} {level.value}"
                        for level, count in issue_counts.items()
                        if count > 0
                    )

                    if has_only_unknown:
                        self.console.print(
                            f"   • {self._bright_yellow(platform_name)}: Unknown nodes found ({issue_summary})"
                        )
                    else:
                        self.console.print(
                            f"   • {self._bright_red(platform_name)}: Issues found ({issue_summary})"
                        )

        self.console.print()
        self.console.print(
            f"💡 Use {self._bright_cyan('--output results.json')} to save detailed results"
        )
        self.console.print("=" * self.SEPARATOR_LENGTH)
        self.console.print()


def display_analysis_results(
    analysis: AnalysisOutput,
    console: Console | None = None,
    verbose: bool = True,
) -> None:
    """Display analysis results in console.

    Args:
        analysis: AnalysisOutput containing analysis results
        console: Optional Rich console instance
        verbose: Whether to show verbose output
    """
    writer = StaticAnalyzerConsoleWriter(console=console, verbose=verbose)
    writer.write_analysis_results(analysis)
