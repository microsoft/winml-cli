# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# ruff: noqa: RUF001, SIM108
# RUF001: info emoji used intentionally in user-facing output
# SIM108: explicit if-else is clearer for params formatting
"""Console writer for HTP export monitoring.

This module provides real-time console output using Rich library,
matching the exact format of the baseline implementation.
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.tree import Tree

from ...core.hierarchy_utils import build_rich_tree
from .base_writer import ExportData, ExportStep, StepAwareWriter, step
from .step_data import HIERARCHY_SOURCE_ONNX_METADATA


if TYPE_CHECKING:
    from .step_data import NodeTaggingData


class ConsoleWriter(StepAwareWriter):
    """Real-time console output with Rich formatting."""

    # Configuration constants matching baseline
    MAX_HIERARCHY_LINES = 30
    TOP_NODES_COUNT = 20
    SEPARATOR_LENGTH = 80
    CONSOLE_WIDTH = 120

    def __init__(self, console: Console | None = None, verbose: bool = True) -> None:
        """Initialize console writer.

        Args:
            console: Rich console instance (optional)
            verbose: Whether to output verbose information
        """
        super().__init__()
        self.verbose = verbose

        # Set UTF-8 encoding environment variable as a fallback
        if not os.environ.get("PYTHONIOENCODING"):
            os.environ["PYTHONIOENCODING"] = "utf-8:replace"

        self.console = console or Console(
            width=self.CONSOLE_WIDTH,
            force_terminal=True,
            legacy_windows=False,
            highlight=False,  # Disable automatic highlighting
        )
        self._current_step = 0
        self._total_steps = 6

    def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
        """Default handler - do nothing for unhandled steps."""
        return 0

    # Styling utilities matching baseline
    def _bright_cyan(self, text: str | int | float) -> str:
        """Format text in bright cyan."""
        return f"[bold cyan]{text}[/bold cyan]"

    def _bright_green(self, text: str) -> str:
        """Format text in bright green."""
        return f"[bold green]{text}[/bold green]"

    def _bright_red(self, text: str) -> str:
        """Format text in bright red."""
        return f"[bold red]{text}[/bold red]"

    def _bright_magenta(self, text: str) -> str:
        """Format text in bright magenta."""
        return f"[bold magenta]{text}[/bold magenta]"

    def _bright_yellow(self, text: str) -> str:
        """Format text in bright yellow."""
        return f"[bold yellow]{text}[/bold yellow]"

    def _bold(self, text: str) -> str:
        """Format text in bold."""
        return f"[bold]{text}[/bold]"

    def _dim(self, text: str) -> str:
        """Format text in dim style."""
        return f"[dim]{text}[/dim]"

    def _format_bool(self, value: bool) -> str:
        """Format boolean with color."""
        return self._bright_green("True") if value else self._bright_red("False")

    @step(ExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: ExportStep, data: ExportData) -> int:
        """Step 1: Model preparation."""
        if not data.model_prep:
            return 0

        self._current_step = 1
        step_header = f"STEP {self._current_step}/{self._total_steps}: MODEL PREPARATION"
        self.console.print(f"\n📋 {self._bold(step_header)}")
        self.console.print("=" * self.SEPARATOR_LENGTH)

        # Format parameters to match baseline (e.g., 4.4M not 4.4M)
        params_m = data.model_prep.total_parameters / 1e6
        if params_m == int(params_m):
            params_str = f"{int(params_m)}"
        else:
            params_str = f"{params_m:.1f}"

        self.console.print(
            f"✅ Model loaded: {data.model_prep.model_class} "
            f"({self._bright_cyan(data.model_prep.total_modules)} modules, "
            f"{self._bright_cyan(params_str + 'M')} parameters)"
        )
        self.console.print(f"🎯 Export target: {self._bright_magenta(data.output_path)}")
        self.console.print("✅ Model set to evaluation mode")

        return 1

    @step(ExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: ExportStep, data: ExportData) -> int:
        """Step 2: Input generation."""
        if not data.input_gen:
            return 0

        self._current_step = 2
        self.console.print(
            f"\n🔧 {self._bold(f'STEP {self._current_step}/{self._total_steps}: INPUT GENERATION')}"
        )
        self.console.print("=" * self.SEPARATOR_LENGTH)

        if data.input_gen.method == "provided":
            self.console.print("📝 Using provided input specifications")
        else:
            self.console.print(f"🤖 Auto-generating inputs for: {data.model_name}")
            if data.input_gen.model_type:
                self.console.print(
                    f"   • Model type: {self._bright_green(data.input_gen.model_type)}"
                )
            if data.input_gen.task:
                self.console.print(f"   • Detected task: {self._bright_green(data.input_gen.task)}")

        # Display input details
        if data.input_gen.inputs:
            self.console.print("✅ Generated inputs:")
            for name, tensor_info in data.input_gen.inputs.items():
                shape_str = str(tensor_info.shape)
                self.console.print(
                    f"   • {name}: shape={self._bright_green(shape_str)}, "
                    f"dtype={self._bright_green(tensor_info.dtype)}"
                )

        return 1

    @step(ExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: ExportStep, data: ExportData) -> int:
        """Step 3: Hierarchy building."""
        if not data.hierarchy:
            return 0

        self._current_step = 3
        step_header = f"STEP {self._current_step}/{self._total_steps}: HIERARCHY BUILDING"
        self.console.print(f"\n🏗️ {self._bold(step_header)}")
        self.console.print("=" * self.SEPARATOR_LENGTH)

        if data.hierarchy.source == HIERARCHY_SOURCE_ONNX_METADATA:
            # Dynamo path: no forward trace ran; the hierarchy is reconstructed
            # from ONNX node metadata, so avoid trace/execution-step wording.
            self.console.print("🔍 Reconstructing module hierarchy from ONNX node metadata...")
            self.console.print(
                f"✅ Recovered {self._bright_cyan(len(data.hierarchy.hierarchy))} modules "
                "in hierarchy"
            )
        else:
            self.console.print("🔍 Tracing module execution with dummy inputs...")
            self.console.print(
                f"✅ Traced {self._bright_cyan(len(data.hierarchy.hierarchy))} modules in hierarchy"
            )
            if data.hierarchy.execution_steps is not None:
                self.console.print(
                    f"📊 Total execution steps: {self._bright_cyan(data.hierarchy.execution_steps)}"
                )

        # Build and display hierarchy tree
        if data.hierarchy.hierarchy:
            self.console.print("\n🌳 Module Hierarchy:")
            tree = build_rich_tree(data.hierarchy.hierarchy, show_counts=False)
            self._display_truncated_tree(tree)

        return 1

    @step(ExportStep.ONNX_EXPORT)
    def write_onnx_export(self, export_step: ExportStep, data: ExportData) -> int:
        """Step 4: ONNX export."""
        if not data.onnx_export:
            return 0

        self._current_step = 4
        self.console.print(
            f"\n📦 {self._bold(f'STEP {self._current_step}/{self._total_steps}: ONNX EXPORT')}"
        )
        self.console.print("=" * self.SEPARATOR_LENGTH)

        self.console.print("🔧 Export configuration:")
        self.console.print(
            f"   • Opset version: {self._bright_green(str(data.onnx_export.opset_version))}"
        )
        self.console.print(
            f"   • Constant folding: {self._format_bool(data.onnx_export.do_constant_folding)}"
        )

        if data.onnx_export.input_names:
            formatted_inputs = (
                "["
                + ", ".join(
                    self._bright_green(f"'{name}'") for name in data.onnx_export.input_names
                )
                + "]"
            )
            self.console.print(f"📥 Input names: {formatted_inputs}")

        if data.onnx_export.output_names:
            formatted_outputs = (
                "["
                + ", ".join(
                    self._bright_green(f"'{name}'") for name in data.onnx_export.output_names
                )
                + "]"
            )
            self.console.print(f"📤 Output names: {formatted_outputs}")
        else:
            self.console.print(
                f"📤 Output names: {self._bright_yellow('Not detected')} "
                "(model may not have named outputs)"
            )

        self.console.print("✅ ONNX model exported successfully")
        self.console.print(
            f"📦 Model size: {self._bright_cyan(f'{data.onnx_export.onnx_size_mb:.2f}MB')}"
        )

        return 1

    @step(ExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: ExportStep, data: ExportData) -> int:
        """Step 5: Node tagging."""
        if not data.node_tagging:
            return 0

        self._current_step = 5
        step_header = f"STEP {self._current_step}/{self._total_steps}: ONNX NODE TAGGING"
        self.console.print(f"\n🔗 {self._bold(step_header)}")
        self.console.print("=" * self.SEPARATOR_LENGTH)

        self.console.print("✅ Node tagging completed successfully")
        self.console.print(
            f"📈 Coverage: {self._bright_cyan(f'{data.node_tagging.coverage:.1f}%')}"
        )
        self.console.print(
            f"📊 Tagged nodes: {self._bright_cyan(len(data.node_tagging.tagged_nodes))}"
            f"/{self._bright_cyan(data.node_tagging.total_nodes)}"
        )

        # Display tagging statistics
        self._display_tagging_statistics(data.node_tagging)

        # Display nodes by hierarchy (verbose only)
        if self.verbose:
            self._display_nodes_by_hierarchy(data.node_tagging.tagged_nodes)

        # Display hierarchy tree with node counts
        if data.hierarchy and data.hierarchy.hierarchy and data.node_tagging.tagged_nodes:
            self.console.print("\n🌳 Complete HF Hierarchy with ONNX Nodes:")
            self.console.print("-" * 60)
            tree = build_rich_tree(
                data.hierarchy.hierarchy,
                show_counts=True,
                tagged_nodes=data.node_tagging.tagged_nodes,
            )
            self._display_truncated_tree(tree)

        return 1

    @step(ExportStep.TAG_INJECTION)
    def write_tag_injection(self, export_step: ExportStep, data: ExportData) -> int:
        """Step 6: Tag injection."""
        if not data.tag_injection:
            return 0

        self._current_step = 6
        self.console.print(
            f"\n🏷️ {self._bold(f'STEP {self._current_step}/{self._total_steps}: TAG INJECTION')}"
        )
        self.console.print("=" * self.SEPARATOR_LENGTH)

        if data.tag_injection.tags_injected:
            self.console.print("🔧 Injecting hierarchy tags into ONNX model...")
            self.console.print("✅ Tags successfully embedded as node attributes")
        else:
            self.console.print("ℹ️ Hierarchy tag injection skipped (clean ONNX by default)")

        self.console.print(f"💾 Model saved to: {self._bright_magenta(data.output_path)}")

        return 1

    def _display_truncated_tree(self, tree: Tree, max_lines: int | None = None) -> None:
        """Display a tree with optional truncation."""
        if max_lines is None:
            max_lines = self.MAX_HIERARCHY_LINES

        # Create a temporary console to capture the tree output
        string_buffer = io.StringIO()
        temp_console = Console(
            file=string_buffer,
            width=self.CONSOLE_WIDTH,
            force_terminal=False,
            legacy_windows=False,
            highlight=False,
        )
        temp_console.print(tree)

        # Get lines and apply truncation
        lines = string_buffer.getvalue().strip().split("\n")
        if len(lines) <= max_lines:
            self.console.print(tree)
        else:
            # Create truncated tree
            truncated_tree = Tree(tree.label)
            self._build_truncated_tree(tree, truncated_tree, max_lines - 1)
            self.console.print(truncated_tree)
            self.console.print(
                f"... showing first {max_lines} lines ({self._bold('truncated for console')})"
            )

    def _build_truncated_tree(self, source_tree: Tree, target_tree: Tree, max_lines: int) -> int:
        """Build a truncated version of the tree that fits within max_lines."""
        line_count = 1  # Start with root

        # Helper to add nodes up to limit
        def add_nodes_to_limit(source_children: Any, target_parent: Any, current_count: int) -> int:
            count = current_count
            for child in source_children:
                if count >= max_lines:
                    break
                # Add this child
                target_child = target_parent.add(child.label)
                count += 1

                # Try to add its children
                if hasattr(child, "children") and child.children and count < max_lines:
                    count = add_nodes_to_limit(child.children, target_child, count)
            return count

        # Add nodes from source to target
        if hasattr(source_tree, "children") and source_tree.children:
            line_count = add_nodes_to_limit(source_tree.children, target_tree, line_count)

        return line_count

    def _display_tagging_statistics(self, tagging_data: NodeTaggingData) -> None:
        """Display tagging statistics with percentages."""
        stats = tagging_data.tagging_stats
        total = tagging_data.total_nodes

        if stats and total > 0:
            direct = stats.get("direct_matches", 0)
            parent = stats.get("parent_matches", 0)
            root = stats.get("root_fallbacks", 0)

            direct_pct = (direct / total * 100) if total > 0 else 0
            parent_pct = (parent / total * 100) if total > 0 else 0
            root_pct = (root / total * 100) if total > 0 else 0

            self.console.print(
                f"   • Direct matches: {self._bright_cyan(direct)} "
                f"({self._bright_cyan(f'{direct_pct:.1f}%')})"
            )
            self.console.print(
                f"   • Parent matches: {self._bright_cyan(parent)} "
                f"({self._bright_cyan(f'{parent_pct:.1f}%')})"
            )
            self.console.print(
                f"   • Root fallbacks: {self._bright_cyan(root)} "
                f"({self._bright_cyan(f'{root_pct:.1f}%')})"
            )

        self.console.print(f"✅ Empty tags: {self._bright_cyan('0')}")

    def _display_nodes_by_hierarchy(self, tagged_nodes: dict[str, str]) -> None:
        """Display top nodes grouped by hierarchy."""
        if not tagged_nodes:
            return

        from collections import Counter

        # Count nodes by hierarchy tag
        tag_counts = Counter(tagged_nodes.values())

        self.console.print(
            f"\n📊 Top {self._bright_cyan(self.TOP_NODES_COUNT)} Nodes by Hierarchy:"
        )
        self.console.print("-" * 30)

        sorted_tags = tag_counts.most_common(self.TOP_NODES_COUNT)
        for i, (tag, count) in enumerate(sorted_tags):
            self.console.print(f" {i + 1:2d}. {tag}: {self._bright_cyan(count)} nodes")

    def _extract_operation_type(self, node_name: str) -> str:
        """Extract operation type from node name."""
        # Extract base operation name (e.g., "/embeddings/Add_0" -> "Add")
        base_name = node_name.split("/")[-1]  # Get last part of path
        if "_" in base_name:
            return base_name.split("_")[0]
        # For names without underscore
        return base_name
