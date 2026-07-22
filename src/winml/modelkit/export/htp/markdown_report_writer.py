# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Markdown report writer for HTP export monitoring.

This module provides markdown report generation independent of console output,
following the specification in HTP_METADATA_REPORT_SPEC.md.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import snakemd

from ...core.hierarchy_utils import (
    build_ascii_tree,
    count_direct_and_total_nodes,
    count_nodes_per_tag,
)
from ...core.time_utils import format_timestamp_iso
from .base_writer import ExportData, ExportStep, StepAwareWriter, step
from .step_data import HIERARCHY_SOURCE_ONNX_METADATA


class MarkdownReportWriter(StepAwareWriter):
    """Markdown report writer that generates reports independently from console output.

    The generated markdown uses GitHub-flavored markdown with HTML details/summary
    tags for collapsible sections.
    """

    # Report string constants for potential i18n
    TITLE = "HTP ONNX Export Report"
    TITLE_ERROR = "HTP ONNX Export Report - Error"
    SECTION_EXPORT_STEPS = "Export Process"
    SECTION_MODULE_HIERARCHY = "Module Hierarchy"
    SECTION_NODE_MAPPINGS = "Complete Node Mappings"
    SECTION_SUMMARY = "Export Summary"

    def __init__(self, output_path: str) -> None:
        """Initialize markdown report writer.

        Args:
            output_path: Base output path for the ONNX model
        """
        super().__init__()
        self.output_path = Path(output_path).with_suffix("").as_posix()
        self.report_path = f"{self.output_path}_htp_export_report.md"

        # Initialize SnakeMD document
        self.doc = snakemd.new_doc()

        # Store step data for final report generation
        self._step_results: dict[ExportStep, dict[str, Any]] = {}
        self._start_time = time.time()
        self._export_data: ExportData | None = None  # Will be set on first write
        self._report_generated = False  # Track if report was generated

    def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
        """Default handler - record step completion."""
        self._step_results[export_step] = {
            "completed": True,
            "timestamp": format_timestamp_iso(data.get_step_timestamp(export_step)),
        }
        return 0

    @step(ExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: ExportStep, data: ExportData) -> int:
        """Record model preparation data."""
        # Store reference to ExportData for use in flush()
        if self._export_data is None:
            self._export_data = data

        if data.model_prep:
            self._step_results[export_step] = {
                "completed": True,
                "timestamp": format_timestamp_iso(data.get_step_timestamp(export_step)),
                "data": data.model_prep,
            }
        return 1

    @step(ExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: ExportStep, data: ExportData) -> int:
        """Record input generation data."""
        if data.input_gen:
            self._step_results[export_step] = {
                "completed": True,
                "timestamp": format_timestamp_iso(data.get_step_timestamp(export_step)),
                "data": data.input_gen,
            }
        return 1

    @step(ExportStep.HIERARCHY)
    def write_hierarchy(self, export_step: ExportStep, data: ExportData) -> int:
        """Record hierarchy building data."""
        if data.hierarchy:
            self._step_results[export_step] = {
                "completed": True,
                "timestamp": format_timestamp_iso(data.get_step_timestamp(export_step)),
                "data": data.hierarchy,
            }
        return 1

    @step(ExportStep.ONNX_EXPORT)
    def write_onnx_export(self, export_step: ExportStep, data: ExportData) -> int:
        """Record ONNX export data."""
        if data.onnx_export:
            self._step_results[export_step] = {
                "completed": True,
                "timestamp": format_timestamp_iso(data.get_step_timestamp(export_step)),
                "data": data.onnx_export,
            }
        return 1

    @step(ExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: ExportStep, data: ExportData) -> int:
        """Record node tagging data."""
        if data.node_tagging:
            self._step_results[export_step] = {
                "completed": True,
                "timestamp": format_timestamp_iso(data.get_step_timestamp(export_step)),
                "data": data.node_tagging,
            }
        return 1

    @step(ExportStep.TAG_INJECTION)
    def write_tag_injection(self, export_step: ExportStep, data: ExportData) -> int:
        """Record tag injection data."""
        if data.tag_injection:
            self._step_results[export_step] = {
                "completed": True,
                "timestamp": format_timestamp_iso(data.get_step_timestamp(export_step)),
                "data": data.tag_injection,
            }

        # Report generation moved to flush() to get final export_time

        return 1

    def _generate_report(self, data: ExportData) -> None:
        """Generate the complete markdown report."""
        # Only generate once
        if self._report_generated:
            return

        try:
            # Clear any existing content and reinitialize document
            self.doc = snakemd.new_doc()
            # Header
            self.doc.add_heading(self.TITLE, level=1)

            # Metadata lines
            # Get export start time from first step or current time
            export_start = data.get_step_timestamp(ExportStep.MODEL_PREP)
            if export_start:
                self.doc.add_paragraph(f"**Generated**: {format_timestamp_iso(export_start)}")
            else:
                self.doc.add_paragraph(f"**Generated**: {format_timestamp_iso(time.time())}")
            self.doc.add_paragraph(f"**Model**: {data.model_name}")
            self.doc.add_paragraph(f"**Output**: {data.output_path}")
            self.doc.add_paragraph("**Strategy**: HTP (Hierarchical Tracing and Projection)")
            # Calculate export time from timestamps
            export_time = self._calculate_export_time(
                self._export_data if self._export_data else data
            )
            self.doc.add_paragraph(f"**Export Time**: {export_time:.2f}s")

            # Export Process Steps
            self.doc.add_heading(self.SECTION_EXPORT_STEPS, level=2)

            # Step 1: Model Preparation
            self._write_model_prep_section(data)

            # Step 2: Input Generation
            self._write_input_gen_section(data)

            # Step 3: Hierarchy Building
            self._write_hierarchy_section(data)

            # Step 4: ONNX Export
            self._write_onnx_export_section(data)

            # Step 5: Node Tagging
            self._write_node_tagging_section(data)

            # Step 6: Tag Injection
            self._write_tag_injection_section(data)

            # Module Hierarchy
            self._write_module_hierarchy_section(data)

            # Complete Node Mappings
            self._write_node_mappings_section(data)

            # Export Summary
            self._write_summary_section(data)

            # Footer
            self.doc.add_horizontal_rule()
            self.doc.add_paragraph("*Generated by HTP Exporter v1.0*")

            # Mark report as generated
            self._report_generated = True

        except Exception as e:
            # If markdown generation fails, create a minimal report
            self.doc = snakemd.new_doc()
            self.doc.add_heading("HTP ONNX Export Report - Error", level=1)
            self.doc.add_paragraph(f"**Error generating report**: {e!s}")
            self.doc.add_paragraph(f"**Model**: {data.model_name}")
            export_time = self._calculate_export_time(data)
            self.doc.add_paragraph(f"**Export Time**: {export_time:.2f}s")

    def _write_model_prep_section(self, data: ExportData) -> None:
        """Write model preparation section."""
        self.doc.add_heading("✅ Step 1/6: Model Preparation", level=3)

        if data.model_prep:
            params_formatted = self._format_params(data.model_prep.total_parameters)
            items = [
                f"**Model Class**: {data.model_prep.model_class}",
                f"**Total Modules**: {data.model_prep.total_modules}",
                f"**Total Parameters**: {data.model_prep.total_parameters:,} ({params_formatted})",
                "**Status**: Model set to evaluation mode",
            ]
            self.doc.add_unordered_list(items)

    def _write_input_gen_section(self, data: ExportData) -> None:
        """Write input generation section."""
        self.doc.add_heading("✅ Step 2/6: Input Generation", level=3)

        if data.input_gen:
            items = [
                f"**Method**: {data.input_gen.method}",
                f"**Model Type**: {data.input_gen.model_type}",
                f"**Detected Task**: {data.input_gen.task}",
                "**Generated Inputs**:",
            ]
            self.doc.add_unordered_list(items)

            # Input table
            headers = ["Input Name", "Shape", "Data Type"]
            rows = []
            for name, tensor_info in data.input_gen.inputs.items():
                rows.append([name, str(tensor_info.shape), tensor_info.dtype])

            self.doc.add_table(headers, rows, [snakemd.Table.Align.LEFT] * 3)

    def _write_hierarchy_section(self, data: ExportData) -> None:
        """Write hierarchy building section."""
        self.doc.add_heading("✅ Step 3/6: Hierarchy Building", level=3)

        if data.hierarchy:
            if data.hierarchy.source == HIERARCHY_SOURCE_ONNX_METADATA:
                items = [
                    f"**Modules Reconstructed**: {len(data.hierarchy.hierarchy)}",
                    "**Source**: ONNX node metadata (dynamo, no execution trace)",
                    "**Status**: Module hierarchy reconstructed from ONNX metadata",
                ]
            else:
                items = [
                    f"**Modules Traced**: {len(data.hierarchy.hierarchy)}",
                    f"**Execution Steps**: {data.hierarchy.execution_steps}",
                    "**Status**: Module hierarchy successfully traced",
                ]
            self.doc.add_unordered_list(items)

            # Add module hierarchy tree preview
            self.doc.add_heading("Module Hierarchy Preview", level=4)

            # Collapsible section with full hierarchy
            self.doc.add_raw("<details>")
            self.doc.add_raw("<summary>Click to expand module hierarchy</summary>")
            self.doc.add_raw("")

            # Generate complete ASCII tree using shared utility
            tree_lines = build_ascii_tree(data.hierarchy.hierarchy)
            tree_text = "\n".join(tree_lines)

            self.doc.add_raw("```")
            self.doc.add_raw(tree_text)
            self.doc.add_raw("```")
            self.doc.add_raw("")
            self.doc.add_raw("</details>")

    def _write_onnx_export_section(self, data: ExportData) -> None:
        """Write ONNX export section."""
        self.doc.add_heading("✅ Step 4/6: ONNX Export", level=3)

        if data.onnx_export:
            items = [
                "**Configuration**:",
            ]
            self.doc.add_unordered_list(items)

            # Add configuration sub-items
            config_items = [
                f"Opset Version: {data.onnx_export.opset_version}",
                f"Constant Folding: {data.onnx_export.do_constant_folding}",
                f"Output Names: {data.onnx_export.output_names}",
            ]
            self.doc.add_unordered_list(config_items)

            # Add remaining items
            remaining_items = [
                f"**Model Size**: {data.onnx_export.onnx_size_mb:.2f} MB",
                "**Status**: Successfully exported",
            ]
            self.doc.add_unordered_list(remaining_items)

    def _write_node_tagging_section(self, data: ExportData) -> None:
        """Write node tagging section."""
        self.doc.add_heading("✅ Step 5/6: Node Tagging", level=3)

        if data.node_tagging:
            coverage = data.node_tagging.coverage
            stats = data.node_tagging.tagging_stats

            tagged_count = len(data.node_tagging.tagged_nodes)
            items = [
                f"**Total ONNX Nodes**: {data.node_tagging.total_nodes}",
                f"**Tagged Nodes**: {tagged_count} ({coverage:.1f}% coverage)",
                "**Tagging Statistics**:",
            ]
            self.doc.add_unordered_list(items)

            # Statistics table
            headers = ["Match Type", "Count", "Percentage"]
            rows = []

            for stat_type in [
                "direct_matches",
                "parent_matches",
                "root_fallbacks",
                "empty_tags",
            ]:
                if stat_type in stats:
                    count = stats[stat_type]
                    pct = (
                        (count / data.node_tagging.total_nodes * 100)
                        if data.node_tagging.total_nodes > 0
                        else 0
                    )
                    rows.append([stat_type.replace("_", " ").title(), str(count), f"{pct:.1f}%"])

            self.doc.add_table(headers, rows, [snakemd.Table.Align.LEFT] * 3)

            # Add Top 20 Nodes by Hierarchy (similar to console output)
            if data.node_tagging and data.node_tagging.tagged_nodes:
                from collections import Counter

                # Count nodes by hierarchy tag
                tag_counts = Counter(data.node_tagging.tagged_nodes.values())
                sorted_tags = tag_counts.most_common(20)

                self.doc.add_heading("Top 20 Nodes by Hierarchy", level=4)
                self.doc.add_raw("")  # Empty line

                # Create numbered list as code block to match console format
                self.doc.add_raw("```")
                for i, (tag, count) in enumerate(sorted_tags, 1):
                    self.doc.add_raw(f" {i:2d}. {tag}: {count} nodes")
                self.doc.add_raw("```")
                self.doc.add_raw("")  # Empty line after the list

            # Add complete hierarchy with node counts
            if data.hierarchy:
                self.doc.add_heading("Complete HF Hierarchy with ONNX Nodes", level=4)

                # Count nodes per hierarchy path
                node_counts = count_nodes_per_tag(data.node_tagging.tagged_nodes)

                # Generate tree with counts and ONNX nodes as leaves using shared utility
                tree_lines = build_ascii_tree(
                    data.hierarchy.hierarchy,
                    show_counts=True,
                    node_counts=node_counts,
                    tagged_nodes=data.node_tagging.tagged_nodes,
                    show_nodes_as_leaves=True,
                )

                self.doc.add_raw("<details>")
                self.doc.add_raw(
                    "<summary>Click to expand complete hierarchy with node counts</summary>"
                )
                self.doc.add_raw("")

                # Add tree as a single code block
                tree_content = "\n".join(tree_lines)
                self.doc.add_raw("```")
                self.doc.add_raw(tree_content)
                self.doc.add_raw("```")
                self.doc.add_raw("")
                self.doc.add_raw("</details>")

    def _write_tag_injection_section(self, data: ExportData) -> None:
        """Write tag injection section."""
        self.doc.add_heading("✅ Step 6/6: Tag Injection", level=3)

        if data.tag_injection:
            embed_status = (
                "Embedded in ONNX" if data.tag_injection.tags_injected else "Stripped (clean ONNX)"
            )
            items = [
                f"**Hierarchy Tags**: {embed_status}",
                f"**Output File**: {Path(self.output_path).name}",
                "**Status**: Export completed successfully",
            ]
            self.doc.add_unordered_list(items)

    def _write_module_hierarchy_section(self, data: ExportData) -> None:
        """Write module hierarchy section."""
        self.doc.add_heading("Module Hierarchy", level=2)

        # Add note about diagram generation - keep for now but can be made conditional
        self.doc.add_paragraph("*Diagram generation disabled for stability.*")

        if not data.hierarchy:
            self.doc.add_paragraph("No hierarchy data available.")
            return

        # Get hierarchy data for module table
        hierarchy = data.hierarchy.hierarchy

        # Module table with reordered columns
        order_label = (
            "Discovery Order"
            if data.hierarchy.source == HIERARCHY_SOURCE_ONNX_METADATA
            else "Execution Order"
        )
        self.doc.add_heading(f"Module List (Sorted by {order_label})", level=3)

        # Count direct and total nodes for each module if available
        direct_counts: dict[str, int] = {}
        total_counts: dict[str, int] = {}
        if data.node_tagging and data.node_tagging.tagged_nodes:
            direct_counts, total_counts = count_direct_and_total_nodes(
                data.node_tagging.tagged_nodes
            )

        headers = [order_label, "Class Name", "Nodes", "Tag", "Scope"]
        rows = []

        # Sort by the source-specific encounter order
        sorted_items = sorted(
            hierarchy.items(),
            key=lambda x: (
                x[1].execution_order if x[1].execution_order is not None else 999999,
                x[0],
            ),
        )

        for path, module_info in sorted_items:
            display_path = "[ROOT]" if path == "" else path
            exec_order = (
                str(module_info.execution_order) if module_info.execution_order is not None else "-"
            )

            # Get node counts for this module
            tag = module_info.traced_tag if hasattr(module_info, "traced_tag") else ""
            direct_count = direct_counts.get(tag, 0) if tag else 0
            total_count = total_counts.get(tag, 0) if tag else 0
            nodes_str = f"{direct_count}/{total_count}"

            rows.append(
                [
                    exec_order,
                    self._escape_markdown(module_info.class_name),
                    nodes_str,
                    self._escape_markdown(module_info.traced_tag),
                    self._escape_markdown(display_path),
                ]
            )

        # Create markdown table
        self.doc.add_table(headers, rows, [snakemd.Table.Align.LEFT] * 5)

    def _write_node_mappings_section(self, data: ExportData) -> None:
        """Write complete node mappings section."""
        self.doc.add_heading("Complete Node Mappings", level=2)

        if not data.node_tagging:
            self.doc.add_paragraph("No node tagging data available.")
            return

        node_count = len(data.node_tagging.tagged_nodes)

        # Collapsible section
        self.doc.add_raw("<details>")
        self.doc.add_raw(f"<summary>Click to expand all {node_count} node mappings</summary>")
        self.doc.add_raw("")
        self.doc.add_raw("```")

        # Sort and write all mappings
        for node_name, hierarchy_tag in sorted(data.node_tagging.tagged_nodes.items()):
            self.doc.add_raw(f"{node_name} -> {hierarchy_tag}")

        self.doc.add_raw("```")
        self.doc.add_raw("")
        self.doc.add_raw("</details>")

    def _calculate_export_time(self, data: ExportData) -> float:
        """Get export time from ExportData."""
        # Use the stored reference if available (which has the final export_time)
        if self._export_data and self._export_data.export_time > 0:
            return self._export_data.export_time
        # Otherwise use the passed data
        return data.export_time if data.export_time > 0 else 0.0

    def _write_summary_section(self, data: ExportData) -> None:
        """Write export summary section."""
        self.doc.add_heading("Export Summary", level=2)

        # Performance Metrics
        self.doc.add_heading("Performance Metrics", level=3)

        # Calculate export time from timestamps
        total_time = self._calculate_export_time(data)
        metrics = [
            f"**Export Time**: {total_time:.2f}s",
            f"**Module Processing**: ~{total_time * 0.2:.2f}s",
            f"**ONNX Conversion**: ~{total_time * 0.5:.2f}s",
            f"**Node Tagging**: ~{total_time * 0.3:.2f}s",
        ]
        self.doc.add_unordered_list(metrics)

        # Coverage Statistics
        self.doc.add_heading("Coverage Statistics", level=3)

        traced_modules = len(data.hierarchy.hierarchy) if data.hierarchy else 0
        total_modules = data.model_prep.total_modules if data.model_prep else 0
        onnx_nodes = data.node_tagging.total_nodes if data.node_tagging else 0
        tagged_nodes = len(data.node_tagging.tagged_nodes) if data.node_tagging else 0
        coverage = data.node_tagging.coverage if data.node_tagging else 0.0
        empty_tags = (
            data.node_tagging.tagging_stats.get("empty_tags", 0) if data.node_tagging else 0
        )

        modules_label = (
            "Recovered Modules"
            if data.hierarchy and data.hierarchy.source == HIERARCHY_SOURCE_ONNX_METADATA
            else "Traced Modules"
        )
        stats = [
            f"**Hierarchy Modules**: {total_modules}",
            f"**{modules_label}**: {traced_modules}/{total_modules}",
            f"**ONNX Nodes**: {onnx_nodes}",
            f"**Tagged Nodes**: {tagged_nodes} ({coverage:.1f}%)",
            f"**Empty Tags**: {empty_tags}",
        ]
        self.doc.add_unordered_list(stats)

        # Output Files
        self.doc.add_heading("Output Files", level=3)

        onnx_size = data.onnx_export.onnx_size_mb if data.onnx_export else 0.0
        files = [
            f"**ONNX Model**: `{self.output_path}` ({onnx_size:.2f} MB)",
            f"**Metadata**: `{self.output_path}_htp_metadata.json`",
            f"**Report**: `{self.report_path}`",
        ]
        self.doc.add_unordered_list(files)

    def _format_params(self, param_count: int) -> str:
        """Format parameter count in human readable format."""
        if param_count >= 1e9:
            return f"{param_count / 1e9:.1f}B"
        if param_count >= 1e6:
            return f"{param_count / 1e6:.1f}M"
        if param_count >= 1e3:
            return f"{param_count / 1e3:.1f}K"
        return str(param_count)

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for markdown tables."""
        # Replace pipe characters and newlines
        return text.replace("|", "\\|").replace("\n", " ").replace("\r", "")

    # Using shared hierarchy_utils for all tree building

    def flush(self) -> None:
        """Generate report and write markdown to file."""
        try:
            # Generate the report with the latest data (including final export_time)
            if self._export_data:
                self._generate_report(self._export_data)

            # Ensure output directory exists
            Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)

            with Path(self.report_path).open("w", encoding="utf-8") as f:
                f.write(str(self.doc))
        except OSError as e:
            print(f"Error writing report to {self.report_path}: {e}")
            # Re-raise to maintain error propagation
            raise

    def close(self) -> None:
        """Close and write file."""
        self.flush()
        super().close()
