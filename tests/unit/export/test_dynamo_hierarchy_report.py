# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Writer-level tests for the dynamo module-hierarchy report.

These exercise the actual ``MetadataWriter`` tree output (not just the flat map
``build_module_hierarchy`` returns), covering the two failure modes the shared
writers previously had:

  * same-class named siblings (an attention block's query/key/value Linears)
    collapsing to a single tree entry because children were keyed by class name,
    and
  * a sparse hierarchy whose intermediate container scope (a ``ModuleList``)
    never emitted its own entry, orphaning the whole subtree from the tree.
"""

from __future__ import annotations

import io
import logging

from onnx import ModelProto, NodeProto, StringStringEntryProto, helper
from rich.console import Console

from winml.modelkit.core.hierarchy_utils import find_immediate_children
from winml.modelkit.core.onnx_node_tagger import DynamoMetadataTagger
from winml.modelkit.export.htp import HTPExporter, HTPExportMonitor
from winml.modelkit.export.htp.base_writer import ExportData, ExportStep
from winml.modelkit.export.htp.console_writer import ConsoleWriter
from winml.modelkit.export.htp.markdown_report_writer import MarkdownReportWriter
from winml.modelkit.export.htp.metadata_writer import MetadataWriter
from winml.modelkit.export.htp.step_data import (
    HIERARCHY_SOURCE_ONNX_METADATA,
    HIERARCHY_SOURCE_TRACE,
    HierarchyData,
    ModelPrepData,
    ModuleInfo,
)


NAME_SCOPES_KEY = "pkg.torch.onnx.name_scopes"
CLASS_HIERARCHY_KEY = "pkg.torch.onnx.class_hierarchy"


def _node(op_type: str, name: str, name_scopes: list[str], class_hierarchy: list[str]) -> NodeProto:
    node = helper.make_node(op_type, inputs=["x"], outputs=["y"], name=name)
    node.metadata_props.append(StringStringEntryProto(key=NAME_SCOPES_KEY, value=repr(name_scopes)))
    node.metadata_props.append(
        StringStringEntryProto(key=CLASS_HIERARCHY_KEY, value=repr(class_hierarchy))
    )
    return node


def _model(nodes: list[NodeProto]) -> ModelProto:
    return helper.make_model(helper.make_graph(nodes, "g", inputs=[], outputs=[]))


def _attention_model() -> ModelProto:
    """One block with an attention module exposing query/key/value Linears.

    The ``blocks`` ModuleList never appears as its own scope (torch skips
    container modules), so the root's only child is the compound ``blocks.0``.
    """
    return _model(
        [
            _node(
                "MatMul",
                f"n_{name}",
                ["", "blocks.0", "blocks.0.attn", f"blocks.0.attn.{name}", "linear"],
                [
                    "pkg.Net",
                    "pkg.Blk",
                    "pkg.Attention",
                    "torch.nn.modules.linear.Linear",
                    "aten.linear.default",
                ],
            )
            for name in ("query", "key", "value")
        ]
    )


def _to_module_info(flat: dict[str, dict]) -> dict[str, ModuleInfo]:
    return {
        scope: ModuleInfo(
            class_name=info["class_name"],
            traced_tag=info["traced_tag"],
            execution_order=info["execution_order"],
        )
        for scope, info in flat.items()
    }


class TestFindImmediateChildrenSparse:
    """Nearest-present-ancestor nesting for sparse/compound scopes."""

    def test_compound_root_child_attaches_to_root(self) -> None:
        # "blocks.0" has no "blocks" ancestor entry, so it is a root child and
        # its subtree is not dropped.
        hierarchy = {"": {}, "blocks.0": {}, "blocks.0.attn": {}}
        assert find_immediate_children("", hierarchy) == ["blocks.0"]
        assert find_immediate_children("blocks.0", hierarchy) == ["blocks.0.attn"]

    def test_present_container_reparents_index(self) -> None:
        # When the "blocks" container IS present, "blocks.0" nests under it.
        hierarchy = {"": {}, "blocks": {}, "blocks.0": {}}
        assert find_immediate_children("", hierarchy) == ["blocks"]
        assert find_immediate_children("blocks", hierarchy) == ["blocks.0"]


class TestMetadataWriterTree:
    """The persisted MetadataWriter tree preserves every reconstructed module."""

    def _tree(self) -> dict:
        flat = DynamoMetadataTagger().build_module_hierarchy(_attention_model())
        writer = MetadataWriter("unused.json")
        return writer._build_hierarchical_modules(_to_module_info(flat))

    def test_same_class_siblings_all_serialized(self) -> None:
        tree = self._tree()
        # root -> Blk.0 -> Attention -> {Linear.query, Linear.key, Linear.value}
        attn = tree["children"]["Blk.0"]["children"]["Attention"]
        linears = attn["children"]
        assert set(linears) == {"Linear.query", "Linear.key", "Linear.value"}
        scopes = {child["scope"] for child in linears.values()}
        assert scopes == {
            "blocks.0.attn.query",
            "blocks.0.attn.key",
            "blocks.0.attn.value",
        }

    def test_sparse_root_subtree_present(self) -> None:
        tree = self._tree()
        # The compound root child "blocks.0" is present, not orphaned.
        assert "Blk.0" in tree["children"]
        assert tree["children"]["Blk.0"]["scope"] == "blocks.0"


def _hierarchy_data(source: str, execution_steps: int | None) -> HierarchyData:
    return HierarchyData(
        hierarchy={"": ModuleInfo(class_name="Net", traced_tag="/Net")},
        execution_steps=execution_steps,
        source=source,
    )


def _console_output(hierarchy: HierarchyData) -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=120, no_color=True)
    writer = ConsoleWriter(console=console)
    writer.write(ExportStep.HIERARCHY, ExportData(hierarchy=hierarchy))
    return buf.getvalue()


class TestHierarchySourceWording:
    """Console/metadata wording must match how the hierarchy was obtained."""

    def test_dynamo_source_uses_reconstruction_wording(self) -> None:
        out = _console_output(_hierarchy_data(HIERARCHY_SOURCE_ONNX_METADATA, None))
        assert "Reconstructing module hierarchy from ONNX" in out
        # No forward trace ran, so no trace/execution-step claims.
        assert "Tracing module execution" not in out
        assert "execution steps" not in out.lower()

    def test_trace_source_uses_trace_wording(self) -> None:
        out = _console_output(_hierarchy_data(HIERARCHY_SOURCE_TRACE, 42))
        assert "Tracing module execution" in out
        assert "Total execution steps" in out
        assert "42" in out

    def test_metadata_records_dynamo_source_without_fake_step_count(self) -> None:
        writer = MetadataWriter("unused.json")
        writer.write(
            ExportStep.HIERARCHY,
            ExportData(hierarchy=_hierarchy_data(HIERARCHY_SOURCE_ONNX_METADATA, None)),
        )
        info = writer.builder._tracing_info
        assert info.source == HIERARCHY_SOURCE_ONNX_METADATA
        assert info.builder == "DynamoMetadataTagger"
        # Module count must not be reported as an execution-step total.
        assert info.execution_steps == 0

    def test_metadata_records_trace_source_and_steps(self) -> None:
        writer = MetadataWriter("unused.json")
        writer.write(
            ExportStep.HIERARCHY,
            ExportData(hierarchy=_hierarchy_data(HIERARCHY_SOURCE_TRACE, 7)),
        )
        info = writer.builder._tracing_info
        assert info.source == HIERARCHY_SOURCE_TRACE
        assert info.builder == "TracingHierarchyBuilder"
        assert info.execution_steps == 7


class TestDynamoHierarchyRecoveryWarning:
    """A completely missing dynamo hierarchy is visible but non-fatal."""

    def test_warns_when_nonempty_graph_has_no_usable_metadata(self, caplog) -> None:
        exporter = HTPExporter()
        exporter._node_tagger = DynamoMetadataTagger()
        model = _model([helper.make_node("Identity", ["x"], ["y"], name="bare")])

        with caplog.at_level(logging.WARNING):
            hierarchy = exporter._recover_dynamo_hierarchy(model)

        assert hierarchy == {}
        assert "no usable module hierarchy metadata" in caplog.text
        assert "--no-dynamo" in caplog.text

    def test_does_not_warn_when_hierarchy_is_recovered(self, caplog) -> None:
        exporter = HTPExporter()
        exporter._node_tagger = DynamoMetadataTagger()

        with caplog.at_level(logging.WARNING):
            hierarchy = exporter._recover_dynamo_hierarchy(_attention_model())

        assert hierarchy
        assert "no usable module hierarchy metadata" not in caplog.text


def _summary_data(source: str) -> ExportData:
    return ExportData(
        hierarchy=HierarchyData(
            hierarchy={"": ModuleInfo(class_name="Net", traced_tag="/Net")},
            source=source,
        ),
        model_prep=ModelPrepData(model_class="Net", total_modules=3, total_parameters=0),
    )


class TestExportSummaryModulesLabel:
    """The export summary must not claim a trace when modules were recovered."""

    def _monitor_summary(self, source: str) -> str:
        monitor = HTPExportMonitor("unused.onnx", verbose=True)
        buf = io.StringIO()
        monitor.console = Console(file=buf, width=120, no_color=True)
        monitor.data = _summary_data(source)
        monitor._print_summary()
        return buf.getvalue()

    def test_console_summary_dynamo_says_recovered(self) -> None:
        out = self._monitor_summary(HIERARCHY_SOURCE_ONNX_METADATA)
        assert "Recovered modules:" in out
        assert "Traced modules:" not in out

    def test_console_summary_trace_says_traced(self) -> None:
        out = self._monitor_summary(HIERARCHY_SOURCE_TRACE)
        assert "Traced modules:" in out
        assert "Recovered modules:" not in out

    def _markdown_summary(self, source: str) -> str:
        writer = MarkdownReportWriter("unused.md")
        writer._write_summary_section(_summary_data(source))
        return str(writer.doc)

    def test_markdown_summary_dynamo_says_recovered(self) -> None:
        out = self._markdown_summary(HIERARCHY_SOURCE_ONNX_METADATA)
        assert "Recovered Modules" in out
        assert "Traced Modules" not in out

    def test_markdown_summary_trace_says_traced(self) -> None:
        out = self._markdown_summary(HIERARCHY_SOURCE_TRACE)
        assert "Traced Modules" in out
        assert "Recovered Modules" not in out
