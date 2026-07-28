# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the optimizer applicability analysis (dry-run) module.

Follows the Cardinal Rules:
- #1 No hardcoded architectures — models are built generically in-test.
- #2 Code-generated results — expectations derive from the constructed graphs.
- #4 No skips — deterministic assertions only (clamp/surgery, graph diffing).
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from winml.modelkit.optim import CapabilityFinding, NodeRef, analyze_model
from winml.modelkit.optim.analysis import (
    _collect_initializers,
    _collect_nodes,
    _diff_initializers,
    _diff_nodes,
)
from winml.modelkit.optim.pipes import get_all_capabilities


# =============================================================================
# MODEL BUILDERS (code-generated, no hardcoded architecture assumptions)
# =============================================================================


def _finalize(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _matmul_add_model() -> onnx.ModelProto:
    """MatMul followed by Add — a canonical MatMul+Add(->Gemm) fusion candidate."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 16])
    w = numpy_helper.from_array(np.random.randn(8, 16).astype(np.float32), "W")
    b = numpy_helper.from_array(np.random.randn(16).astype(np.float32), "B")
    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["mm"], name="mm"),
        helper.make_node("Add", ["mm", "B"], ["y"], name="addbias"),
    ]
    return _finalize(helper.make_graph(nodes, "matmul_add", [x], [y], initializer=[w, b]))


def _extreme_constant_model() -> onnx.ModelProto:
    """Add of a runtime input and an initializer holding an extreme value."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [4])
    big = numpy_helper.from_array(np.full(4, 1e30, dtype=np.float32), "BIG")
    node = helper.make_node("Add", ["x", "BIG"], ["z"], name="addbig")
    return _finalize(helper.make_graph([node], "extreme_const", [x], [z], initializer=[big]))


def _benign_model() -> onnx.ModelProto:
    """Add of a runtime input and an initializer with ordinary values."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [4])
    small = numpy_helper.from_array(np.ones(4, dtype=np.float32), "SMALL")
    node = helper.make_node("Add", ["x", "SMALL"], ["z"], name="addsmall")
    return _finalize(helper.make_graph([node], "benign", [x], [z], initializer=[small]))


# =============================================================================
# NODE / INITIALIZER DIFF HELPERS
# =============================================================================


class TestNodeDiff:
    """The node diff distinguishes removed, added and modified nodes."""

    def test_removed_added_modified(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
        w = numpy_helper.from_array(np.zeros((8, 8), np.float32), "W")

        base_graph = helper.make_graph(
            [
                helper.make_node("Relu", ["x"], ["r"], name="relu"),
                helper.make_node("Add", ["r", "W"], ["y"], name="add"),
            ],
            "base",
            [x],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])],
            initializer=[w],
        )
        probe_graph = helper.make_graph(
            [
                # 'r'-producing node gone -> removed; 'y' now a Gemm -> modified.
                helper.make_node("Gemm", ["x", "W"], ["y"], name="gemm"),
                # brand-new output 'z' -> added.
                helper.make_node("Identity", ["y"], ["z"], name="cast"),
            ],
            "probe",
            [x],
            [helper.make_tensor_value_info("z", TensorProto.FLOAT, [4, 8])],
            initializer=[w],
        )

        base_table: dict = {}
        probe_table: dict = {}
        _collect_nodes(base_graph, (), base_table)
        _collect_nodes(probe_graph, (), probe_table)
        removed, added, modified = _diff_nodes(base_table, probe_table)

        assert [n.op_type for n in removed] == ["Relu"]
        assert [n.op_type for n in added] == ["Identity"]
        assert [n.op_type for n in modified] == ["Gemm"]

    def test_identical_graphs_have_no_diff(self) -> None:
        model = _matmul_add_model()
        table_a: dict = {}
        table_b: dict = {}
        _collect_nodes(model.graph, (), table_a)
        _collect_nodes(model.graph, (), table_b)
        removed, added, modified = _diff_nodes(table_a, table_b)
        assert not removed and not added and not modified

    def test_subgraph_nodes_are_collected(self) -> None:
        """Nodes inside a control-flow subgraph are included in the table."""
        then_graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["t"], name="then_id")],
            "then",
            [],
            [helper.make_tensor_value_info("t", TensorProto.FLOAT, [1])],
        )
        else_graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["e"], name="else_id")],
            "else",
            [],
            [helper.make_tensor_value_info("e", TensorProto.FLOAT, [1])],
        )
        outer = helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["out"],
                    name="if",
                    then_branch=then_graph,
                    else_branch=else_graph,
                )
            ],
            "outer",
            [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
        )
        table: dict = {}
        _collect_nodes(outer, (), table)
        op_types = sorted(ref.op_type for _, ref in table.values())
        assert op_types == ["Identity", "Identity", "If"]


class TestInitializerDiff:
    """The initializer diff reports removed, added and modified constants."""

    def test_modified_initializer_detected(self) -> None:
        base = _extreme_constant_model()
        probe = _extreme_constant_model()
        # Clamp the probe's initializer in place.
        clamped = numpy_helper.from_array(np.full(4, 1e3, dtype=np.float32), "BIG")
        probe.graph.initializer[0].CopyFrom(clamped)

        removed, added, modified = _diff_initializers(
            _collect_initializers(base), _collect_initializers(probe)
        )
        assert removed == []
        assert added == []
        assert modified == ["BIG"]


# =============================================================================
# RESULT TYPES
# =============================================================================


class TestCapabilityFinding:
    def test_applicable_reflects_any_change(self) -> None:
        empty = CapabilityFinding(
            name="x",
            python_name="x",
            enable_flag="--enable-x",
            category="misc",
            description="",
            pipe_name="p",
        )
        assert empty.applicable is False

        node = NodeRef("MatMul", "mm", ("mm",))
        with_change = CapabilityFinding(
            name="x",
            python_name="x",
            enable_flag="--enable-x",
            category="misc",
            description="",
            pipe_name="p",
            removed_nodes=[node],
        )
        assert with_change.applicable is True
        assert with_change.affected_node_count == 1
        assert with_change.op_histogram("removed") == [("MatMul", 1)]

    def test_node_ref_label_uses_name_then_output(self) -> None:
        assert NodeRef("MatMul", "mm", ("mm_out",)).label() == "MatMul 'mm'"
        assert NodeRef("Add", "", ("y",)).label() == "Add 'y'"


# =============================================================================
# END-TO-END ANALYSIS
# =============================================================================


class TestAnalyzeModel:
    """analyze_model probes the real pipeline and reports applicable caps."""

    def test_clamp_reported_for_extreme_constant(self) -> None:
        findings = analyze_model(_extreme_constant_model(), get_all_capabilities())
        by_name = {f.name: f for f in findings}
        assert "clamp-constant-values" in by_name
        clamp = by_name["clamp-constant-values"]
        assert clamp.applicable
        assert "BIG" in clamp.modified_initializers

    def test_clamp_not_reported_for_benign_constant(self) -> None:
        findings = analyze_model(_benign_model(), get_all_capabilities())
        names = {f.name for f in findings}
        assert "clamp-constant-values" not in names

    def test_matmul_add_fusion_detected(self) -> None:
        findings = analyze_model(_matmul_add_model(), get_all_capabilities())
        by_name = {f.name: f for f in findings}
        assert "matmul-add-fusion" in by_name, sorted(by_name)
        finding = by_name["matmul-add-fusion"]
        # The standalone MatMul is consumed by the fusion.
        assert any(n.op_type == "MatMul" for n in finding.removed_nodes)

    def test_findings_are_capability_findings(self) -> None:
        findings = analyze_model(_extreme_constant_model(), get_all_capabilities())
        assert all(isinstance(f, CapabilityFinding) for f in findings)
        assert all(f.applicable for f in findings)

    def test_input_model_not_mutated(self) -> None:
        model = _extreme_constant_model()
        before_nodes = len(model.graph.node)
        before_inits = len(model.graph.initializer)
        before_value = numpy_helper.to_array(model.graph.initializer[0]).copy()

        analyze_model(model, get_all_capabilities())

        assert len(model.graph.node) == before_nodes
        assert len(model.graph.initializer) == before_inits
        after_value = numpy_helper.to_array(model.graph.initializer[0])
        np.testing.assert_array_equal(before_value, after_value)
