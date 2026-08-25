# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for produced-operator EP/device support checking.

Follows the Cardinal Rules:
- #1 No hardcoded architectures — models are built generically in-test and the
  produced operators are derived from the real dry-run diff.
- #2 Code-generated results — expectations derive from the constructed graphs
  and crafted runtime results, never from an LLM.
- #4 No skips — deterministic assertions only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

from tests.unit.test_helpers import stable_test_node_keys
from winml.modelkit.analyze.core import runtime_checker as runtime_checker_module
from winml.modelkit.analyze.models.runtime_checks import PatternRuntime, RuntimeTestResult
from winml.modelkit.analyze.models.support_level import SupportLevel
from winml.modelkit.analyze.optim_output import (
    OptimizationOutputSupport,
    ProducedOperatorSupport,
    check_optimization_output_support,
)
from winml.modelkit.optim import iter_optimization_outputs
from winml.modelkit.optim.pipes import get_all_capabilities
from winml.modelkit.pattern import (
    OperatorPattern,
    PatternMatchResult,
    PatternType,
    SkeletonMatchResult,
)


if TYPE_CHECKING:
    import pytest


# =============================================================================
# MODEL BUILDERS
# =============================================================================


def _matmul_add_model() -> ModelProto:
    """MatMul followed by Add — a canonical MatMul+Add(->Gemm) fusion candidate."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 16])
    w = numpy_helper.from_array(np.random.randn(8, 16).astype(np.float32), "W")
    b = numpy_helper.from_array(np.random.randn(16).astype(np.float32), "B")
    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["mm"], name="mm"),
        helper.make_node("Add", ["mm", "B"], ["y"], name="addbias"),
    ]
    graph = helper.make_graph(nodes, "matmul_add", [x], [y], initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    return model


def _runtime_for_output(output_name: str, *, compile_ok: bool, run_ok: bool) -> PatternRuntime:
    """Craft a PatternRuntime whose matched node emits ``output_name``."""
    pattern = OperatorPattern(
        pattern_id="OP/ai.onnx/Gemm",
        pattern_type=PatternType.OPERATOR,
        namespace="ai.onnx",
        op_type="Gemm",
        description="",
    )
    node = helper.make_node("Gemm", ["in"], [output_name], name="gemm_node")
    skeleton = SkeletonMatchResult(
        pattern=pattern,
        matched_nodes=[node],
        matched_node_keys=stable_test_node_keys([node]),
        matcher=None,
    )
    match = PatternMatchResult(
        skeleton_match_result=skeleton,
        schema_input_to_value={},
        schema_output_to_value={},
        type_param_to_type={},
    )
    return PatternRuntime(
        pattern_id=pattern.pattern_id,
        result=RuntimeTestResult(compile=compile_ok, run=run_ok),
        pattern_match=match,
    )


def _patch_op_support(monkeypatch: pytest.MonkeyPatch, *, compile_ok: bool, run_ok: bool) -> None:
    """Patch RuntimeChecker.op_support to classify every produced node deterministically."""

    def fake_op_support(self, *, node_output_filter=None, on_node_result=None, **_kw):
        outputs = sorted(node_output_filter or [])
        return [_runtime_for_output(out, compile_ok=compile_ok, run_ok=run_ok) for out in outputs]

    monkeypatch.setattr(runtime_checker_module.RuntimeChecker, "op_support", fake_op_support)


# =============================================================================
# DATACLASS BEHAVIOUR
# =============================================================================


class TestOptimizationOutputSupport:
    def test_worst_support_empty_is_supported(self) -> None:
        opt = OptimizationOutputSupport(
            name="x",
            enable_flag="--enable-x",
            category="misc",
            description="",
            pipe_name="p",
        )
        assert opt.worst_support is SupportLevel.SUPPORTED

    def test_worst_support_picks_most_concerning(self) -> None:
        opt = OptimizationOutputSupport(
            name="x",
            enable_flag="--enable-x",
            category="misc",
            description="",
            pipe_name="p",
            operators=[
                ProducedOperatorSupport("A", "A 'a'", "added", SupportLevel.SUPPORTED),
                ProducedOperatorSupport("B", "B 'b'", "added", SupportLevel.UNSUPPORTED),
                ProducedOperatorSupport("C", "C 'c'", "modified", SupportLevel.PARTIAL),
            ],
        )
        assert opt.worst_support is SupportLevel.UNSUPPORTED

    def test_worst_support_unknown_below_partial(self) -> None:
        opt = OptimizationOutputSupport(
            name="x",
            enable_flag="--enable-x",
            category="misc",
            description="",
            pipe_name="p",
            operators=[
                ProducedOperatorSupport("A", "A 'a'", "added", SupportLevel.SUPPORTED),
                ProducedOperatorSupport("B", "B 'b'", "added", SupportLevel.UNKNOWN),
            ],
        )
        assert opt.worst_support is SupportLevel.UNKNOWN

    def test_support_counts(self) -> None:
        opt = OptimizationOutputSupport(
            name="x",
            enable_flag="--enable-x",
            category="misc",
            description="",
            pipe_name="p",
            operators=[
                ProducedOperatorSupport("A", "A 'a'", "added", SupportLevel.SUPPORTED),
                ProducedOperatorSupport("B", "B 'b'", "added", SupportLevel.SUPPORTED),
                ProducedOperatorSupport("C", "C 'c'", "modified", SupportLevel.PARTIAL),
            ],
        )
        counts = opt.support_counts()
        assert counts[SupportLevel.SUPPORTED] == 2
        assert counts[SupportLevel.PARTIAL] == 1

    def test_to_dict_includes_graph_delta_and_target_support(self) -> None:
        """Structured JSON retains actionable graph and support evidence."""
        from winml.modelkit.optim import NodeRef

        opt = OptimizationOutputSupport(
            name="static-split-to-slice",
            enable_flag="--enable-static-split-to-slice",
            category="rewrite",
            description="Replace static Split with Slice.",
            pipe_name="algebraic",
            removed_nodes=[NodeRef("Split", "split", ("a", "b"), "ai.onnx")],
            added_nodes=[NodeRef("Slice", "slice_0", ("a",), "com.microsoft")],
            modified_initializers=["starts"],
            operators=[
                ProducedOperatorSupport(
                    "com.microsoft::Slice",
                    "com.microsoft::Slice 'slice_0'",
                    "added",
                    SupportLevel.SUPPORTED,
                )
            ],
        )

        assert opt.to_dict() == {
            "name": "static-split-to-slice",
            "enable_flag": "--enable-static-split-to-slice",
            "category": "rewrite",
            "description": "Replace static Split with Slice.",
            "pipe_name": "algebraic",
            "worst_support": "supported",
            "support_counts": {"supported": 1},
            "graph_delta": {
                "removed_nodes": [{"op_type": "Split", "name": "split", "outputs": ["a", "b"]}],
                "added_nodes": [
                    {
                        "op_type": "Slice",
                        "name": "slice_0",
                        "outputs": ["a"],
                        "domain": "com.microsoft",
                    }
                ],
                "modified_nodes": [],
                "removed_initializers": [],
                "added_initializers": [],
                "modified_initializers": ["starts"],
            },
            "operators": [
                {
                    "op_type": "com.microsoft::Slice",
                    "label": "com.microsoft::Slice 'slice_0'",
                    "change": "added",
                    "support": "supported",
                }
            ],
        }


# =============================================================================
# END-TO-END SUPPORT CHECK
# =============================================================================


class TestCheckOptimizationOutputSupport:
    def test_produced_operator_marked_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_op_support(monkeypatch, compile_ok=True, run_ok=True)
        outputs = list(iter_optimization_outputs(_matmul_add_model(), get_all_capabilities()))

        results = check_optimization_output_support(
            outputs, ep="QNNExecutionProvider", device="NPU", model_path="model.onnx"
        )

        by_name = {r.name: r for r in results}
        assert "matmul-add-fusion" in by_name, sorted(by_name)
        fusion = by_name["matmul-add-fusion"]
        assert fusion.error is None
        assert fusion.operators, "expected at least one produced operator"
        assert all(op.support is SupportLevel.SUPPORTED for op in fusion.operators)
        assert fusion.worst_support is SupportLevel.SUPPORTED

    def test_produced_operator_marked_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_op_support(monkeypatch, compile_ok=False, run_ok=False)
        outputs = list(iter_optimization_outputs(_matmul_add_model(), get_all_capabilities()))

        results = check_optimization_output_support(
            outputs, ep="QNNExecutionProvider", device="NPU", model_path="model.onnx"
        )

        fusion = next(r for r in results if r.name == "matmul-add-fusion")
        assert fusion.worst_support is SupportLevel.UNSUPPORTED
        assert all(op.support is SupportLevel.UNSUPPORTED for op in fusion.operators)

    def test_partial_classification(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_op_support(monkeypatch, compile_ok=False, run_ok=True)
        outputs = list(iter_optimization_outputs(_matmul_add_model(), get_all_capabilities()))

        results = check_optimization_output_support(
            outputs, ep="QNNExecutionProvider", device="NPU", model_path="model.onnx"
        )

        fusion = next(r for r in results if r.name == "matmul-add-fusion")
        assert fusion.worst_support is SupportLevel.PARTIAL

    def test_check_failure_is_captured_as_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(self, **_kw):
            raise RuntimeError("checker exploded")

        monkeypatch.setattr(runtime_checker_module.RuntimeChecker, "op_support", boom)
        outputs = list(iter_optimization_outputs(_matmul_add_model(), get_all_capabilities()))

        results = check_optimization_output_support(
            outputs, ep="QNNExecutionProvider", device="NPU", model_path="model.onnx"
        )

        fusion = next(r for r in results if r.name == "matmul-add-fusion")
        assert fusion.error is not None
        assert "checker exploded" in fusion.error
        # With no operator data the optimization degrades to "supported" (nothing
        # concerning was found), keeping the report non-alarming.
        assert fusion.worst_support is SupportLevel.SUPPORTED

    def test_preserves_input_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_op_support(monkeypatch, compile_ok=True, run_ok=True)
        outputs = list(iter_optimization_outputs(_matmul_add_model(), get_all_capabilities()))

        results = check_optimization_output_support(
            outputs, ep="QNNExecutionProvider", device="NPU", model_path="model.onnx"
        )

        assert [r.name for r in results] == [finding.name for finding, _ in outputs]
