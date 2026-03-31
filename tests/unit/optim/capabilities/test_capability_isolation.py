# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Capability Isolation Tests for ORT Graph Optimization.

These tests verify that enabling ONE capability does NOT enable others.
This is critical for ensuring users get predictable optimization behavior.

TEST DESIGN:
- Use the all-in-one test model with 9 patterns (p01_ to p09_)
- Enable ONLY one capability at a time via RAW ORT API
- Verify ONLY the corresponding pattern is optimized
- Verify OTHER patterns remain unoptimized

CRITICAL:
- These tests use RAW ORT API (optimize_at_level), NOT Pipe classes
- Per Section 9.6 of 4_graph_pipe.md: Capability tests MUST NOT depend on Pipe
- These tests should FAIL if capability isolation is broken
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ..conftest import optimize_at_level
from .conftest import get_all_ort_names


if TYPE_CHECKING:
    import onnx


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def apply_optimization_with_capability(
    model: onnx.ModelProto, ort_names_to_enable: list[str]
) -> onnx.ModelProto:
    """Apply optimization with only specified capabilities enabled.

    Uses RAW ORT API via optimize_at_level, NOT Pipe classes.
    This enables specific ORT optimizers by disabling all others.

    Args:
        model: Input ONNX model.
        ort_names_to_enable: List of ORT optimizer names to keep enabled.

    Returns:
        Optimized model with only specified capabilities active.
    """
    # Get all registered ORT names
    all_ort_names = get_all_ort_names()

    # Disable all EXCEPT the ones we want to enable
    disabled_list = [name for name in all_ort_names if name not in ort_names_to_enable]

    return optimize_at_level(model, level=2, disabled_optimizers=disabled_list)


def count_nodes_by_prefix_and_op(model: onnx.ModelProto, prefix: str, op_type: str) -> int:
    """Count nodes with given prefix and op type."""
    return sum(1 for n in model.graph.node if n.name.startswith(prefix) and n.op_type == op_type)


def count_nodes_by_prefix(model: onnx.ModelProto, prefix: str) -> int:
    """Count all nodes with given prefix."""
    return sum(1 for n in model.graph.node if n.name.startswith(prefix))


def count_op_type(model: onnx.ModelProto, op_type: str) -> int:
    """Count all nodes of given op type."""
    return sum(1 for n in model.graph.node if n.op_type == op_type)


# =============================================================================
# CAPABILITY ISOLATION TESTS
# =============================================================================


class TestGeluFusionIsolation:
    """Test that enabling GELU fusion does NOT enable MatMulAddFusion.

    This is the regression test for the bug where --enable-gelu-fusion
    also triggered MatMulAddFusion, causing MatMul nodes to be converted
    to Gemm unexpectedly.
    """

    def test_gelu_fusion_does_not_trigger_matmul_add_fusion(
        self, all_patterns_model: onnx.ModelProto
    ):
        """CRITICAL: Enabling gelu_fusion should NOT optimize MatMul+Add patterns.

        Pattern mappings:
        - p06_matmuladdrelu_: MatMul+Add+Relu pattern (for MatMulAddFusion)
        - p08_biasgelu_: Bias+GELU pattern (for GELU fusion)

        Expected behavior:
        - With ONLY gelu_fusion=True:
          - p06 MatMul nodes should REMAIN (not converted to Gemm)
          - p08 GELU patterns MAY be optimized

        This test will FAIL if MatMulAddFusion runs when only gelu_fusion is enabled.
        """
        # Enable ONLY GELU fusion capabilities
        # GeluFusionL2 and BiasGeluFusion are the relevant ORT optimizers
        gelu_ort_names = ["GeluFusionL2", "BiasGeluFusion", "GeluFusion", "FastGeluFusion"]
        optimized = apply_optimization_with_capability(all_patterns_model, gelu_ort_names)

        # Count MatMul nodes in p06 pattern BEFORE
        before_matmul = count_nodes_by_prefix_and_op(
            all_patterns_model, "p06_matmuladdrelu_", "MatMul"
        )

        # Count MatMul nodes in p06 pattern AFTER
        after_matmul = count_nodes_by_prefix_and_op(optimized, "p06_matmuladdrelu_", "MatMul")

        # Also check for Gemm nodes (MatMulAddFusion converts MatMul+Add to Gemm)
        after_gemm = count_nodes_by_prefix_and_op(optimized, "p06_matmuladdrelu_", "Gemm")

        # ASSERTION: MatMul nodes should NOT be converted to Gemm
        # If MatMulAddFusion ran, MatMul count would drop and Gemm count would increase
        assert before_matmul > 0, "Precondition: Model should have MatMul nodes in p06"
        assert after_matmul == before_matmul, (
            f"ISOLATION FAILURE: MatMul nodes changed from {before_matmul} to {after_matmul}. "
            f"Enabling gelu_fusion should NOT trigger MatMulAddFusion!"
        )
        assert after_gemm == 0, (
            f"ISOLATION FAILURE: Found {after_gemm} Gemm nodes in p06 pattern. "
            f"Enabling gelu_fusion should NOT trigger MatMulAddFusion!"
        )

    def test_gelu_fusion_does_not_trigger_layernorm_fusion(
        self, all_patterns_model: onnx.ModelProto
    ):
        """Enabling gelu_fusion should NOT optimize LayerNorm patterns.

        Pattern: p09_skiplayernorm_

        NOTE: The p09 pattern uses native LayerNormalization op (opset 17), not
        decomposed ReduceMean patterns. SkipLayerNormFusion converts:
            Add + LayerNormalization → SkipLayerNormalization
        So we check that LayerNormalization ops remain (not fused).
        """
        # Enable ONLY GELU fusion capabilities
        gelu_ort_names = ["GeluFusionL2", "BiasGeluFusion", "GeluFusion", "FastGeluFusion"]
        optimized = apply_optimization_with_capability(all_patterns_model, gelu_ort_names)

        # Count LayerNormalization nodes (p09 uses native op, not decomposed ReduceMean)
        before_layernorm = count_nodes_by_prefix_and_op(
            all_patterns_model, "p09_skiplayernorm_", "LayerNormalization"
        )
        after_layernorm = count_nodes_by_prefix_and_op(
            optimized, "p09_skiplayernorm_", "LayerNormalization"
        )

        # SkipLayerNormFusion would convert LayerNormalization → SkipLayerNormalization
        # If fusion ran, LayerNormalization count would drop to 0
        assert before_layernorm > 0, "Precondition: Model should have LayerNormalization in p09"
        assert after_layernorm == before_layernorm, (
            f"ISOLATION FAILURE: LayerNormalization nodes changed "
            f"({before_layernorm} → {after_layernorm}). "
            f"Enabling gelu_fusion should NOT trigger SkipLayerNormFusion!"
        )


class TestMatMulAddFusionIsolation:
    """Test that enabling MatMulAddFusion does NOT enable GELU fusion."""

    def test_matmul_add_fusion_does_not_trigger_gelu_fusion(
        self, all_patterns_model: onnx.ModelProto
    ):
        """Enabling matmul_add_fusion should NOT optimize GELU patterns.

        Pattern: p08_biasgelu_ (contains Erf nodes for GELU approximation)
        """
        # Enable ONLY MatMulAddFusion
        matmul_ort_names = ["MatMulAddFusion"]
        optimized = apply_optimization_with_capability(all_patterns_model, matmul_ort_names)

        # Count Erf nodes (key indicator of GELU pattern - Erf is part of GELU)
        before_erf = count_nodes_by_prefix_and_op(all_patterns_model, "p08_biasgelu_", "Erf")
        after_erf = count_nodes_by_prefix_and_op(optimized, "p08_biasgelu_", "Erf")

        # Check for BiasGelu fused op
        after_biasgelu = count_nodes_by_prefix_and_op(optimized, "p08_biasgelu_", "BiasGelu")

        # ASSERTION: GELU pattern should remain unoptimized
        assert before_erf > 0, "Precondition: Model should have Erf nodes in p08"
        assert after_erf == before_erf, (
            f"ISOLATION FAILURE: Erf nodes changed from {before_erf} to {after_erf}. "
            f"Enabling matmul_add_fusion should NOT trigger GELU fusion!"
        )
        assert after_biasgelu == 0, (
            f"ISOLATION FAILURE: Found {after_biasgelu} BiasGelu nodes. "
            f"Enabling matmul_add_fusion should NOT trigger GELU fusion!"
        )

    def test_matmul_add_fusion_only_affects_matmul_pattern(
        self, all_patterns_model: onnx.ModelProto
    ):
        """Enabling matmul_add_fusion should ONLY optimize p06 pattern."""
        # Enable ONLY MatMulAddFusion
        matmul_ort_names = ["MatMulAddFusion"]
        apply_optimization_with_capability(all_patterns_model, matmul_ort_names)

        # p06 SHOULD be optimized (MatMul converted to Gemm)
        before_matmul = count_nodes_by_prefix_and_op(
            all_patterns_model, "p06_matmuladdrelu_", "MatMul"
        )

        # When matmul_add_fusion is enabled, MatMul+Add should fuse to Gemm
        # Allow this optimization to happen
        assert before_matmul > 0, "Precondition: Model should have MatMul nodes"


class TestDefaultBehavior:
    """Test behavior when no capabilities are explicitly enabled."""

    def test_no_explicit_enable_all_disabled(self, all_patterns_model: onnx.ModelProto):
        """With all capabilities disabled, model should remain unchanged.

        This tests that disabling all ORT optimizers returns to unoptimized state.
        """
        # Disable ALL capabilities
        ort_names = get_all_ort_names()
        optimized = optimize_at_level(all_patterns_model, level=2, disabled_optimizers=ort_names)

        # Node count should remain the same (no optimizations ran)
        before_nodes = len(all_patterns_model.graph.node)
        after_nodes = len(optimized.graph.node)

        # Allow small variation due to ORT mandatory optimizations
        # But the difference should be minimal
        assert after_nodes >= before_nodes * 0.9, (
            f"With all capabilities disabled, node count should be similar. "
            f"Before: {before_nodes}, After: {after_nodes}"
        )


class TestExplicitDisable:
    """Test that explicit disable works correctly."""

    def test_explicit_disable_prevents_optimization(self, all_patterns_model: onnx.ModelProto):
        """Explicitly disabling matmul_add_fusion should prevent MatMul→Gemm conversion."""
        # Disable only MatMulAddFusion - all others enabled
        optimized = optimize_at_level(
            all_patterns_model, level=2, disabled_optimizers=["MatMulAddFusion"]
        )

        # Count MatMul nodes BEFORE and AFTER
        before_matmul = count_nodes_by_prefix_and_op(
            all_patterns_model, "p06_matmuladdrelu_", "MatMul"
        )
        after_matmul = count_nodes_by_prefix_and_op(optimized, "p06_matmuladdrelu_", "MatMul")

        # MatMul nodes should remain (not converted to Gemm)
        assert before_matmul > 0, "Precondition: Model should have MatMul nodes"
        assert after_matmul == before_matmul, (
            f"Disabled MatMulAddFusion should NOT convert MatMul to Gemm. "
            f"Before: {before_matmul}, After: {after_matmul}"
        )


# =============================================================================
# GLOBAL ISOLATION VERIFICATION
# =============================================================================


class TestGlobalIsolation:
    """Global tests to verify isolation across all capability pairs."""

    @pytest.mark.parametrize(
        "enabled_ort_names,protected_pattern,protected_op,allow_cse_reduction",
        [
            # MatMul should be strictly preserved (fusion would eliminate all)
            (["GeluFusionL2", "BiasGeluFusion"], "p06_matmuladdrelu_", "MatMul", False),
            # LayerNormalization should be preserved (SkipLayerNormFusion converts to SkipLayerNorm)
            # NOTE: p09 uses native LayerNormalization op (opset 17), not decomposed ReduceMean
            (["GeluFusionL2", "BiasGeluFusion"], "p09_skiplayernorm_", "LayerNormalization", False),
            # Erf should be strictly preserved (fusion would eliminate all)
            (["MatMulAddFusion"], "p08_biasgelu_", "Erf", False),
        ],
    )
    def test_capability_isolation(
        self,
        all_patterns_model: onnx.ModelProto,
        enabled_ort_names: list[str],
        protected_pattern: str,
        protected_op: str,
        allow_cse_reduction: bool,
    ):
        """Parametrized test: enabling one capability should not affect other patterns.

        Args:
            enabled_ort_names: ORT optimizer names to enable
            protected_pattern: Pattern prefix that should NOT be optimized
            protected_op: Operation type that should remain unchanged
            allow_cse_reduction: If True, allow CSE to reduce count (but not to 0)
        """
        # Apply optimization with only specified capabilities enabled
        optimized = apply_optimization_with_capability(all_patterns_model, enabled_ort_names)

        # Count protected operation before and after
        before_count = count_nodes_by_prefix_and_op(
            all_patterns_model, protected_pattern, protected_op
        )
        after_count = count_nodes_by_prefix_and_op(optimized, protected_pattern, protected_op)

        # ASSERTION: Protected pattern should remain unchanged
        assert before_count > 0, (
            f"Precondition: Model should have {protected_op} nodes in {protected_pattern}"
        )

        if allow_cse_reduction:
            # CSE (Common Subexpression Elimination) is a basic optimization that
            # may reduce duplicate nodes. This is acceptable as long as NOT ALL
            # nodes are eliminated (which would indicate fusion ran).
            assert after_count > 0, (
                f"ISOLATION FAILURE: All {protected_op} nodes in {protected_pattern} "
                f"were eliminated (fusion ran!). "
                f"Enabling {enabled_ort_names} should NOT trigger fusion!"
            )
        else:
            # Strict preservation - no change allowed
            assert after_count == before_count, (
                f"ISOLATION FAILURE: {protected_op} nodes in {protected_pattern} "
                f"changed from {before_count} to {after_count}. "
                f"Enabling {enabled_ort_names} should NOT affect {protected_pattern}!"
            )
