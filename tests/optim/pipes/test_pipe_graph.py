# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ORTGraphPipe process and integration.

Tests the graph optimization pipe's processing logic using 3-criteria verification.
For ORTGraphPipeConfig tests, see test_graph_pipe_config.py.

Test Design (from design doc Section 9.3):
- Uses isolated minimal models per capability (same as test_pipe_graph_isolated.py)
- Each capability gets its own ONNX model containing ONLY the pattern it optimizes
- BUILDER_REGISTRY provides correct builders that actually trigger ORT fusions
- verify_capability_effect validates 3-criteria system
- Parametrized test covers ALL BoolCapability in GRAPH_CAPABILITIES

Architecture Note:
    This file originally used `all_patterns_model` from generate_patterns.py, which
    combined all patterns into a single comprehensive model. However, that approach
    used simplified builders that don't match ORT's pattern requirements, causing
    many fusion tests to fail. The current design uses isolated models with the
    same correct builders as test_pipe_graph_isolated.py, ensuring consistency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import onnx
import pytest

from winml.modelkit.optim.pipes.graph import GRAPH_CAPABILITIES, ORTGraphPipe, ORTGraphPipeConfig

from ..conftest import verify_capability_effect

# Import BUILDER_REGISTRY and model utilities from isolated tests
# This ensures both test files use the SAME working builders
from .test_pipe_graph_isolated import (
    BUILDER_REGISTRY,
    get_model_for_config,
)


# Enable verbose output for debugging graphpipe optimizations
# Set GRAPHPIPE_VERBOSE=1 to see detailed ORT session options during tests
VERBOSE_GRAPHPIPE = os.environ.get("GRAPHPIPE_VERBOSE", "0") == "1"


# =============================================================================
# GRAPH_FUSED_OPS REGISTRY
# =============================================================================
# Maps ORT optimizer name → list of fused op types it produces

GRAPH_FUSED_OPS: dict[str, list[str]] = {
    # GELU optimizers (5)
    "GeluFusionL2": ["Gelu"],
    "BiasGeluFusion": ["BiasGelu"],
    "FastGeluFusion": ["FastGelu"],
    "QuickGeluFusion": ["QuickGelu"],
    "GeluApproximation": ["FastGelu"],
    # LayerNorm optimizers (4)
    "LayerNormFusionL2": ["LayerNormalization"],
    "SkipLayerNormFusion": ["SkipLayerNormalization"],
    "SimplifiedLayerNormFusion": ["SimplifiedLayerNormalization"],
    # NOTE: AttentionFusion is NOT here - it runs via optimize_model() (ORTFusionPipe),
    # not via graph optimization (ORTGraphPipe with SessionOptions)
    # Conv optimizers (4)
    "ConvAddFusion": [],
    "ConvBNFusion": [],
    "ConvMulFusion": [],
    "ConvActivationFusion": ["FusedConv"],
    # Elimination optimizers (3)
    "EliminateSlice": [],
    "ExpandElimination": [],
    "UnsqueezeElimination": [],
    # GEMM optimizers (3)
    "GemmActivationFusion": ["FusedGemm"],
    "GemmSumFusion": [],
    "GemmTransposeFusion": [],
    # Graph optimizers (3)
    "ConcatSliceElimination": [],
    "DoubleQDQPairsRemover": [],
    "ConstantFolding": [],
    # Layout optimizers (4)
    "TransposeOptimizer": [],
    "NhwcTransformer": [],
    "NchwcTransformer": ["ReorderInput", "ReorderOutput"],
    "ConvAddActivationFusion": ["FusedConv"],
    # MatMul optimizers (6)
    "MatMulAddFusion": ["Gemm"],
    "MatMulActivationFusion": ["FusedMatMul"],
    "MatmulTransposeFusion": ["FusedMatMul"],
    "MatMulScaleFusion": ["FusedMatMul"],
    "MatMul_BatchNormalization_Fusion": ["Gemm"],
    "DynamicQuantizeMatMulFusion": ["DynamicQuantizeMatMul"],
    # Activation optimizers (2)
    "BiasSoftmaxFusion": ["BiasSoftmax"],
    "BiasDropoutFusion": ["BiasDropout"],
    # Misc optimizers (4)
    "GatherSliceToSplitFusion": ["Split"],
    "GatherToSliceFusion": ["Slice"],
    "Pad_Fusion": [],
    "NotWhereFusion": [],
}

# All possible fused ops created by ORTGraphPipe (union of all values)
ALL_GRAPH_FUSED_OPS: list[str] = sorted({op for ops in GRAPH_FUSED_OPS.values() for op in ops})


# =============================================================================
# TEST CASE INFRASTRUCTURE
# =============================================================================


@dataclass
class GraphProcessTestCase:
    """Test case for ORTGraphPipe.process capability verification.

    Attributes:
        ort_name: ORT optimizer name (e.g., "GeluFusionL2") - used for test ID
        config: ORTGraphPipeConfig with target capability enabled
        existence_list: Fused ops that MUST exist after optimization
        non_existence_list: Other fused ops that MUST NOT exist
        min_node_reduction: Expected minimum node reduction (0 for transforms)
        extra_disabled: Additional optimizers to disable for this test (e.g., ConstantFolding)
    """

    ort_name: str
    config: ORTGraphPipeConfig
    existence_list: list[str]
    non_existence_list: list[str]
    min_node_reduction: int = 0
    extra_disabled: list[str] | None = None


def make_test_case(
    ort_name: str,
    min_node_reduction: int = 0,
    extra_disabled: list[str] | None = None,
) -> GraphProcessTestCase:
    """Build test case from ORT optimizer name.

    Auto-generates:
    - config with target capability enabled (from ort_name lookup)
    - existence_list from GRAPH_FUSED_OPS
    - non_existence_list from all OTHER fused ops (excluding dependencies' fused ops)

    Args:
        ort_name: ORT optimizer name (e.g., "GeluFusionL2")
        min_node_reduction: Expected minimum node reduction
        extra_disabled: Additional optimizers to disable for this test
            (e.g., ["ConstantFolding"] for GatherToSliceFusion)

    Returns:
        GraphProcessTestCase with all fields populated

    Raises:
        ValueError: If ort_name not found in GRAPH_CAPABILITIES
    """
    # Find capability and python_name from capability with matching ort_name
    target_cap = None
    python_name = None
    for cap in GRAPH_CAPABILITIES.values():
        if hasattr(cap, "ort_name") and cap.ort_name == ort_name:
            target_cap = cap
            python_name = cap.python_name
            break

    if python_name is None or target_cap is None:
        raise ValueError(f"ORT optimizer '{ort_name}' not found in GRAPH_CAPABILITIES")

    # Collect capabilities to enable (target + dependencies)
    enabled_caps = [python_name]

    # Add dependencies recursively
    deps_to_check = list(getattr(target_cap, "depends_on", ()))
    while deps_to_check:
        dep_name = deps_to_check.pop(0)
        # Find the capability for this dependency
        for cap in GRAPH_CAPABILITIES.values():
            if cap.name == dep_name:
                if cap.python_name not in enabled_caps:
                    enabled_caps.append(cap.python_name)
                    # Check this dependency's dependencies too
                    deps_to_check.extend(getattr(cap, "depends_on", ()))
                break

    # Build config with target capability AND dependencies enabled
    config = ORTGraphPipeConfig(enabled=enabled_caps, verbose=VERBOSE_GRAPHPIPE)

    # Build existence list from target capability
    existence_list = GRAPH_FUSED_OPS.get(ort_name, [])

    # Build allowed ops (target ops + dependency ops)
    allowed_ops = set(existence_list)
    for cap in GRAPH_CAPABILITIES.values():
        # Add ops that this capability (or its deps) can create
        if cap.python_name in enabled_caps and hasattr(cap, "ort_name"):
            dep_ops = GRAPH_FUSED_OPS.get(cap.ort_name, [])
            allowed_ops.update(dep_ops)

    # non_existence_list excludes target ops AND dependency ops
    non_existence_list = [op for op in ALL_GRAPH_FUSED_OPS if op not in allowed_ops]

    return GraphProcessTestCase(
        ort_name=ort_name,
        config=config,
        existence_list=existence_list,
        non_existence_list=non_existence_list,
        min_node_reduction=min_node_reduction,
        extra_disabled=extra_disabled,
    )


# =============================================================================
# TEST CASES FOR ALL GRAPH PIPE CAPABILITIES
# =============================================================================
# BoolCapability test cases with proper skip marks
# EP constraints based on ORT source analysis and test_pipe_graph_isolated.py
#
# SKIP CATEGORIES:
# - EliminateSlice, ExpandElimination: Level 1 optimizers (not controllable via disable list)
# - FastGeluFusion: GPU-only (CUDA/ROCm)
# - BiasSoftmaxFusion, BiasDropoutFusion: CUDA EP only
# - MatMulActivationFusion: DirectML EP only
# - NhwcTransformer, NchwcTransformer: AVX2/AVX512 CPU only

GRAPH_PROCESS_TEST_CASES: list[GraphProcessTestCase | pytest.param] = [
    # GELU capabilities (5)
    make_test_case("GeluFusionL2", min_node_reduction=1),
    make_test_case("BiasGeluFusion", min_node_reduction=1),
    pytest.param(
        make_test_case("FastGeluFusion", min_node_reduction=1),
        marks=pytest.mark.skip(reason="FastGeluFusion requires CUDA/ROCm EP"),
        id="FastGeluFusion",
    ),
    make_test_case("QuickGeluFusion", min_node_reduction=1),
    make_test_case("GeluApproximation", min_node_reduction=1),
    # LayerNorm capabilities (3)
    make_test_case("LayerNormFusionL2", min_node_reduction=1),
    make_test_case("SkipLayerNormFusion", min_node_reduction=1),
    make_test_case("SimplifiedLayerNormFusion", min_node_reduction=1),
    # NOTE: AttentionFusion is tested in test_pipe_fusion_direct.py (ORTFusionPipe)
    # Conv capabilities (4)
    make_test_case("ConvAddFusion", min_node_reduction=1),
    make_test_case("ConvBNFusion", min_node_reduction=1),
    make_test_case("ConvMulFusion", min_node_reduction=1),
    make_test_case("ConvActivationFusion", min_node_reduction=1),
    # Elimination capabilities (1)
    # Note: EliminateSlice and ExpandElimination removed - they run at Level 1
    # and cannot be tested with disable_specified_optimizers (L2+ only)
    make_test_case("UnsqueezeElimination", min_node_reduction=1),
    # GEMM capabilities (3)
    make_test_case("GemmActivationFusion", min_node_reduction=1),
    make_test_case("GemmSumFusion", min_node_reduction=1),
    make_test_case("GemmTransposeFusion", min_node_reduction=1),
    # Graph capabilities (2)
    make_test_case("ConcatSliceElimination", min_node_reduction=1),
    make_test_case("DoubleQDQPairsRemover", min_node_reduction=1),
    # Layout capabilities (4)
    make_test_case("TransposeOptimizer", min_node_reduction=1),
    pytest.param(
        make_test_case("NhwcTransformer", min_node_reduction=1),
        marks=pytest.mark.skip(reason="NhwcTransformer requires AVX2/AVX512 CPU"),
        id="NhwcTransformer",
    ),
    pytest.param(
        make_test_case("NchwcTransformer", min_node_reduction=1),
        marks=pytest.mark.skip(reason="NchwcTransformer requires AVX2/AVX512 CPU"),
        id="NchwcTransformer",
    ),
    make_test_case("ConvAddActivationFusion", min_node_reduction=1),
    # MatMul capabilities (6)
    make_test_case("MatMulAddFusion", min_node_reduction=1),
    pytest.param(
        make_test_case("MatMulActivationFusion", min_node_reduction=1),
        marks=pytest.mark.skip(reason="MatMulActivationFusion requires DirectML EP"),
        id="MatMulActivationFusion",
    ),
    make_test_case("MatmulTransposeFusion", min_node_reduction=1),
    make_test_case("MatMulScaleFusion", min_node_reduction=1),
    make_test_case("MatMul_BatchNormalization_Fusion", min_node_reduction=1),
    make_test_case("DynamicQuantizeMatMulFusion", min_node_reduction=1),
    # Activation capabilities (2) - EP-specific
    pytest.param(
        make_test_case("BiasSoftmaxFusion", min_node_reduction=1),
        marks=pytest.mark.skip(reason="BiasSoftmaxFusion requires CUDA EP"),
        id="BiasSoftmaxFusion",
    ),
    pytest.param(
        make_test_case("BiasDropoutFusion", min_node_reduction=1),
        marks=pytest.mark.skip(reason="BiasDropoutFusion requires CUDA EP"),
        id="BiasDropoutFusion",
    ),
    # Misc capabilities (4)
    # GatherSliceToSplitFusion converts Gather ops to Split - may add nodes
    make_test_case("GatherSliceToSplitFusion", min_node_reduction=-10),
    # GatherToSliceFusion converts Range+Gather to Slice+Unsqueezes - may add nodes
    make_test_case(
        "GatherToSliceFusion",
        min_node_reduction=-5,
        extra_disabled=["ConstantFolding"],
    ),
    make_test_case("Pad_Fusion", min_node_reduction=1),
    make_test_case("NotWhereFusion", min_node_reduction=1),
]


# =============================================================================
# PARAMETRIZED PROCESS TESTS
# =============================================================================


def _get_test_ids() -> list[str]:
    """Generate test IDs for all test cases.

    Returns IDs for parametrized tests, handling both direct GraphProcessTestCase
    and pytest.param wrapped cases.
    """
    ids = []
    for case in GRAPH_PROCESS_TEST_CASES:
        # Check if this is a ParameterSet (pytest.param result) by checking for 'values' attr
        if hasattr(case, "values"):
            # pytest.param - use explicit id if available, else extract from values
            if hasattr(case, "id") and case.id:
                ids.append(case.id)
            else:
                ids.append(case.values[0].ort_name)
        else:
            ids.append(case.ort_name)
    return ids


class TestORTGraphPipeProcess:
    """Tests for ORTGraphPipe.process method with 3-criteria verification.

    Uses isolated minimal models per capability, ensuring patterns actually
    trigger ORT fusions. This is consistent with test_pipe_graph_isolated.py.
    """

    @pytest.mark.parametrize("case", GRAPH_PROCESS_TEST_CASES, ids=_get_test_ids())
    def test_graph_pipe_process(
        self,
        case: GraphProcessTestCase,
    ) -> None:
        """Test ORTGraphPipe.process for a single capability.

        Uses 3-criteria verification:
        1. Node count reduction (min_node_reduction)
        2. Existence check (existence_list fused ops MUST exist)
        3. Isolation check (non_existence_list fused ops MUST NOT exist)

        Architecture Note:
            Uses isolated models from BUILDER_REGISTRY instead of all_patterns_model.
            This ensures patterns actually trigger ORT's fusions. The all_patterns_model
            approach used simplified builders that didn't match ORT's pattern requirements.

        Args:
            case: Test case with capability info and expected results.
        """
        # Check if we have a builder for this ORT name
        if case.ort_name not in BUILDER_REGISTRY:
            pytest.skip(f"No builder registered for {case.ort_name}")

        # Get the builder config and create isolated model
        builder_config = BUILDER_REGISTRY[case.ort_name]
        model = get_model_for_config(builder_config, prefix=f"{case.ort_name}_")

        pipe = ORTGraphPipe()

        # Baseline: Use ORIGINAL model (no ORT processing)
        # IMPORTANT: We cannot use ORTGraphPipe with all disabled as baseline because:
        # - ORT's Level 1 optimizers (like LayerNormFusion) run regardless of disable list
        # - disable_specified_optimizers only works for Level 2+ optimizers
        # - Using original model gives us true "before optimization" state
        baseline = model

        # Optimized: use config from test case
        test_config = case.config

        # Apply extra_disabled if specified in test case
        if case.extra_disabled:
            for opt_name in case.extra_disabled:
                if opt_name not in test_config.disabled_optimizers:
                    test_config.disabled_optimizers.append(opt_name)

        # Also apply extra_disabled from BuilderConfig if specified
        if builder_config.extra_disabled:
            for opt_name in builder_config.extra_disabled:
                if opt_name not in test_config.disabled_optimizers:
                    test_config.disabled_optimizers.append(opt_name)

        optimized = pipe.process(model, test_config)

        # Verify 3-criteria
        verify_capability_effect(
            model_before=baseline,
            model_after=optimized,
            existence_list=case.existence_list,
            non_existence_list=case.non_existence_list,
            min_node_reduction=case.min_node_reduction,
        )


# =============================================================================
# BASIC PROCESS TESTS (from original file)
# =============================================================================


class TestORTGraphPipeBasic:
    """Basic tests for ORTGraphPipe.process method."""

    def test_process_returns_model(self, sample_model: onnx.ModelProto) -> None:
        """Process should return an ONNX ModelProto."""
        pipe = ORTGraphPipe()
        config = ORTGraphPipeConfig(enabled=["gelu_fusion"])

        result = pipe.process(sample_model, config)

        assert isinstance(result, onnx.ModelProto)
        assert result.graph is not None

    def test_process_preserves_model_structure(self, sample_model: onnx.ModelProto) -> None:
        """Process should preserve basic model structure."""
        pipe = ORTGraphPipe()
        config = ORTGraphPipeConfig(enabled=["gelu_fusion"])

        result = pipe.process(sample_model, config)

        assert len(result.graph.input) == len(sample_model.graph.input)
        assert len(result.graph.output) == len(sample_model.graph.output)

    def test_process_with_all_disabled(self, sample_model: onnx.ModelProto) -> None:
        """Process with all disabled (default) should return model unchanged."""
        pipe = ORTGraphPipe()
        config = ORTGraphPipeConfig()  # All disabled by default

        result = pipe.process(sample_model, config)

        assert isinstance(result, onnx.ModelProto)
        # Node count should be preserved when all optimizers disabled
        assert len(result.graph.node) == len(sample_model.graph.node)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestORTGraphPipeIntegration:
    """Integration tests for ORTGraphPipe end-to-end workflow."""

    def test_pipe_class_attributes(self) -> None:
        """Verify ORTGraphPipe has required class attributes."""
        assert hasattr(ORTGraphPipe, "name")
        assert hasattr(ORTGraphPipe, "capabilities")
        assert ORTGraphPipe.name == "ort_graph"
        assert isinstance(ORTGraphPipe.capabilities, dict)
        assert len(ORTGraphPipe.capabilities) > 0

    def test_pipe_has_capabilities_dict(self) -> None:
        """ORTGraphPipe should have a capabilities class attribute."""
        caps = ORTGraphPipe.capabilities

        assert isinstance(caps, dict)
        assert len(caps) > 0
        assert caps is GRAPH_CAPABILITIES

    def test_end_to_end_workflow(self, sample_model: onnx.ModelProto) -> None:
        """Test complete workflow from kwargs to processed model."""
        # Step 1: Build config from user kwargs (isolation mode)
        config = ORTGraphPipe.build_config(gelu_fusion=True, matmul_add_fusion=False)

        # Step 2: Create pipe and process model
        pipe = ORTGraphPipe()
        result = pipe.process(sample_model, config)

        # Verify results
        assert isinstance(result, onnx.ModelProto)
        assert config.optimization_level == 2
        # gelu_fusion enabled, matmul_add_fusion disabled
        assert "GeluFusionL2" not in config.disabled_optimizers
        assert "MatMulAddFusion" in config.disabled_optimizers

    def test_should_process_returns_true(self) -> None:
        """should_process returns True when optimization_level > 0."""
        config = ORTGraphPipeConfig()  # Level 2 by default
        assert ORTGraphPipe.should_process(config) is True


# =============================================================================
# REGISTRY VALIDATION TESTS
# =============================================================================


class TestGraphFusedOpsRegistry:
    """Tests to validate GRAPH_FUSED_OPS registry completeness."""

    def test_all_graph_capabilities_have_fused_ops_entry(self) -> None:
        """Every capability in GRAPH_CAPABILITIES should have a GRAPH_FUSED_OPS entry."""
        missing = [
            cap.ort_name
            for cap in GRAPH_CAPABILITIES.values()
            if hasattr(cap, "ort_name") and cap.ort_name and cap.ort_name not in GRAPH_FUSED_OPS
        ]

        assert not missing, f"Missing GRAPH_FUSED_OPS entries: {missing}"

    def test_graph_fused_ops_no_unknown_optimizers(self) -> None:
        """GRAPH_FUSED_OPS should not have entries for unknown optimizers."""
        known_ort_names = {
            cap.ort_name
            for cap in GRAPH_CAPABILITIES.values()
            if hasattr(cap, "ort_name") and cap.ort_name
        }

        unknown = [name for name in GRAPH_FUSED_OPS if name not in known_ort_names]
        assert not unknown, f"Unknown optimizers in GRAPH_FUSED_OPS: {unknown}"

    def test_all_graph_fused_ops_unique(self) -> None:
        """ALL_GRAPH_FUSED_OPS should have unique entries."""
        assert len(ALL_GRAPH_FUSED_OPS) == len(set(ALL_GRAPH_FUSED_OPS))

    def test_test_cases_cover_all_bool_capabilities(self) -> None:
        """Test cases should cover all BoolCapability in GRAPH_CAPABILITIES."""
        from winml.modelkit.optim.registry import BoolCapability

        # Optimizers that cannot be tested with the standard enable/disable pattern:
        # - L1 optimizers: Run at ORT_ENABLE_BASIC level, not Level 2+
        # - default=True optimizers: These are for DISABLING, not enabling
        l1_untestable = {"EliminateSlice", "ExpandElimination", "ConstantFolding"}

        # Get all BoolCapability ORT names (excluding L1 untestable)
        bool_cap_ort_names = {
            cap.ort_name
            for cap in GRAPH_CAPABILITIES.values()
            if isinstance(cap, BoolCapability)
            and hasattr(cap, "ort_name")
            and cap.ort_name
            and cap.ort_name not in l1_untestable
        }

        # Get ORT names from test cases (handle both direct and pytest.param wrapped)
        tested_ort_names = set()
        for case in GRAPH_PROCESS_TEST_CASES:
            if hasattr(case, "values"):
                # pytest.param wrapped case
                tested_ort_names.add(case.values[0].ort_name)
            else:
                tested_ort_names.add(case.ort_name)

        missing = bool_cap_ort_names - tested_ort_names
        assert not missing, f"Missing test cases for BoolCapabilities: {missing}"
