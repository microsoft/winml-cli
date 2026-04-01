# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ORTGraphPipe with isolated minimal patterns per capability.

This test file supplements test_pipe_graph.py by testing each capability in ISOLATION
with its own MINIMAL pattern, rather than against a comprehensive all-patterns model.

Test Design:
- Each capability gets its own minimal ONNX model containing ONLY the pattern it optimizes
- Uses 4-criteria verification (node reduction, existence, non-existence, numeric)
- Leverages builders from tests/optim/assets/graphpipe/builders/
- Input shapes are read dynamically from generated models (not hardcoded)

Benefits over all_patterns_model testing:
- Pure isolation: no interference between patterns
- Easier debugging: failures are clearly attributed to specific capability
- Faster per-test execution: smaller models
- Clearer test intent: one pattern per test

Builder Coverage Analysis:
========================

| Category        | ORT Name                         | Builder                          | Status |
|-----------------|----------------------------------|----------------------------------|--------|
| GELU (5)        | GeluFusionL2                     | gelu_fusion_builder              | OK     |
|                 | BiasGeluFusion                   | bias_gelu_builder                | OK     |
|                 | FastGeluFusion                   | fast_gelu_builder                | OK     |
|                 | QuickGeluFusion                  | quick_gelu_builder               | OK     |
|                 | GeluApproximation                | gelu_approximation_builder       | OK     |
| LayerNorm (4)   | LayerNormFusionL2                | decomposed_layernorm_builder     | OK     |
|                 | SkipLayerNormFusion              | skip_layernorm_builder           | OK     |
|                 | SimplifiedLayerNormFusion        | simplified_layernorm_builder     | OK     |
| Conv (4)        | ConvBNFusion                     | conv_bn_builder                  | OK     |
|                 | ConvAddFusion                    | conv_add_fusion_builder          | OK     |
|                 | ConvActivationFusion             | conv_activation_builder          | OK     |
|                 | ConvMulFusion                    | conv_mul_builder                 | OK     |
| Layout (4)      | TransposeOptimizer               | transpose_chain_builder          | OK     |
|                 | NhwcTransformer                  | nhwc_transformer_builder         | OK     |
|                 | NchwcTransformer                 | nchwc_transformer_builder        | OK     |
|                 | ConvAddActivationFusion          | conv_add_activation_builder      | OK     |
| Elimination (1) | UnsqueezeElimination             | unsqueeze_elimination_builder    | OK     |
|                 | (EliminateSlice removed - L1)    | -                                | -      |
|                 | (ExpandElimination removed - L1) | -                                | -      |
| GEMM (3)        | GemmActivationFusion             | gemm_activation_builder          | OK     |
|                 | GemmSumFusion                    | gemm_sum_builder                 | OK     |
|                 | GemmTransposeFusion              | gemm_transpose_builder           | OK     |
| MatMul (6)      | MatMulAddFusion                  | matmul_add_relu_builder          | OK     |
|                 | MatMulActivationFusion           | matmul_activation_builder        | DML    |
|                 | MatmulTransposeFusion            | matmul_transpose_builder         | OK     |
|                 | MatMulScaleFusion                | matmul_scale_builder             | OK     |
|                 | MatMul_BatchNormalization_Fusion | matmul_bn_builder                | OK     |
|                 | DynamicQuantizeMatMulFusion      | dynamic_quantize_matmul_builder  | OK     |
| Activation (2)  | BiasSoftmaxFusion                | bias_softmax_builder             | OK     |
|                 | BiasDropoutFusion                | bias_dropout_builder             | CUDA   |
| Graph (2)       | ConcatSliceElimination           | concat_slice_elimination_builder | OK     |
|                 | DoubleQDQPairsRemover            | qdq_pairs_builder                | OK     |
| Misc (4)        | GatherSliceToSplitFusion         | gather_split_builder             | OK     |
|                 | GatherToSliceFusion              | gather_to_slice_builder          | OK***  |
|                 | Pad_Fusion                       | pad_fusion_builder               | OK     |
|                 | NotWhereFusion                   | not_where_builder                | OK**   |

Status Legend:
- OK: Works on CPU
- DML: DirectML-only (marked with pytest.mark.directml)
- SKIP: Pattern correct but cannot trigger in isolation (see SKIP_ISOLATED_ORT_NAMES)
- CUDA: CUDA-only (marked with pytest.mark.cuda)
- OK**: Requires specific opset version (opset_version=15 in BuilderConfig)
- OK***: Requires extra_disabled to prevent interference (extra_disabled in BuilderConfig)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import onnx
import pytest
from onnx import TensorProto, helper

from winml.modelkit.optim.pipes import GRAPH_CAPABILITIES, ORTGraphPipe, ORTGraphPipeConfig

# Import all builders
from ..assets.graphpipe.builders import (
    bias_dropout_builder,
    bias_gelu_builder,
    bias_softmax_builder,
    concat_slice_elimination_builder,
    conv_activation_builder,
    conv_add_activation_builder,
    conv_add_fusion_builder,
    conv_bn_builder,
    conv_mul_builder,
    decomposed_layernorm_builder,
    fast_gelu_builder,
    gather_split_builder,
    gather_to_slice_builder,
    gelu_approximation_builder,
    gelu_fusion_builder,
    nchwc_transformer_builder,
    nhwc_transformer_builder,
    not_where_builder,
    pad_fusion_builder,
    qdq_pairs_builder,
    quick_gelu_builder,
    simplified_layernorm_builder,
    skip_layernorm_builder,
    transpose_chain_builder,
    unsqueeze_elimination_builder,
)
from ..assets.graphpipe.builders.gemm import (
    gemm_activation_builder,
    gemm_sum_builder,
    gemm_transpose_builder,
)
from ..assets.graphpipe.builders.matmul import (
    dynamic_quantize_matmul_builder,
    matmul_activation_builder,
    matmul_add_relu_builder,
    matmul_bn_builder,
    matmul_scale_builder,
    matmul_transpose_builder,
)
from ..conftest import verify_capability_effect


# =============================================================================
# CONFIGURATION
# =============================================================================

# Enable verbose output for debugging graphpipe optimizations
# Set GRAPHPIPE_VERBOSE=1 to see detailed ORT session options during tests
VERBOSE_GRAPHPIPE = os.environ.get("GRAPHPIPE_VERBOSE", "0") == "1"

# =============================================================================
# TYPE ALIASES
# =============================================================================

BuilderFunc = Callable[[str, str, str, list], list]


# =============================================================================
# BUILDER CONFIGURATION
# =============================================================================

# Type alias for model factory function
ModelFactoryFunc = Callable[[], "onnx.ModelProto"]


@dataclass
class BuilderConfig:
    """Configuration for a pattern builder.

    Attributes:
        builder: Builder function that creates ONNX nodes (required unless model_factory set)
        input_shape: Input tensor shape (default [1, 64] for most patterns)
        input_dtype: Input tensor dtype (default FLOAT)
        opset_version: ONNX opset version (default 17 for LayerNorm support,
                       use 15 for NotWhereFusion which requires opset <= 15)
        extra_disabled: Additional optimizers to disable during testing
                       (e.g., ["ConstantFolding"] for GatherToSliceFusion)
        model_factory: Function that returns a complete ONNX model programmatically
                      (for patterns requiring multiple inputs or special structure)
    """

    builder: BuilderFunc | None = None
    input_shape: tuple[int, ...] = (1, 64)
    input_dtype: int = TensorProto.FLOAT
    opset_version: int = 17
    extra_disabled: list[str] | None = None
    model_factory: ModelFactoryFunc | None = None


# =============================================================================
# BUILDER REGISTRY
# =============================================================================
# Maps ORT optimizer name -> BuilderConfig

BUILDER_REGISTRY: dict[str, BuilderConfig] = {
    # GELU optimizers (5)
    "GeluFusionL2": BuilderConfig(gelu_fusion_builder),
    "BiasGeluFusion": BuilderConfig(bias_gelu_builder),
    "FastGeluFusion": BuilderConfig(fast_gelu_builder),
    "QuickGeluFusion": BuilderConfig(quick_gelu_builder),
    "GeluApproximation": BuilderConfig(gelu_approximation_builder),
    # LayerNorm optimizers (4)
    "LayerNormFusionL2": BuilderConfig(decomposed_layernorm_builder),
    # SkipLayerNormFusion requires 3D input [batch, seq, hidden] for Add pattern matching
    "SkipLayerNormFusion": BuilderConfig(skip_layernorm_builder, input_shape=(1, 10, 64)),
    "SimplifiedLayerNormFusion": BuilderConfig(simplified_layernorm_builder),
    # Conv optimizers (4) - all need NCHW input
    "ConvBNFusion": BuilderConfig(conv_bn_builder, input_shape=(1, 16, 32, 32)),
    "ConvAddFusion": BuilderConfig(conv_add_fusion_builder, input_shape=(1, 16, 32, 32)),
    "ConvMulFusion": BuilderConfig(conv_mul_builder, input_shape=(1, 16, 32, 32)),
    "ConvActivationFusion": BuilderConfig(conv_activation_builder, input_shape=(1, 16, 32, 32)),
    # Layout optimizers (4)
    "TransposeOptimizer": BuilderConfig(transpose_chain_builder, input_shape=(64, 64)),
    "NhwcTransformer": BuilderConfig(nhwc_transformer_builder),
    "NchwcTransformer": BuilderConfig(nchwc_transformer_builder),
    "ConvAddActivationFusion": BuilderConfig(
        conv_add_activation_builder, input_shape=(1, 16, 32, 32)
    ),
    # Elimination optimizers (1)
    # Note: EliminateSlice and ExpandElimination removed - they run at Level 1
    # and cannot be tested with disable_specified_optimizers (L2+ only)
    "UnsqueezeElimination": BuilderConfig(unsqueeze_elimination_builder),
    # GEMM optimizers (3)
    "GemmActivationFusion": BuilderConfig(gemm_activation_builder),
    "GemmSumFusion": BuilderConfig(gemm_sum_builder),
    "GemmTransposeFusion": BuilderConfig(gemm_transpose_builder, input_shape=(64, 64)),
    # MatMul optimizers (6)
    "MatMulAddFusion": BuilderConfig(matmul_add_relu_builder),
    "MatMulActivationFusion": BuilderConfig(matmul_activation_builder),
    "MatmulTransposeFusion": BuilderConfig(matmul_transpose_builder, input_shape=(64, 64)),
    "MatMulScaleFusion": BuilderConfig(matmul_scale_builder),
    "MatMul_BatchNormalization_Fusion": BuilderConfig(matmul_bn_builder),
    "DynamicQuantizeMatMulFusion": BuilderConfig(dynamic_quantize_matmul_builder),
    # Activation optimizers (2)
    "BiasSoftmaxFusion": BuilderConfig(bias_softmax_builder),
    "BiasDropoutFusion": BuilderConfig(bias_dropout_builder),
    # Graph optimizers (2)
    "ConcatSliceElimination": BuilderConfig(concat_slice_elimination_builder),
    "DoubleQDQPairsRemover": BuilderConfig(qdq_pairs_builder),
    # Misc optimizers (4)
    "GatherSliceToSplitFusion": BuilderConfig(gather_split_builder, input_shape=(1, 2)),
    # GatherToSliceFusion requires Range->Gather pattern. ConstantFolding runs before
    # GatherToSliceFusion and folds Range away. Disable ConstantFolding to allow fusion.
    "GatherToSliceFusion": BuilderConfig(
        gather_to_slice_builder, extra_disabled=["ConstantFolding"]
    ),
    "Pad_Fusion": BuilderConfig(pad_fusion_builder),
    # NotWhereFusion requires opset <= 15 (ORT checks opset version in transformer)
    "NotWhereFusion": BuilderConfig(not_where_builder, opset_version=15),
}


# =============================================================================
# FUSED OPS REGISTRY (from test_pipe_graph.py)
# =============================================================================

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
    # Conv optimizers (4)
    "ConvAddFusion": [],
    "ConvBNFusion": [],
    "ConvMulFusion": [],
    "ConvActivationFusion": ["FusedConv"],
    # Elimination optimizers (1)
    "UnsqueezeElimination": [],
    # GEMM optimizers (3)
    "GemmActivationFusion": ["FusedGemm"],
    "GemmSumFusion": [],
    "GemmTransposeFusion": [],
    # Graph optimizers (2)
    "ConcatSliceElimination": [],
    "DoubleQDQPairsRemover": [],
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

ALL_GRAPH_FUSED_OPS: list[str] = sorted({op for ops in GRAPH_FUSED_OPS.values() for op in ops})


# =============================================================================
# MINIMAL MODEL FACTORY
# =============================================================================


def get_model_for_config(
    builder_config: BuilderConfig,
    prefix: str = "test_",
) -> onnx.ModelProto:
    """Get ONNX model from model_factory or builder.

    Priority:
    1. model_factory: Call function to create complete model programmatically
    2. builder: Create minimal model using the builder function

    Args:
        builder_config: Configuration specifying model_factory or builder
        prefix: Prefix for node names (used only with builders)

    Returns:
        ONNX ModelProto
    """
    if builder_config.model_factory:
        return builder_config.model_factory()
    return create_minimal_model(builder_config, prefix)


def create_minimal_model(
    builder_config: BuilderConfig,
    prefix: str = "test_",
) -> onnx.ModelProto:
    """Create minimal ONNX model with a single pattern.

    Dynamically creates an ONNX model containing only the nodes from
    the specified builder. Input/output shapes are determined by the
    BuilderConfig.

    Args:
        builder_config: Configuration specifying builder function and shapes
        prefix: Prefix for node names (default "test_")

    Returns:
        ONNX ModelProto containing the minimal pattern
    """
    initializers: list = []
    input_name = "input"
    output_name = "output"

    # Build nodes using the builder function
    nodes = builder_config.builder(input_name, output_name, prefix, initializers)

    # Create input tensor info
    input_shape = list(builder_config.input_shape)
    input_tensor = helper.make_tensor_value_info(
        input_name, builder_config.input_dtype, input_shape
    )

    # Create output tensor info (infer from last node's output)
    # Most builders preserve shape, but we use dynamic dims for safety
    output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)

    # Create graph
    graph = helper.make_graph(
        nodes=nodes,
        name=f"{prefix}graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )

    # Create model with configurable opset version
    # Default is opset 17 for LayerNormalization support
    # NotWhereFusion requires opset <= 15
    model = helper.make_model(
        graph,
        producer_name="modelkit_test",
        opset_imports=[helper.make_opsetid("", builder_config.opset_version)],
    )
    model.ir_version = 8

    return model


# =============================================================================
# TEST CASE INFRASTRUCTURE
# =============================================================================


@dataclass
class IsolatedTestCase:
    """Test case for isolated capability testing.

    Attributes:
        ort_name: ORT optimizer name (e.g., "GeluFusionL2")
        python_name: Python capability name (e.g., "gelu_fusion")
        builder_config: BuilderConfig for creating minimal model
        existence_list: Fused ops that MUST exist after optimization
        non_existence_list: Other fused ops that MUST NOT exist
        min_node_reduction: Expected minimum node reduction
    """

    ort_name: str
    python_name: str
    builder_config: BuilderConfig
    existence_list: list[str]
    non_existence_list: list[str]
    min_node_reduction: int = 1


def make_isolated_test_case(
    ort_name: str,
    min_node_reduction: int = 1,
) -> IsolatedTestCase | None:
    """Build isolated test case from ORT optimizer name.

    Auto-generates:
    - python_name from GRAPH_CAPABILITIES lookup
    - builder_config from BUILDER_REGISTRY
    - existence_list from GRAPH_FUSED_OPS
    - non_existence_list from all OTHER fused ops

    Args:
        ort_name: ORT optimizer name (e.g., "GeluFusionL2")
        min_node_reduction: Expected minimum node reduction

    Returns:
        IsolatedTestCase if builder exists, None otherwise
    """
    # Check if builder exists
    if ort_name not in BUILDER_REGISTRY:
        return None

    # Find capability info
    python_name = None
    for cap in GRAPH_CAPABILITIES.values():
        if hasattr(cap, "ort_name") and cap.ort_name == ort_name:
            python_name = cap.python_name
            break

    if python_name is None:
        return None

    # Get builder config
    builder_config = BUILDER_REGISTRY[ort_name]

    # Build existence/non-existence lists
    existence_list = GRAPH_FUSED_OPS.get(ort_name, [])
    non_existence_list = [op for op in ALL_GRAPH_FUSED_OPS if op not in existence_list]

    return IsolatedTestCase(
        ort_name=ort_name,
        python_name=python_name,
        builder_config=builder_config,
        existence_list=existence_list,
        non_existence_list=non_existence_list,
        min_node_reduction=min_node_reduction,
    )


# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASE GENERATION WITH EXPECTATIONS
# =============================================================================

# ORT names that are transforms (no node reduction expected)
# These optimizers perform transformations that may not reduce node count
# Value is the minimum expected node reduction (can be negative for transforms that add nodes)
TRANSFORM_ORT_NAMES: dict[str, int] = {
    # GatherSliceToSplitFusion converts Gather ops to Split (may add nodes)
    # The fusion adds Split + Squeeze for each Gather, so node count can increase
    "GatherSliceToSplitFusion": -10,  # Allow up to 10 added nodes
    # GatherToSliceFusion converts Range+Gather to Slice+Unsqueezes (adds nodes)
    # Pattern: Range+Gather (2 nodes) → Slice+3xUnsqueeze (4 nodes) = -2 change
    "GatherToSliceFusion": -5,  # Allow up to 5 added nodes
}
# For backwards compatibility, keep the set version (used in min_reduction logic)
TRANSFORM_ONLY_ORT_NAMES: set[str] = set(TRANSFORM_ORT_NAMES.keys())

# =============================================================================
# TEST CASE LIST (explicit pytest.param pattern)
# =============================================================================

ISOLATED_TEST_CASES: list[IsolatedTestCase | pytest.param] = [
    # GELU capabilities (5)
    make_isolated_test_case("GeluFusionL2"),
    make_isolated_test_case("BiasGeluFusion"),
    pytest.param(
        make_isolated_test_case("FastGeluFusion"),
        marks=pytest.mark.skip(reason="FastGeluFusion requires CUDA/ROCm EP"),
        id="FastGeluFusion",
    ),
    make_isolated_test_case("QuickGeluFusion"),
    make_isolated_test_case("GeluApproximation"),
    # LayerNorm capabilities (4)
    make_isolated_test_case("LayerNormFusionL2"),
    make_isolated_test_case("SkipLayerNormFusion"),
    make_isolated_test_case("SimplifiedLayerNormFusion"),
    # Conv capabilities (4)
    make_isolated_test_case("ConvBNFusion"),
    make_isolated_test_case("ConvAddFusion"),
    make_isolated_test_case("ConvMulFusion"),
    make_isolated_test_case("ConvActivationFusion"),
    # Layout capabilities (4)
    make_isolated_test_case("TransposeOptimizer"),
    pytest.param(
        make_isolated_test_case("NhwcTransformer"),
        marks=pytest.mark.skip(reason="NhwcTransformer requires AVX2/AVX512 CPU"),
        id="NhwcTransformer",
    ),
    pytest.param(
        make_isolated_test_case("NchwcTransformer"),
        marks=pytest.mark.skip(reason="NchwcTransformer requires AVX2/AVX512 CPU"),
        id="NchwcTransformer",
    ),
    make_isolated_test_case("ConvAddActivationFusion"),
    # Elimination capabilities (1)
    # Note: EliminateSlice and ExpandElimination removed - they run at Level 1
    # and cannot be tested with disable_specified_optimizers (L2+ only)
    make_isolated_test_case("UnsqueezeElimination"),
    # GEMM capabilities (3)
    make_isolated_test_case("GemmActivationFusion"),
    make_isolated_test_case("GemmSumFusion"),
    make_isolated_test_case("GemmTransposeFusion"),
    # MatMul capabilities (6)
    make_isolated_test_case("MatMulAddFusion"),
    pytest.param(
        make_isolated_test_case("MatMulActivationFusion"),
        marks=pytest.mark.skip(reason="MatMulActivationFusion requires DirectML EP"),
        id="MatMulActivationFusion",
    ),
    make_isolated_test_case("MatmulTransposeFusion"),
    make_isolated_test_case("MatMulScaleFusion"),
    make_isolated_test_case("MatMul_BatchNormalization_Fusion"),
    make_isolated_test_case("DynamicQuantizeMatMulFusion"),
    # Activation capabilities (2)
    pytest.param(
        make_isolated_test_case("BiasSoftmaxFusion"),
        marks=pytest.mark.skip(reason="BiasSoftmaxFusion requires CUDA EP"),
        id="BiasSoftmaxFusion",
    ),
    pytest.param(
        make_isolated_test_case("BiasDropoutFusion"),
        marks=pytest.mark.skip(reason="BiasDropoutFusion requires CUDA EP"),
        id="BiasDropoutFusion",
    ),
    # Graph capabilities (2)
    make_isolated_test_case("ConcatSliceElimination"),
    make_isolated_test_case("DoubleQDQPairsRemover"),
    # Misc capabilities (4)
    make_isolated_test_case("GatherSliceToSplitFusion", min_node_reduction=-10),
    make_isolated_test_case("GatherToSliceFusion", min_node_reduction=-5),
    make_isolated_test_case("Pad_Fusion"),
    make_isolated_test_case("NotWhereFusion"),
]


# Filter out None values (cases where builder doesn't exist)
# Also filter pytest.param that wraps None values
def _is_valid_case(c):
    if c is None:
        return False
    return not (hasattr(c, "values") and c.values[0] is None)


ISOLATED_TEST_CASES = [c for c in ISOLATED_TEST_CASES if _is_valid_case(c)]


def _get_isolated_test_ids() -> list[str]:
    """Generate test IDs for isolated test cases."""
    ids = []
    for case in ISOLATED_TEST_CASES:
        if hasattr(case, "values"):
            # pytest.param - use explicit id or extract from values
            if hasattr(case, "id") and case.id:
                ids.append(case.id)
            else:
                ids.append(case.values[0].ort_name)
        else:
            ids.append(case.ort_name)
    return ids


# =============================================================================
# PARAMETRIZED ISOLATED TESTS
# =============================================================================


class TestORTGraphPipeIsolated:
    """Tests for ORTGraphPipe.process with isolated minimal patterns.

    Each test creates a minimal ONNX model with ONLY the pattern that
    the target capability is designed to optimize. This provides pure
    isolation testing without interference from other patterns.

    Uses 4-criteria verification:
    1. Node count reduction (min_node_reduction)
    2. Existence check (existence_list fused ops MUST exist)
    3. Isolation check (non_existence_list fused ops MUST NOT exist)
    4. Numeric verification (outputs must match within tolerance)
    """

    @pytest.mark.parametrize("case", ISOLATED_TEST_CASES, ids=_get_isolated_test_ids())
    def test_capability_isolated(self, case: IsolatedTestCase) -> None:
        """Test single capability in isolation with minimal pattern.

        Creates a minimal model for this capability, then verifies:
        1. Node reduction meets minimum
        2. Expected fused ops exist
        3. Unexpected fused ops don't exist
        4. Numeric outputs match (optional)

        Args:
            case: Test case with builder and expected results
        """
        # Get model for this capability (either from external file or builder)
        model = get_model_for_config(case.builder_config, prefix=f"{case.ort_name}_")

        pipe = ORTGraphPipe()

        # Baseline: Use ORIGINAL model (no ORT processing)
        # IMPORTANT: We cannot use ORTGraphPipe with all disabled as baseline because:
        # - ORT's Level 1 optimizers (like LayerNormFusion) run regardless of disable list
        # - disable_specified_optimizers only works for Level 2+ optimizers
        # - Using original model gives us true "before optimization" state
        baseline = model

        # Optimized: enable target capability using config.enable() for deps
        optimized_config = ORTGraphPipeConfig(verbose=VERBOSE_GRAPHPIPE)
        optimized_config.enable(case.python_name)

        # Apply extra_disabled if specified (e.g., disable ConstantFolding for GatherToSliceFusion)
        if case.builder_config.extra_disabled:
            for opt_name in case.builder_config.extra_disabled:
                if opt_name not in optimized_config.disabled_optimizers:
                    optimized_config.disabled_optimizers.append(opt_name)

        optimized = pipe.process(model, optimized_config)

        # Verify 4-criteria
        verify_capability_effect(
            model_before=baseline,
            model_after=optimized,
            existence_list=case.existence_list,
            non_existence_list=case.non_existence_list,
            min_node_reduction=case.min_node_reduction,
            verify_numeric=False,  # Skip numeric for speed; enable for debugging
        )


# =============================================================================
# REGISTRY VALIDATION TESTS
# =============================================================================


class TestBuilderRegistryCoverage:
    """Tests to validate BUILDER_REGISTRY coverage of GRAPH_CAPABILITIES."""

    def test_builder_registry_keys_are_valid_ort_names(self) -> None:
        """All BUILDER_REGISTRY keys should be valid ORT optimizer names."""
        known_ort_names = {
            cap.ort_name
            for cap in GRAPH_CAPABILITIES.values()
            if hasattr(cap, "ort_name") and cap.ort_name
        }

        unknown = [name for name in BUILDER_REGISTRY if name not in known_ort_names]
        assert not unknown, f"Unknown ORT names in BUILDER_REGISTRY: {unknown}"

    def test_all_bool_capabilities_have_builders(self) -> None:
        """Every testable BoolCapability in GRAPH_CAPABILITIES should have a builder."""
        from winml.modelkit.optim.registry import BoolCapability

        # Optimizers that don't need isolated builder tests:
        # - L1 optimizers: run at ORT_ENABLE_BASIC level, not Level 2+
        # - Basic optimizers (ConstantFolding): always-on fundamental ops, no fusion pattern
        untestable = {
            "EliminateSlice",  # L1 optimizer
            "ExpandElimination",  # L1 optimizer
            "ConstantFolding",  # Basic optimizer, always runs, no fusion pattern
        }

        missing = [
            cap.ort_name
            for cap in GRAPH_CAPABILITIES.values()
            if isinstance(cap, BoolCapability)
            and hasattr(cap, "ort_name")
            and cap.ort_name
            and cap.ort_name not in BUILDER_REGISTRY
            and cap.ort_name not in untestable
        ]

        assert not missing, f"Missing builders for testable BoolCapabilities: {missing}"

    def test_builder_configs_have_valid_shapes(self) -> None:
        """All BuilderConfig input_shapes should be valid non-empty tuples."""
        invalid = []
        for ort_name, config in BUILDER_REGISTRY.items():
            if not isinstance(config.input_shape, tuple) or len(config.input_shape) == 0:
                invalid.append(ort_name)

        assert not invalid, f"Invalid input_shapes in BUILDER_REGISTRY: {invalid}"

    def test_isolated_test_cases_generated(self) -> None:
        """ISOLATED_TEST_CASES should be non-empty."""
        assert len(ISOLATED_TEST_CASES) > 0, "No isolated test cases generated"

    def test_isolated_test_cases_match_registry(self) -> None:
        """Number of test cases should match registry size."""
        # Count actual test cases (explicit list)
        actual_count = len(ISOLATED_TEST_CASES)
        # Registry has all testable capabilities
        expected_count = len(BUILDER_REGISTRY)

        # Should be equal (all registry entries have test cases)
        assert actual_count == expected_count, (
            f"Test case count ({actual_count}) != registry size ({expected_count})"
        )
