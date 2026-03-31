# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for pattern matching on ONNX models."""

from pathlib import Path

import onnx
import onnx.helper as oh
import pytest

from winml.modelkit.pattern import (
    Gelu2Pattern,
    MatMulAddPattern,
    PatternMatcher,
    ReshapeGemmReshapePattern,
)


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"
ERF_CONVNEXT_MODEL_PATH = FIXTURES_DIR / "erf-convnext-tiny.onnx"
GELU_CONVNEXT_MODEL_PATH = FIXTURES_DIR / "gelu-convnext-tiny.onnx"


class TestErfConvNextPatternMatching:
    """Tests for pattern matching on erf-convnext-tiny.onnx model.

    This model uses Erf-based GELU activation pattern and MatMul+Add for linear layers.
    """

    @pytest.fixture
    def erf_convnext_model(self):
        """Load the Erf ConvNeXt model for testing."""
        if not ERF_CONVNEXT_MODEL_PATH.exists():
            pytest.skip(f"Test model not found: {ERF_CONVNEXT_MODEL_PATH}")
        return onnx.load(str(ERF_CONVNEXT_MODEL_PATH))

    def test_all_patterns_matching(self, erf_convnext_model):
        """Test that all 3 patterns are matched with expected counts.

        Expected counts:
        - Gelu pattern: 18 (Erf-based GELU activations)
        - MatMulAdd pattern: 36 (linear layers use MatMul+Add)
        - ReshapeGemmReshape pattern: 0 (this model uses MatMul+Add, not Gemm)
        """
        matcher = PatternMatcher(erf_convnext_model)

        gelu_pattern = Gelu2Pattern()
        matmuladd_pattern = MatMulAddPattern()
        reshape_gemm_reshape_pattern = ReshapeGemmReshapePattern()

        matcher.register_pattern(gelu_pattern)
        matcher.register_pattern(matmuladd_pattern)
        matcher.register_pattern(reshape_gemm_reshape_pattern)

        results = matcher.match()

        # Categorize results by pattern type
        gelu_results = [
            r for r in results if isinstance(r.skeleton_match_result.pattern, Gelu2Pattern)
        ]
        matmuladd_results = [
            r for r in results if isinstance(r.skeleton_match_result.pattern, MatMulAddPattern)
        ]
        reshape_gemm_results = [
            r
            for r in results
            if isinstance(r.skeleton_match_result.pattern, ReshapeGemmReshapePattern)
        ]

        # Verify expected counts
        assert len(gelu_results) == 18, f"Expected 18 Gelu matches, found {len(gelu_results)}"
        assert len(matmuladd_results) == 36, (
            f"Expected 36 MatMulAdd matches, found {len(matmuladd_results)}"
        )
        assert len(reshape_gemm_results) == 0, (
            f"Expected 0 ReshapeGemmReshape matches, found {len(reshape_gemm_results)}"
        )

        # Verify all matches are removable
        for r in results:
            assert r.skeleton_match_result.removable is True, (
                f"Expected removable=True for {r.skeleton_match_result.pattern.__class__.__name__}"
            )

    def test_gelu_pattern_match_structure(self, erf_convnext_model):
        """Test that GELU pattern matches have the expected structure."""
        matcher = PatternMatcher(erf_convnext_model)
        gelu_pattern = Gelu2Pattern()
        matcher.register_pattern(gelu_pattern)

        results = matcher.match()
        assert len(results) > 0, "Expected to find at least one GELU match"

        # Check structure of first match
        result = results[0]

        # Check schema mappings exist
        assert "X" in result.schema_input_to_value
        assert "Y" in result.schema_output_to_value

        # Check type mapping exists
        assert "T" in result.type_param_to_type
        assert result.type_param_to_type["T"].startswith("tensor(")

        # Check input infos
        assert "X" in result.input_infos
        x_info = result.input_infos["X"]
        assert x_info.shape is not None
        assert isinstance(x_info.is_constant, bool)

        # Check matched nodes (GELU has 5 nodes: Div, Erf, Add, Mul, Mul)
        assert len(result.skeleton_match_result.matched_nodes) == 5

        # Check removable flag
        assert result.skeleton_match_result.removable is True

    def test_matmuladd_pattern_match_structure(self, erf_convnext_model):
        """Test that MatMulAdd pattern matches have the expected structure."""
        matcher = PatternMatcher(erf_convnext_model)
        matmuladd_pattern = MatMulAddPattern()
        matcher.register_pattern(matmuladd_pattern)

        results = matcher.match()
        assert len(results) > 0, "Expected to find at least one MatMulAdd match"

        # Check structure of first match
        result = results[0]

        # Check schema mappings exist
        assert "A" in result.schema_input_to_value
        assert "B" in result.schema_input_to_value
        assert "C" in result.schema_input_to_value
        assert "Y" in result.schema_output_to_value

        # Check type mapping exists
        assert "T" in result.type_param_to_type

        # Check input infos
        assert "A" in result.input_infos
        assert "B" in result.input_infos
        assert "C" in result.input_infos

        # B (weights) and C (bias) should typically be constants in linear layers
        b_info = result.input_infos["B"]
        c_info = result.input_infos["C"]
        assert b_info.shape is not None
        assert c_info.shape is not None
        # B should be 2D, C should be 1D
        assert len(b_info.shape) == 2
        assert len(c_info.shape) == 1

        # Check matched nodes (MatMulAdd has 2 nodes: MatMul, Add)
        assert len(result.skeleton_match_result.matched_nodes) == 2

        # Check removable flag
        assert result.skeleton_match_result.removable is True


class TestGeluConvNextPatternMatching:
    """Tests for pattern matching on gelu-convnext-tiny.onnx model.

    This model uses Gemm operator directly (not MatMul+Add) wrapped with Reshape ops,
    and has built-in GELU nodes (not the Erf-based GELU pattern).
    """

    @pytest.fixture
    def gelu_convnext_model(self):
        """Load the GELU ConvNeXt model for testing."""
        if not GELU_CONVNEXT_MODEL_PATH.exists():
            pytest.skip(f"Test model not found: {GELU_CONVNEXT_MODEL_PATH}")
        return onnx.load(str(GELU_CONVNEXT_MODEL_PATH))

    def test_all_patterns_matching(self, gelu_convnext_model):
        """Test that all 3 patterns are matched with expected counts.

        Expected counts:
        - Gelu pattern: 0 (this model uses GELU op, not Erf-based pattern)
        - MatMulAdd pattern: 0 (this model uses Gemm, not MatMul+Add)
        - ReshapeGemmReshape pattern: 36 (linear layers use Reshape->Gemm->Reshape)
        """
        matcher = PatternMatcher(gelu_convnext_model)

        gelu_pattern = Gelu2Pattern()
        matmuladd_pattern = MatMulAddPattern()
        reshape_gemm_reshape_pattern = ReshapeGemmReshapePattern()

        matcher.register_pattern(gelu_pattern)
        matcher.register_pattern(matmuladd_pattern)
        matcher.register_pattern(reshape_gemm_reshape_pattern)

        results = matcher.match()

        # Categorize results by pattern type
        gelu_results = [
            r for r in results if isinstance(r.skeleton_match_result.pattern, Gelu2Pattern)
        ]
        matmuladd_results = [
            r for r in results if isinstance(r.skeleton_match_result.pattern, MatMulAddPattern)
        ]
        reshape_gemm_results = [
            r
            for r in results
            if isinstance(r.skeleton_match_result.pattern, ReshapeGemmReshapePattern)
        ]

        # Verify expected counts
        assert len(gelu_results) == 0, f"Expected 0 Gelu matches, found {len(gelu_results)}"
        assert len(matmuladd_results) == 0, (
            f"Expected 0 MatMulAdd matches, found {len(matmuladd_results)}"
        )
        assert len(reshape_gemm_results) == 36, (
            f"Expected 36 ReshapeGemmReshape matches, found {len(reshape_gemm_results)}"
        )

        # Verify all matches are removable
        for r in results:
            assert r.skeleton_match_result.removable is True, (
                f"Expected removable=True for {r.skeleton_match_result.pattern.__class__.__name__}"
            )

    def test_reshape_gemm_reshape_structure(self, gelu_convnext_model):
        """Test that ReshapeGemmReshape matches have the expected structure."""
        matcher = PatternMatcher(gelu_convnext_model)
        pattern = ReshapeGemmReshapePattern()
        matcher.register_pattern(pattern)

        results = matcher.match()
        assert len(results) > 0, "Expected to find ReshapeGemmReshape matches"

        # Check structure of first match
        result = results[0]

        # Check schema mappings exist (uses MatMulAdd schema)
        assert "A" in result.schema_input_to_value
        assert "B" in result.schema_input_to_value
        assert "C" in result.schema_input_to_value
        assert "Y" in result.schema_output_to_value

        # Check type mapping exists
        assert "T" in result.type_param_to_type

        # Check input infos
        assert "A" in result.input_infos
        assert "B" in result.input_infos
        assert "C" in result.input_infos

        # B should be 2D, C should be 1D
        b_info = result.input_infos["B"]
        c_info = result.input_infos["C"]
        assert b_info.shape is not None
        assert c_info.shape is not None
        assert len(b_info.shape) == 2, f"Expected B to be 2D, got {len(b_info.shape)}D"
        assert len(c_info.shape) == 1, f"Expected C to be 1D, got {len(c_info.shape)}D"

        # Check matched nodes (ReshapeGemmReshape has 3 nodes: Reshape, Gemm, Reshape)
        assert len(result.skeleton_match_result.matched_nodes) == 3

        # Check removable flag
        assert result.skeleton_match_result.removable is True


def _make_gelu_model_with_unnamed_nodes() -> onnx.ModelProto:
    """Build a minimal ONNX model with a Gelu2 pattern plus extra nodes with empty names.

    Graph: X -> Div -> Erf -> Add -> Mul -> Mul -> Y  (Gelu2Pattern, all named)
           X -> [unnamed Slice] -> [unnamed Slice] -> Z  (extra unnamed nodes)
    """
    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 8])  # noqa: N806
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 8])  # noqa: N806
    Z = oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, None)  # noqa: N806

    sqrt2 = oh.make_tensor("sqrt2", onnx.TensorProto.FLOAT, [], [1.4142135])
    half = oh.make_tensor("half", onnx.TensorProto.FLOAT, [], [0.5])
    one = oh.make_tensor("one_val", onnx.TensorProto.FLOAT, [], [1.0])
    axes = oh.make_tensor("axes_t", onnx.TensorProto.INT64, [1], [1])
    starts = oh.make_tensor("starts_t", onnx.TensorProto.INT64, [1], [0])
    ends = oh.make_tensor("ends_t", onnx.TensorProto.INT64, [1], [4])

    # Gelu2 pattern nodes (all named).
    # Topology: X/sqrt2 -> Erf -> +1 -> (X * result) * 0.5
    # Node 3: Mul(X, add_out);  Node 4: Mul(mul3_out, 0.5)
    div = oh.make_node("Div", ["X", "sqrt2"], ["div_out"], name="div_node")
    erf = oh.make_node("Erf", ["div_out"], ["erf_out"], name="erf_node")
    add = oh.make_node("Add", ["erf_out", "one_val"], ["add_out"], name="add_node")
    mul1 = oh.make_node("Mul", ["X", "add_out"], ["mul1_out"], name="mul1_node")
    mul2 = oh.make_node("Mul", ["mul1_out", "half"], ["Y"], name="mul2_node")

    # Extra unnamed Slice nodes (simulating real-world models with missing names)
    slice1 = oh.make_node("Slice", ["X", "starts_t", "ends_t", "axes_t"], ["slice1_out"], name="")
    slice2 = oh.make_node("Slice", ["slice1_out", "starts_t", "ends_t", "axes_t"], ["Z"], name="")

    graph = oh.make_graph(
        [div, erf, add, mul1, mul2, slice1, slice2],
        "test_graph",
        [X],
        [Y, Z],
        initializer=[sqrt2, half, one, axes, starts, ends],
    )
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])


class TestPatternMatchingWithUnnamedNodes:
    """Pattern matching should skip unnamed nodes, not fail entirely."""

    def test_unnamed_nodes_do_not_raise(self):
        """PatternMatcher must not raise for models with some empty-named nodes."""
        model = _make_gelu_model_with_unnamed_nodes()
        # Should not raise InvalidPatternMatcherModelError
        matcher = PatternMatcher(model)
        assert matcher is not None

    def test_named_patterns_still_matched_despite_unnamed_nodes(self):
        """Gelu2 pattern is matched even when the model contains unnamed Slice nodes."""
        model = _make_gelu_model_with_unnamed_nodes()
        matcher = PatternMatcher(model)
        matcher.register_pattern(Gelu2Pattern())
        results = matcher.match()
        assert len(results) == 1, f"Expected 1 Gelu2Pattern match, got {len(results)}"

    def test_raise_on_invalid_model_false_still_works(self):
        """raise_on_invalid_model=False still works after the workaround."""
        model = _make_gelu_model_with_unnamed_nodes()
        matcher = PatternMatcher(model, raise_on_invalid_model=False)
        matcher.register_pattern(Gelu2Pattern())
        results = matcher.match()
        assert len(results) == 1
