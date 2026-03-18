"""Unit tests for pattern matching on ONNX models."""

from pathlib import Path

import onnx
import pytest

from winml.modelkit.pattern import (
    Gelu2Pattern,
    MatMulAddPattern,
    PatternMatcher,
)
from winml.modelkit.pattern.gemm_patterns import ReshapeGemmReshapePattern


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
        assert (
            len(matmuladd_results) == 36
        ), f"Expected 36 MatMulAdd matches, found {len(matmuladd_results)}"
        assert (
            len(reshape_gemm_results) == 0
        ), f"Expected 0 ReshapeGemmReshape matches, found {len(reshape_gemm_results)}"

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
        assert (
            len(matmuladd_results) == 0
        ), f"Expected 0 MatMulAdd matches, found {len(matmuladd_results)}"
        assert (
            len(reshape_gemm_results) == 36
        ), f"Expected 36 ReshapeGemmReshape matches, found {len(reshape_gemm_results)}"

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
