# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for PatternExtractor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import onnx
import pytest
from onnx import TensorProto, helper

from tests.unit.test_helpers import stable_test_node_keys as _stable_test_node_keys
from winml.modelkit.analyze import ModelStats, ONNXModel, PatternExtractor
from winml.modelkit.pattern import PatternMatchResult, SkeletonMatchResult, SubgraphPattern


@pytest.fixture
def simple_model_proto() -> onnx.ModelProto:
    """Create a simple ONNX model proto for testing."""
    input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    # Create a simple graph with Conv and Relu
    conv_node = helper.make_node("Conv", ["input", "weight"], ["conv_out"], name="conv")
    relu_node = helper.make_node("Relu", ["conv_out"], ["output"], name="relu")

    graph_def = helper.make_graph([conv_node, relu_node], "test_graph", [input1], [output])
    return helper.make_model(
        graph_def, producer_name="test", opset_imports=[helper.make_opsetid("", 13)]
    )


@pytest.fixture
def simple_onnx_model(simple_model_proto: onnx.ModelProto) -> ONNXModel:
    """Create a simple ONNXModel for testing."""
    return ONNXModel.from_onnx_model(simple_model_proto, "test.onnx")


@pytest.fixture
def mock_subgraph_pattern() -> SubgraphPattern:
    """Create a mock SubgraphPattern for testing."""
    return SubgraphPattern(
        pattern_id="SUBGRAPH/TestPattern",
        pattern_name="TestPattern",
        operators=["Conv", "Relu"],
        node_topology={
            "conv_node": "Conv",
            "relu_node": "Relu",
        },
        edge_topology=[("conv_node", "relu_node")],
    )


class TestPatternExtractorInit:
    """Tests for PatternExtractor initialization."""

    def test_init_with_valid_model(self, simple_onnx_model: ONNXModel) -> None:
        """Test initialization with valid ONNXModel."""
        extractor = PatternExtractor(simple_onnx_model)
        assert extractor.model == simple_onnx_model

    def test_init_with_invalid_model_raises_type_error(self) -> None:
        """Test initialization with non-ONNXModel raises TypeError."""
        with pytest.raises(TypeError, match="Expected ONNXModel"):
            PatternExtractor("not_a_model")  # type: ignore[arg-type]

    def test_init_with_none_raises_type_error(self) -> None:
        """Test initialization with None raises TypeError."""
        with pytest.raises(TypeError, match="Expected ONNXModel"):
            PatternExtractor(None)  # type: ignore[arg-type]


class TestPatternExtractorModelProperty:
    """Tests for model property."""

    def test_model_property_returns_model(self, simple_onnx_model: ONNXModel) -> None:
        """Test model property returns the ONNXModel."""
        extractor = PatternExtractor(simple_onnx_model)
        assert extractor.model is simple_onnx_model


class TestPatternExtractorSummary:
    """Tests for summary method."""

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_summary_returns_dict_with_expected_keys(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test summary returns dict with 'summary' and 'subgraph_patterns' keys."""
        # Mock RuleLoader to return empty pattern list
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = []
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        result = extractor.summary()

        assert isinstance(result, dict)
        assert "summary" in result
        assert "subgraph_patterns" in result

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_summary_metadata_is_model_metadata(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test summary 'summary' key contains ModelStats."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = []
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        result = extractor.summary()

        assert isinstance(result["summary"], ModelStats)
        assert result["summary"].model_path == "test.onnx"

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_summary_subgraph_patterns_is_list(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test summary 'subgraph_patterns' key contains list."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = []
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        result = extractor.summary()

        assert isinstance(result["subgraph_patterns"], list)

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_summary_includes_detected_pattern_count(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test summary metadata includes correct detected_pattern_count."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = []
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        result = extractor.summary()

        # Since no patterns are matched, count should be empty dict
        assert result["summary"].detected_pattern_count == {}


class TestPatternExtractorExtractSubgraphPatterns:
    """Tests for extract_subgraph_patterns method."""

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_extract_with_no_patterns_returns_empty_list(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test extract_subgraph_patterns with no pattern definitions."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = []
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        patterns = extractor.extract_subgraph_patterns()

        assert patterns == []

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_extract_with_patterns_calls_match(
        self,
        mock_config_cls: MagicMock,
        simple_onnx_model: ONNXModel,
        mock_subgraph_pattern: SubgraphPattern,
    ) -> None:
        """Test extract_subgraph_patterns calls _match_subgraph_pattern_from_model_tags."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = [mock_subgraph_pattern]
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)

        # Patch _match_subgraph_pattern_from_model_tags to verify it's called
        with patch.object(
            extractor, "_match_subgraph_pattern_from_model_tags", return_value=[]
        ) as mock_match:
            patterns = extractor.extract_subgraph_patterns()

            mock_match.assert_called_once_with(mock_subgraph_pattern)
            assert patterns == []

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_extract_returns_matched_patterns(
        self,
        mock_config_cls: MagicMock,
        simple_onnx_model: ONNXModel,
        mock_subgraph_pattern: SubgraphPattern,
    ) -> None:
        """Test extract_subgraph_patterns returns matched patterns."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = [mock_subgraph_pattern]
        mock_config_cls.return_value = mock_config

        # Create a mock PatternMatchResult with proper NodeProto objects
        from onnx import helper

        # Create mock node protos
        conv_node = helper.make_node("Conv", ["input"], ["conv_out"], name="conv1")
        relu_node = helper.make_node("Relu", ["conv_out"], ["output"], name="relu1")

        # Create SkeletonMatchResult
        skeleton_result = SkeletonMatchResult(
            pattern=mock_subgraph_pattern,
            matched_nodes=[conv_node, relu_node],
            matched_node_keys=_stable_test_node_keys([conv_node, relu_node]),
            matcher=None,
        )

        mock_match = PatternMatchResult(
            skeleton_match_result=skeleton_result,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
        )

        extractor = PatternExtractor(simple_onnx_model)

        with patch.object(
            extractor, "_match_subgraph_pattern_from_model_tags", return_value=[mock_match]
        ) as _:
            patterns = extractor.extract_subgraph_patterns()

            assert len(patterns) == 1
            assert patterns[0] == mock_match


class TestPatternExtractorMatchSubgraphPatternFromModelTags:
    """Tests for _match_subgraph_pattern_from_model_tags method."""

    def test_match_returns_empty_list(
        self,
        simple_onnx_model: ONNXModel,
        mock_subgraph_pattern: SubgraphPattern,
    ) -> None:
        """Test _match_subgraph_pattern_from_model_tags returns empty list (mock implementation)."""
        extractor = PatternExtractor(simple_onnx_model)
        matches = extractor._match_subgraph_pattern_from_model_tags(mock_subgraph_pattern)

        # Current implementation is a mock that returns empty list
        assert matches == []


class TestPatternExtractorGetSubgraphPatterns:
    """Tests for get_subgraph_patterns method."""

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_get_patterns_calls_unified_config(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test get_subgraph_patterns calls UnifiedPatternConfig.get_htp_patterns."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = []
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        patterns = extractor.get_subgraph_patterns()

        mock_config.get_htp_patterns.assert_called_once()
        assert patterns == []

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_get_patterns_returns_loaded_patterns(
        self,
        mock_config_cls: MagicMock,
        simple_onnx_model: ONNXModel,
        mock_subgraph_pattern: SubgraphPattern,
    ) -> None:
        """Test get_subgraph_patterns returns patterns from UnifiedPatternConfig."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = [mock_subgraph_pattern]
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        patterns = extractor.get_subgraph_patterns()

        assert len(patterns) == 1
        assert patterns[0] == mock_subgraph_pattern

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_get_patterns_returns_empty_list_when_no_rules(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test get_subgraph_patterns returns empty list when no patterns found."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = []
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)
        patterns = extractor.get_subgraph_patterns()

        assert patterns == []


class TestPatternExtractorModelSummary:
    """Tests for model_summary method."""

    def test_model_summary_returns_metadata(self, simple_onnx_model: ONNXModel) -> None:
        """Test model_summary returns ModelStats."""
        extractor = PatternExtractor(simple_onnx_model)
        metadata = extractor.model_summary()

        assert isinstance(metadata, ModelStats)
        assert metadata.model_path == "test.onnx"
        assert metadata.opset_version == 13

    def test_model_summary_with_pattern_count(self, simple_onnx_model: ONNXModel) -> None:
        """Test model_summary includes detected_pattern_count."""
        extractor = PatternExtractor(simple_onnx_model)
        pattern_count_dict = {"SUBGRAPH/GELU_Erf": 5}
        metadata = extractor.model_summary(detected_pattern_count=pattern_count_dict)

        assert metadata.detected_pattern_count == pattern_count_dict

    def test_model_summary_default_pattern_count_is_zero(
        self, simple_onnx_model: ONNXModel
    ) -> None:
        """Test model_summary default detected_pattern_count is empty dict."""
        extractor = PatternExtractor(simple_onnx_model)
        metadata = extractor.model_summary()

        assert metadata.detected_pattern_count == {}

    def test_model_summary_includes_operator_counts(self, simple_onnx_model: ONNXModel) -> None:
        """Test model_summary includes operator statistics."""
        extractor = PatternExtractor(simple_onnx_model)
        metadata = extractor.model_summary()

        assert metadata.total_operators == 2
        assert metadata.unique_operator_types == 2
        assert "Conv" in metadata.operator_counts
        assert "Relu" in metadata.operator_counts


class TestPatternExtractorIntegration:
    """Integration tests for PatternExtractor."""

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_full_workflow(
        self,
        mock_config_cls: MagicMock,
        simple_onnx_model: ONNXModel,
        mock_subgraph_pattern: SubgraphPattern,
    ) -> None:
        """Test complete workflow from initialization to summary."""
        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = [mock_subgraph_pattern]
        mock_config_cls.return_value = mock_config

        # Initialize extractor
        extractor = PatternExtractor(simple_onnx_model)

        # Generate summary
        result = extractor.summary()

        # Verify result structure
        assert "summary" in result
        assert "subgraph_patterns" in result
        assert isinstance(result["summary"], ModelStats)
        assert isinstance(result["subgraph_patterns"], list)

        # Verify metadata
        assert result["summary"].model_path == "test.onnx"
        assert result["summary"].total_operators == 2

    @patch("winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig")
    def test_workflow_with_multiple_patterns(
        self, mock_config_cls: MagicMock, simple_onnx_model: ONNXModel
    ) -> None:
        """Test workflow with multiple pattern definitions."""
        pattern1 = SubgraphPattern(
            pattern_id="SUBGRAPH/Pattern1",
            pattern_name="Pattern1",
            operators=["Conv"],
            node_topology={"conv": "Conv"},
            edge_topology=[],
        )
        pattern2 = SubgraphPattern(
            pattern_id="SUBGRAPH/Pattern2",
            pattern_name="Pattern2",
            operators=["Relu"],
            node_topology={"relu": "Relu"},
            edge_topology=[],
        )

        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = [pattern1, pattern2]
        mock_config_cls.return_value = mock_config

        extractor = PatternExtractor(simple_onnx_model)

        # Extract patterns
        patterns = extractor.extract_subgraph_patterns()

        # Should process both patterns (even if no matches found)
        assert isinstance(patterns, list)

        # Verify both patterns were loaded
        loaded_patterns = extractor.get_subgraph_patterns()
        assert len(loaded_patterns) == 2
