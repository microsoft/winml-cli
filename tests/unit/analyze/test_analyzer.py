# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ONNXStaticAnalyzer."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import onnx
import pytest

from winml.modelkit.analyze import (
    Action,
    ActionItem,
    AnalysisOutput,
    AnalysisResult,
    AnalyzerConfig,
    EPSupport,
    IHVType,
    Information,
    ModelStats,
    ONNXStaticAnalyzer,
    SupportLevel,
)
from winml.modelkit.analyze.analyzer import (
    _build_runtime_debug_details_summary,
    _build_subgraph_runtime_results,
)
from winml.modelkit.analyze.models.runtime_checks import PatternRuntime, RuntimeTestResult
from winml.modelkit.optim import WinMLOptimizationConfig


def test_build_subgraph_runtime_results_preserves_selected_alternative_metadata() -> None:
    """Final selected alternatives retain action metadata for optimization config."""
    pattern_match = MagicMock()
    pattern_match.match_id = "match-1"
    action_items = [
        {
            "type": "GraphOptimization",
            "optimization_options": {"test_fusion": True},
        }
    ]
    merge_prep_entries = [
        {
            "pattern_id": "SUBGRAPH/SourcePattern",
            "match_id": "match-1",
            "support_status": "partial",
            "alternatives": [
                {
                    "pattern_to_id": "SUBGRAPH/SelectedAlternative",
                    "enabled": True,
                    "details": "Use the selected alternative.",
                    "reason": "The alternative is supported.",
                    "action_items": action_items,
                }
            ],
            "candidates": [
                {
                    "pattern_id": "SUBGRAPH/SelectedAlternative",
                    "is_alternative": True,
                    "status": "ok",
                    "compile": True,
                    "run": True,
                }
            ],
        }
    ]

    runtime_results = _build_subgraph_runtime_results(
        [pattern_match],
        merge_prep_entries,
    )

    assert len(runtime_results) == 1
    runtime_result = runtime_results[0]
    assert runtime_result.result.classification == SupportLevel.PARTIAL
    assert runtime_result.pattern_match is pattern_match
    assert len(runtime_result.alternatives) == 1
    selected_alternative = runtime_result.alternatives[0]
    assert selected_alternative.pattern_id == "SUBGRAPH/SelectedAlternative"
    assert selected_alternative.result.classification == SupportLevel.SUPPORTED
    assert selected_alternative.details == "Use the selected alternative."
    assert selected_alternative.action_items == action_items


class TestAnalyzerConfig:
    """Tests for AnalyzerConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AnalyzerConfig()
        assert config.enable_information is False
        assert config.pattern_detection_timeout == 300
        assert config.max_memory_mb == 2048
        assert config.rule_database_path is None

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = AnalyzerConfig(
            enable_information=True,
            pattern_detection_timeout=600,
            max_memory_mb=4096,
            rule_database_path="/custom/rules",
        )
        assert config.enable_information is True
        assert config.pattern_detection_timeout == 600
        assert config.max_memory_mb == 4096
        assert config.rule_database_path == "/custom/rules"


class TestAnalysisResult:
    """Tests for AnalysisResult wrapper."""

    @pytest.fixture
    def mock_output(self) -> AnalysisOutput:
        """Create mock AnalysisOutput."""
        metadata = ModelStats(
            model_path="test.onnx",
            opset_version=13,
            total_operators=10,
            operator_counts={"Conv": 5, "Relu": 5},
            unique_operator_types=2,
            detected_pattern_count={},
        )

        ihv_support = EPSupport(
            ihv_type=IHVType.QC,
            ep_type="QNNExecutionProvider",
            runtime_support=True,
            has_errors=False,
            has_warnings=False,
            classification={
                SupportLevel.SUPPORTED: ["Conv", "Relu"],
                SupportLevel.PARTIAL: [],
                SupportLevel.UNSUPPORTED: [],
                SupportLevel.UNKNOWN: [],
            },
            information=[],
        )

        return AnalysisOutput(
            metadata=metadata,
            results=[ihv_support],
        )

    def test_analysis_result_init(self, mock_output: AnalysisOutput) -> None:
        """Test AnalysisResult initialization."""
        result = AnalysisResult(output=mock_output)
        assert result.output == mock_output

    def test_repr(self, mock_output: AnalysisOutput) -> None:
        """Test string representation."""
        result = AnalysisResult(output=mock_output)
        assert repr(result) == "AnalysisResult(patterns_by_ep={})"

    def test_is_fully_supported_true(self, mock_output: AnalysisOutput) -> None:
        """Test is_fully_supported returns True when all ops are supported."""
        result = AnalysisResult(output=mock_output)
        assert result.is_fully_supported() is True
        assert result.is_fully_supported("QNNExecutionProvider") is True

    def test_is_fully_supported_false_with_unsupported_ops(
        self, mock_output: AnalysisOutput
    ) -> None:
        """Test is_fully_supported returns False when unsupported ops exist."""
        mock_output.results[0].runtime_support = False
        mock_output.results[0].classification[SupportLevel.UNSUPPORTED] = ["Upsample"]

        result = AnalysisResult(output=mock_output)
        assert result.is_fully_supported() is False

    def test_is_fully_supported_no_results(self) -> None:
        """Test is_fully_supported with no results."""
        metadata = ModelStats(
            model_path="test.onnx",
            opset_version=13,
            total_operators=0,
            operator_counts={},
            unique_operator_types=0,
            detected_pattern_count={},
        )
        output = AnalysisOutput(
            metadata=metadata,
            results=[],
        )
        result = AnalysisResult(output=output)
        assert result.is_fully_supported() is False

    def test_is_fully_supported_invalid_ep(self, mock_output: AnalysisOutput) -> None:
        """Test is_fully_supported with invalid EP name."""
        result = AnalysisResult(output=mock_output)
        assert result.is_fully_supported("InvalidEP") is False

    def test_has_errors_false(self, mock_output: AnalysisOutput) -> None:
        """Test has_errors returns False when no unsupported patterns exist."""
        result = AnalysisResult(output=mock_output)
        assert result.has_errors() is False
        assert result.has_errors("QNNExecutionProvider") is False

    def test_has_errors_true_with_unsupported(self, mock_output: AnalysisOutput) -> None:
        """Test has_errors returns True when unsupported patterns exist."""
        mock_output.results[0].classification[SupportLevel.UNSUPPORTED] = ["Upsample"]
        mock_output.results[0].has_errors = True

        result = AnalysisResult(output=mock_output)
        assert result.has_errors() is True
        assert result.has_errors("QNNExecutionProvider") is True

    def test_has_errors_no_results(self) -> None:
        """Test has_errors with no results."""
        metadata = ModelStats(
            model_path="test.onnx",
            opset_version=13,
            total_operators=0,
            operator_counts={},
            unique_operator_types=0,
            detected_pattern_count={},
        )
        output = AnalysisOutput(
            metadata=metadata,
            results=[],
        )
        result = AnalysisResult(output=output)
        assert result.has_errors() is False

    def test_has_errors_invalid_ep(self, mock_output: AnalysisOutput) -> None:
        """Test has_errors with invalid EP name."""
        result = AnalysisResult(output=mock_output)
        assert result.has_errors("InvalidEP") is False

    def test_has_warnings_false(self, mock_output: AnalysisOutput) -> None:
        """Test has_warnings returns False when no partial patterns exist."""
        result = AnalysisResult(output=mock_output)
        assert result.has_warnings() is False
        assert result.has_warnings("QNNExecutionProvider") is False

    def test_has_warnings_true_with_partial(self, mock_output: AnalysisOutput) -> None:
        """Test has_warnings returns True when partial patterns exist."""
        mock_output.results[0].classification[SupportLevel.PARTIAL] = ["Resize"]
        mock_output.results[0].has_warnings = True

        result = AnalysisResult(output=mock_output)
        assert result.has_warnings() is True
        assert result.has_warnings("QNNExecutionProvider") is True

    def test_has_warnings_no_results(self) -> None:
        """Test has_warnings with no results."""
        metadata = ModelStats(
            model_path="test.onnx",
            opset_version=13,
            total_operators=0,
            operator_counts={},
            unique_operator_types=0,
            detected_pattern_count={},
        )
        output = AnalysisOutput(
            metadata=metadata,
            results=[],
        )
        result = AnalysisResult(output=output)
        assert result.has_warnings() is False

    def test_has_warnings_invalid_ep(self, mock_output: AnalysisOutput) -> None:
        """Test has_warnings with invalid EP name."""
        result = AnalysisResult(output=mock_output)
        assert result.has_warnings("InvalidEP") is False

    def test_get_lint_result_all_supported(self, mock_output: AnalysisOutput) -> None:
        """Test get_lint_result with all supported patterns (no errors/warnings)."""
        result = AnalysisResult(output=mock_output)
        lint = result.get_lint_result()

        assert lint.errors == 0
        assert lint.warnings == 0
        assert lint.info == 0
        assert lint.passed is True
        assert lint.error_patterns == []
        assert lint.warning_patterns == []
        assert lint.information == []
        assert isinstance(lint.optimization_config, WinMLOptimizationConfig)

    def test_get_lint_result_with_errors(self, mock_output: AnalysisOutput) -> None:
        """Test get_lint_result with unsupported patterns (errors)."""
        mock_output.results[0].classification[SupportLevel.UNSUPPORTED] = ["Upsample", "NonZero"]
        mock_output.results[0].has_errors = True

        result = AnalysisResult(output=mock_output)
        lint = result.get_lint_result()

        assert lint.errors == 2
        assert lint.warnings == 0
        assert lint.info == 0
        assert lint.passed is False
        assert lint.error_patterns == ["Upsample", "NonZero"]
        assert lint.warning_patterns == []
        assert lint.information == []
        assert isinstance(lint.optimization_config, WinMLOptimizationConfig)

    def test_get_lint_result_with_warnings(self, mock_output: AnalysisOutput) -> None:
        """Test get_lint_result with partial patterns (warnings)."""
        mock_output.results[0].classification[SupportLevel.PARTIAL] = ["Resize", "Shape"]
        mock_output.results[0].has_warnings = True

        result = AnalysisResult(output=mock_output)
        lint = result.get_lint_result()

        assert lint.errors == 0
        assert lint.warnings == 2
        assert lint.info == 0
        assert lint.passed is False  # Passed is False when warnings exist
        assert lint.error_patterns == []
        assert lint.warning_patterns == ["Resize", "Shape"]
        assert lint.information == []
        assert isinstance(lint.optimization_config, WinMLOptimizationConfig)

    def test_get_lint_result_with_information(self, mock_output: AnalysisOutput) -> None:
        """Test get_lint_result with information items."""
        info1 = Information(
            action=None,
            explanation="Optimization opportunity 1",
            pattern_id="SUBGRAPH/GELU",
        )
        info2 = Information(
            action=None,
            explanation="Optimization opportunity 2",
            pattern_id="SUBGRAPH/LayerNorm",
        )
        mock_output.results[0].information = [info1, info2]

        result = AnalysisResult(output=mock_output)
        lint = result.get_lint_result()

        assert lint.errors == 0
        assert lint.warnings == 0
        assert lint.info == 2
        assert lint.passed is True
        assert lint.error_patterns == []
        assert lint.warning_patterns == []
        assert lint.information == [info1, info2]
        assert isinstance(lint.optimization_config, WinMLOptimizationConfig)

    def test_get_lint_result_comprehensive(self, mock_output: AnalysisOutput) -> None:
        """Test get_lint_result with errors, warnings, and info."""
        mock_output.results[0].classification[SupportLevel.UNSUPPORTED] = ["Upsample"]
        mock_output.results[0].classification[SupportLevel.PARTIAL] = ["Resize", "Shape"]
        mock_output.results[0].has_errors = True
        mock_output.results[0].has_warnings = True
        info1 = Information(
            action=None,
            explanation="Info 1",
            pattern_id="SUBGRAPH/GELU",
        )
        mock_output.results[0].information = [info1]

        result = AnalysisResult(output=mock_output)
        lint = result.get_lint_result()

        assert lint.errors == 1
        assert lint.warnings == 2
        assert lint.info == 1
        assert lint.passed is False
        assert lint.error_patterns == ["Upsample"]
        assert lint.warning_patterns == ["Resize", "Shape"]
        assert lint.information == [info1]
        assert isinstance(lint.optimization_config, WinMLOptimizationConfig)

    def test_get_lint_result_no_results(self) -> None:
        """Test get_lint_result with no results."""
        metadata = ModelStats(
            model_path="test.onnx",
            opset_version=13,
            total_operators=0,
            operator_counts={},
            unique_operator_types=0,
            detected_pattern_count={},
        )
        output = AnalysisOutput(
            metadata=metadata,
            results=[],
        )
        result = AnalysisResult(output=output)
        lint = result.get_lint_result()

        assert lint.errors == 0
        assert lint.warnings == 0
        assert lint.info == 0
        assert lint.passed is True
        assert lint.error_patterns == []
        assert lint.warning_patterns == []
        assert lint.information == []
        assert isinstance(lint.optimization_config, WinMLOptimizationConfig)

    def test_get_lint_result_filtered_by_ep(self, mock_output: AnalysisOutput) -> None:
        """Test get_lint_result filtered by EP."""
        # Add another EP with different patterns
        intel_support = EPSupport(
            ihv_type=IHVType.INTEL,
            ep_type="OpenVINOExecutionProvider",
            runtime_support=False,
            has_errors=True,
            has_warnings=False,
            classification={
                SupportLevel.SUPPORTED: [],
                SupportLevel.PARTIAL: [],
                SupportLevel.UNSUPPORTED: ["InstanceNorm"],
                SupportLevel.UNKNOWN: [],
            },
            information=[],
        )
        mock_output.results.append(intel_support)

        result = AnalysisResult(output=mock_output)

        # Get lint result for QNN only (no errors)
        lint_qnn = result.get_lint_result("QNNExecutionProvider")
        assert lint_qnn.errors == 0
        assert lint_qnn.passed is True
        assert lint_qnn.error_patterns == []
        assert isinstance(lint_qnn.optimization_config, WinMLOptimizationConfig)

        # Get lint result for Intel only (has errors)
        lint_intel = result.get_lint_result("OpenVINOExecutionProvider")
        assert lint_intel.errors == 1
        assert lint_intel.passed is False
        assert lint_intel.error_patterns == ["InstanceNorm"]
        assert isinstance(lint_intel.optimization_config, WinMLOptimizationConfig)

        # Get lint result for all EPs (aggregated)
        lint_all = result.get_lint_result()
        assert lint_all.errors == 1
        assert lint_all.passed is False
        assert "InstanceNorm" in lint_all.error_patterns
        assert isinstance(lint_all.optimization_config, WinMLOptimizationConfig)

    def test_get_unsupported_operators_empty(self, mock_output: AnalysisOutput) -> None:
        """Test get_unsupported_operators with all supported ops."""
        result = AnalysisResult(output=mock_output)
        unsupported = result.get_unsupported_operators()
        assert unsupported == []

    def test_get_unsupported_operators_with_unsupported_and_partial(
        self, mock_output: AnalysisOutput
    ) -> None:
        """Test get_unsupported_operators returns unsupported and partial ops."""
        mock_output.results[0].classification[SupportLevel.UNSUPPORTED] = ["Upsample"]
        mock_output.results[0].classification[SupportLevel.PARTIAL] = ["Resize"]

        result = AnalysisResult(output=mock_output)
        unsupported = result.get_unsupported_operators()
        assert "Resize" in unsupported
        assert "Upsample" in unsupported
        assert len(unsupported) == 2

    def test_get_unsupported_operators_filtered_by_ep(self, mock_output: AnalysisOutput) -> None:
        """Test get_unsupported_operators filtered by EP."""
        # Add another IHV with different ops
        intel_support = EPSupport(
            ihv_type=IHVType.INTEL,
            ep_type="OpenVINOExecutionProvider",
            runtime_support=False,
            has_errors=True,
            has_warnings=False,
            classification={
                SupportLevel.SUPPORTED: [],
                SupportLevel.PARTIAL: [],
                SupportLevel.UNSUPPORTED: ["Gelu"],
                SupportLevel.UNKNOWN: [],
            },
            information=[],
        )
        mock_output.results.append(intel_support)

        result = AnalysisResult(output=mock_output)

        # Get for QNN only
        unsupported_qnn = result.get_unsupported_operators("QNNExecutionProvider")
        assert unsupported_qnn == []

        # Get for OpenVINO only
        unsupported_intel = result.get_unsupported_operators("OpenVINOExecutionProvider")
        assert "Gelu" in unsupported_intel

        # Get for all EPs
        unsupported_all = result.get_unsupported_operators()
        assert "Gelu" in unsupported_all

    def test_to_json(self, mock_output: AnalysisOutput) -> None:
        """Test to_json exports valid JSON."""
        result = AnalysisResult(output=mock_output)
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "metadata" in json_str
        assert "results" in json_str

    def test_to_dict(self, mock_output: AnalysisOutput) -> None:
        """Test to_dict exports dictionary."""
        result = AnalysisResult(output=mock_output)
        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["metadata"]["opset_version"] == 13

    def test_get_optimization_config_no_actions(self, mock_output: AnalysisOutput) -> None:
        """Test get_optimization_config with no actions."""
        result = AnalysisResult(output=mock_output)
        config = result.get_optimization_config()

        assert isinstance(config, WinMLOptimizationConfig)
        assert config.get("gelu_fusion", False) is False
        assert config.get("layer_norm_fusion", False) is False
        assert config.get("matmul_add_fusion", False) is False
        assert config.get("attention_fusion", False) is False
        assert config.get("reshape_fusion", False) is False

    def test_get_optimization_config_with_gelu_pattern(self, mock_output: AnalysisOutput) -> None:
        """Test get_optimization_config detects GELU pattern."""
        # Add information with GELU action
        gelu_action = Action(
            pattern_from_id="SUBGRAPH/GeluPattern",
            pattern_to_id="OP/com.microsoft/Gelu",
            details="Replace GELU pattern with single operator",
            action_items=[
                ActionItem(
                    type="GraphOptimization",
                    optimization_options={"gelu_fusion": True},
                )
            ],
        )
        mock_output.results[0].information = [
            Information(
                pattern_id="SUBGRAPH/GeluPattern",
                explanation="GELU pattern detected",
                actions=[gelu_action],
            )
        ]

        result = AnalysisResult(output=mock_output)
        config = result.get_optimization_config()

        assert config.get("gelu_fusion", False) is True
        assert config.get("layer_norm_fusion", False) is False
        assert config.get("matmul_add_fusion", False) is False

    def test_get_optimization_config_with_multiple_patterns(
        self, mock_output: AnalysisOutput
    ) -> None:
        """Test get_optimization_config detects multiple patterns."""
        # Add multiple actions
        gelu_action = Action(
            pattern_from_id="SUBGRAPH/Gelu1",
            pattern_to_id="OP/com.microsoft/Gelu",
            details="Replace GELU pattern",
            action_items=[
                ActionItem(
                    type="GraphOptimization",
                    optimization_options={"gelu_fusion": True},
                )
            ],
        )
        layernorm_action = Action(
            pattern_from_id="SUBGRAPH/LayerNormalizationPattern",
            pattern_to_id="OP/ai.onnx/LayerNormalization",
            details="Replace LayerNorm pattern",
            action_items=[
                ActionItem(
                    type="GraphOptimization",
                    optimization_options={"layer_norm_fusion": True},
                )
            ],
        )
        gemm_action = Action(
            pattern_from_id="SUBGRAPH/GemmPattern",
            pattern_to_id="OP/ai.onnx/Gemm",
            details="Replace Gemm pattern",
            action_items=[
                ActionItem(
                    type="GraphOptimization",
                    optimization_options={"matmul_add_fusion": True},
                )
            ],
        )

        mock_output.results[0].information = [
            Information(
                pattern_id="SUBGRAPH/Gelu1",
                explanation="GELU detected",
                actions=[gelu_action],
            ),
            Information(
                pattern_id="SUBGRAPH/LayerNormalizationPattern",
                explanation="LayerNorm detected",
                actions=[layernorm_action],
            ),
            Information(
                pattern_id="SUBGRAPH/GemmPattern",
                explanation="Gemm detected",
                actions=[gemm_action],
            ),
        ]

        result = AnalysisResult(output=mock_output)
        config = result.get_optimization_config()

        assert config.get("gelu_fusion", False) is True
        assert config.get("layer_norm_fusion", False) is True
        assert config.get("matmul_add_fusion", False) is True
        assert config.get("attention_fusion", False) is False
        assert config.get("reshape_fusion", False) is False

    def test_get_optimization_config_filtered_by_ep(self, mock_output: AnalysisOutput) -> None:
        """Test get_optimization_config filtered by EP."""
        # Add Intel EP with different patterns
        intel_action = Action(
            pattern_from_id="SUBGRAPH/AttentionPattern",
            pattern_to_id="OP/com.microsoft/Attention",
            details="Replace Attention pattern",
            action_items=[
                ActionItem(
                    type="GraphOptimization",
                    optimization_options={"attention_fusion": True},
                )
            ],
        )
        intel_support = EPSupport(
            ihv_type=IHVType.INTEL,
            ep_type="OpenVINOExecutionProvider",
            runtime_support=True,
            has_errors=False,
            has_warnings=False,
            classification={
                SupportLevel.SUPPORTED: [],
                SupportLevel.PARTIAL: [],
                SupportLevel.UNSUPPORTED: [],
                SupportLevel.UNKNOWN: [],
            },
            information=[
                Information(
                    pattern_id="SUBGRAPH/AttentionPattern",
                    explanation="Attention detected",
                    actions=[intel_action],
                )
            ],
        )
        mock_output.results.append(intel_support)

        result = AnalysisResult(output=mock_output)

        # Get config for Intel only
        config = result.get_optimization_config(ep="OpenVINOExecutionProvider")
        assert config.get("attention_fusion", False) is True
        assert config.get("gelu_fusion", False) is False

    def test_get_optimization_config_underscore_format(self, mock_output: AnalysisOutput) -> None:
        """Test get_optimization_config handles underscore format keys."""
        # Test with underscore format like "matmul_add_fusion"
        matmul_action = Action(
            pattern_from_id="SUBGRAPH/MatMulAddPattern",
            pattern_to_id="OP/ai.onnx/Gemm",
            details="Fuse MatMul+Add to Gemm",
            action_items=[
                ActionItem(
                    type="GraphOptimization",
                    optimization_options={"matmul_add_fusion": True},
                )
            ],
        )
        mock_output.results[0].information = [
            Information(
                pattern_id="SUBGRAPH/MatMulAddPattern",
                explanation="MatMul+Add pattern detected",
                actions=[matmul_action],
            )
        ]

        result = AnalysisResult(output=mock_output)
        config = result.get_optimization_config()

        # Should correctly detect underscore format
        assert config.get("matmul_add_fusion", False) is True
        assert config.get("gelu_fusion", False) is False
        assert config.get("layer_norm_fusion", False) is False

    def test_get_optimization_config_custom_option(self, mock_output: AnalysisOutput) -> None:
        """Test get_optimization_config accepts custom optimization options."""
        # Test with custom optimization option (any key is allowed)
        custom_action = Action(
            pattern_from_id="SUBGRAPH/CustomPattern",
            pattern_to_id="OP/Custom",
            details="Custom optimization",
            action_items=[
                ActionItem(
                    type="GraphOptimization",
                    optimization_options={"custom_fusion": True},
                )
            ],
        )
        mock_output.results[0].information = [
            Information(
                pattern_id="SUBGRAPH/CustomPattern",
                explanation="Custom pattern",
                actions=[custom_action],
            )
        ]

        result = AnalysisResult(output=mock_output)
        config = result.get_optimization_config()

        # Should accept any custom option
        assert config.get("custom_fusion", False) is True

    def test_get_optimization_config_normalizes_kebab_case(
        self, mock_output: AnalysisOutput
    ) -> None:
        """Test get_optimization_config normalizes kebab-case keys to snake_case."""
        rtr_action = Action(
            pattern_from_id="SUBGRAPH/ReshapeTransposeReshapeOverlyHighDimPattern",
            pattern_to_id="SUBGRAPH/ReshapeTransposeReshapeLowDimPattern",
            details="RTR optimization",
            action_items=[
                ActionItem(
                    type="GraphOptimization",
                    optimization_options={"highdimRTR-lowdimRTR": True},
                )
            ],
        )
        mock_output.results[0].information = [
            Information(
                pattern_id="SUBGRAPH/ReshapeTransposeReshapeOverlyHighDimPattern",
                explanation="RTR pattern detected",
                actions=[rtr_action],
            )
        ]

        result = AnalysisResult(output=mock_output)
        config = result.get_optimization_config()

        # Kebab-case key should be normalized to underscore
        assert config.get("highdimRTR_lowdimRTR", False) is True
        # Original kebab-case key should NOT be present
        assert "highdimRTR-lowdimRTR" not in config

    def test_get_optimization_config_mixed_kebab_and_snake(
        self, mock_output: AnalysisOutput
    ) -> None:
        """Test get_optimization_config handles mix of kebab-case and snake_case keys."""
        action = Action(
            pattern_from_id="SUBGRAPH/TestPattern",
            pattern_to_id="OP/Test",
            details="Mixed key test",
            action_items=[
                ActionItem(
                    type="GraphOptimization",
                    optimization_options={
                        "already_snake": True,
                        "kebab-style-key": True,
                    },
                )
            ],
        )
        mock_output.results[0].information = [
            Information(
                pattern_id="SUBGRAPH/TestPattern",
                explanation="Test",
                actions=[action],
            )
        ]

        result = AnalysisResult(output=mock_output)
        config = result.get_optimization_config()

        assert config.get("already_snake", False) is True
        assert config.get("kebab_style_key", False) is True


class TestRuntimeDebugDetailsSummary:
    """Tests for runtime debug_details summary aggregation."""

    def test_build_runtime_debug_details_summary_groups_and_records_unknown(self) -> None:
        """Should group by support level and record unknown nodes as a key list."""
        runtime_summary = {
            "op_runtime_check_result": [
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Conv",
                    result=RuntimeTestResult(
                        compile=True,
                        run=True,
                        debug_details={
                            "node_stable_key": "node_conv",
                            "case_indices": ("case_1", "case_2"),
                            "table_path": "rules/conv.parquet",
                            "table_file": "conv.parquet",
                            "match_status": "op_match",
                        },
                    ),
                ),
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Resize",
                    result=RuntimeTestResult(
                        compile=False,
                        run=True,
                        debug_details={
                            "node_stable_key": "node_resize",
                            "case_indices": ["case_3"],
                            "table_path": "rules/resize.parquet",
                            "table_file": "resize.parquet",
                            "match_status": "op_match",
                        },
                    ),
                ),
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Unknown",
                    result=RuntimeTestResult(
                        compile=True,
                        run=True,
                        no_data=True,
                        debug_details={
                            "node_stable_key": "node_unknown",
                            "case_indices": ["case_4"],
                            "table_path": "rules/unknown.parquet",
                            "table_file": "unknown.parquet",
                            "match_status": "op_match",
                        },
                    ),
                ),
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Unsupported",
                    result=RuntimeTestResult(
                        compile=False,
                        run=False,
                        debug_details={
                            "node_stable_key": "node_unsupported",
                            "case_indices": ["case_5"],
                            "table_path": "rules/unsupported.parquet",
                            "table_file": "unsupported.parquet",
                            "match_status": "pattern_match",
                        },
                    ),
                )
            ],
        }

        summary = _build_runtime_debug_details_summary(runtime_summary)

        assert summary is not None
        assert set(summary.keys()) == {"unknown", "supported", "partial", "unsupported"}
        # "unknown" must be the first key in output order.
        assert next(iter(summary)) == "unknown"

        assert summary["supported"]["node_conv"].case_indices == ["case_1", "case_2"]
        assert summary["supported"]["node_conv"].table_path == "rules/conv.parquet"
        assert summary["supported"]["node_conv"].table_file == "conv.parquet"
        assert summary["supported"]["node_conv"].match_status == "op_match"

        assert summary["partial"]["node_resize"].case_indices == ["case_3"]
        assert summary["partial"]["node_resize"].table_path == "rules/resize.parquet"
        assert summary["partial"]["node_resize"].table_file == "resize.parquet"
        assert summary["partial"]["node_resize"].match_status == "op_match"

        assert summary["unsupported"]["node_unsupported"].case_indices == ["case_5"]
        assert summary["unsupported"]["node_unsupported"].table_path == "rules/unsupported.parquet"
        assert summary["unsupported"]["node_unsupported"].table_file == "unsupported.parquet"
        assert summary["unsupported"]["node_unsupported"].match_status == "pattern_match"

        # Unknown nodes are recorded as a plain list of node keys (no case data).
        assert summary["unknown"] == ["node_unknown"]
        assert "node_unknown" not in summary["supported"]
        assert "node_unknown" not in summary["partial"]
        assert "node_unknown" not in summary["unsupported"]

    def test_build_runtime_debug_details_summary_merges_same_node(self) -> None:
        """Should merge complementary debug fields for the same node key."""
        runtime_summary = {
            "op_runtime_check_result": [
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Conv",
                    result=RuntimeTestResult(
                        compile=True,
                        run=True,
                        debug_details={
                            "node_stable_key": "node_conv",
                            "table_path": "rules/conv.parquet",
                            "match_status": "op_match",
                        },
                    ),
                ),
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Conv",
                    result=RuntimeTestResult(
                        compile=True,
                        run=True,
                        debug_details={
                            "node_stable_key": "node_conv",
                            "case_indices": ("case_42",),
                            "table_file": "conv.parquet",
                            "match_status": "pattern_match",
                        },
                    ),
                ),
            ],
        }

        summary = _build_runtime_debug_details_summary(runtime_summary)

        assert summary is not None
        node_entry = summary["supported"]["node_conv"]
        assert node_entry.table_path == "rules/conv.parquet"
        assert node_entry.table_file == "conv.parquet"
        assert node_entry.case_indices == ["case_42"]
        assert node_entry.match_status == "pattern_match"


class TestONNXStaticAnalyzer:
    """Tests for ONNXStaticAnalyzer."""

    def test_init_default_config(self) -> None:
        """Test analyzer initialization with default config."""
        analyzer = ONNXStaticAnalyzer()
        assert analyzer.config is not None
        assert analyzer.config.enable_information is False

    def test_init_custom_config(self) -> None:
        """Test analyzer initialization with custom config."""
        config = AnalyzerConfig(enable_information=True, max_memory_mb=4096)
        analyzer = ONNXStaticAnalyzer(config=config)
        assert analyzer.config.enable_information is True
        assert analyzer.config.max_memory_mb == 4096

    def test_analyze_file_not_found(self) -> None:
        """Test analyze with non-existent file."""
        analyzer = ONNXStaticAnalyzer()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            analyzer.analyze("nonexistent.onnx", ep="QNNExecutionProvider", device="NPU")

    @patch("winml.modelkit.analyze.analyzer.Path.exists")
    @patch("onnx.load")
    @patch("onnx.checker.check_model")
    def test_analyze_invalid_onnx(
        self,
        mock_check_model: Mock,
        mock_load: Mock,
        mock_exists: Mock,
    ) -> None:
        """Test analyze with invalid ONNX file."""
        mock_exists.return_value = True
        mock_load.side_effect = OSError("Invalid ONNX file")

        analyzer = ONNXStaticAnalyzer()
        with pytest.raises(RuntimeError, match="Failed to load ONNX model"):
            analyzer.analyze("invalid.onnx", ep="QNNExecutionProvider", device="NPU")

    @patch("winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep", return_value=True)
    @patch("winml.modelkit.analyze.core.onnx_loader.ONNXLoader")
    @patch("winml.modelkit.analyze.core.pattern_extractor.PatternExtractor")
    @patch("winml.modelkit.analyze.core.runtime_checker.RuntimeChecker")
    def test_analyze_from_proto_single_ep(
        self,
        mock_runtime_checker_cls: Mock,
        mock_pattern_extractor_cls: Mock,
        mock_onnx_loader_cls: Mock,
        _mock_has_rule: Mock,
    ) -> None:
        """Test analyze_from_proto with single EP."""
        # Setup mocks
        mock_model = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_model
        mock_onnx_loader_cls.return_value = mock_loader

        mock_extractor = MagicMock()
        mock_extractor.summary.return_value = {
            "summary": ModelStats(
                model_path="test.onnx",
                opset_version=13,
                total_operators=10,
                operator_counts={"Conv": 10},
                unique_operator_types=1,
                detected_pattern_count={},
            ),
            "subgraph_patterns": [],
        }
        mock_pattern_extractor_cls.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker.summary.return_value = {
            "op_runtime_check_result": [],
        }
        mock_runtime_checker_cls.return_value = mock_checker

        # Create analyzer
        analyzer = ONNXStaticAnalyzer()
        mock_information_engine_cls = MagicMock()
        mock_information_engine_cls.return_value.summary.return_value = []
        analyzer.information_engine_cls = mock_information_engine_cls
        subgraph_runtime_results = [
            PatternRuntime(
                pattern_id="SUBGRAPH/SelectedPattern",
                result=RuntimeTestResult(compile=True, run=True),
            )
        ]

        # Mock model proto
        model_proto = MagicMock(spec=onnx.ModelProto)

        # Analyze
        with patch(
            "winml.modelkit.analyze.analyzer._build_subgraph_runtime_results",
            return_value=subgraph_runtime_results,
        ) as mock_build_subgraph_runtime_results:
            result = analyzer.analyze_from_proto(
                model_proto=model_proto,
                ep="QNNExecutionProvider",
                device="NPU",
                enable_information=True,
            )

        # Assertions
        assert isinstance(result, AnalysisResult)
        assert len(result.output.results) == 1
        assert result.output.results[0].ihv_type == IHVType.QC

        # Verify RuntimeChecker was called once
        assert mock_runtime_checker_cls.call_count == 1
        mock_build_subgraph_runtime_results.assert_called_once_with([], [])
        assert (
            mock_information_engine_cls.call_args.kwargs["subgraph_runtime_results"]
            is subgraph_runtime_results
        )

    @patch(
        "winml.modelkit.analyze.analyzer._build_pattern_status_by_node_key",
        return_value={"node-1": "supported"},
    )
    @patch("winml.modelkit.analyze.core.onnx_loader.ONNXLoader")
    @patch("winml.modelkit.analyze.core.pattern_extractor.PatternExtractor")
    @patch("winml.modelkit.analyze.core.runtime_checker.RuntimeChecker")
    def test_run_unknown_op_without_pattern_parquet_checks_pattern_then_ops(
        self,
        mock_runtime_checker_cls: Mock,
        mock_pattern_extractor_cls: Mock,
        mock_onnx_loader_cls: Mock,
        _mock_pattern_status: Mock,
    ) -> None:
        """Local probing checks matched patterns before remaining operators."""
        mock_model = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_model
        mock_onnx_loader_cls.return_value = mock_loader

        pattern_match = MagicMock()
        mock_extractor = MagicMock()

        def summary_with_local_pattern_check(**kwargs: object) -> dict[str, object]:
            local_pattern_checker = kwargs["local_pattern_checker"]
            assert callable(local_pattern_checker)
            local_pattern_checker(pattern_match, "table_not_found", False)
            return {
                "summary": ModelStats(
                    model_path="test.onnx",
                    opset_version=13,
                    total_operators=1,
                    operator_counts={"Conv": 1},
                    unique_operator_types=1,
                    detected_pattern_count={
                        "QNNExecutionProvider": {"SUBGRAPH/Test": 1}
                    },
                ),
                "subgraph_patterns": [pattern_match],
                "parquet_lookup_supported": False,
                "pattern_optimization_hints": [],
            }

        mock_extractor.summary.side_effect = summary_with_local_pattern_check
        mock_pattern_extractor_cls.return_value = mock_extractor

        pattern_checker = MagicMock()
        pattern_checker.check_pattern_locally.return_value = RuntimeTestResult(
            compile=True,
            run=True,
        )
        op_checker = MagicMock()
        op_checker.summary.return_value = {"op_runtime_check_result": []}
        mock_runtime_checker_cls.side_effect = [pattern_checker, op_checker]
        on_ep_start = MagicMock()

        result = ONNXStaticAnalyzer().analyze_from_proto(
            model_proto=MagicMock(spec=onnx.ModelProto),
            ep="QNNExecutionProvider",
            device="NPU",
            enable_information=False,
            run_unknown_op=True,
            on_ep_start=on_ep_start,
        )

        assert isinstance(result, AnalysisResult)
        assert mock_runtime_checker_cls.call_count == 2
        assert mock_runtime_checker_cls.call_args_list[0].kwargs == {
            "ep": "QNNExecutionProvider",
            "device": "NPU",
            "model": mock_model,
        }
        assert mock_runtime_checker_cls.call_args_list[1].kwargs == {
            "ep": "QNNExecutionProvider",
            "device": "NPU",
            "model": mock_model,
            "pattern_matched_node_status_by_key": {"node-1": "supported"},
        }
        pattern_checker.check_pattern_locally.assert_called_once_with(
            pattern_match,
            fallback_reason="table_not_found",
            for_debug=False,
        )
        pattern_checker.close_local_checks.assert_called_once_with()
        on_ep_start.assert_called_once()
        assert on_ep_start.call_args.args[2] is False
        op_checker.summary.assert_called_once_with(
            for_debug=False,
            run_unknown_op=True,
            save_node_types=None,
            on_node_result=None,
        )
        op_checker.close_local_checks.assert_called_once_with()

    def test_analyze_from_proto_resolves_auto_device_for_pinned_ep(self) -> None:
        from winml.modelkit.session import EPDeviceTarget

        resolved = EPDeviceTarget(ep="OpenVINOExecutionProvider", device="gpu")
        with (
            patch(
                "winml.modelkit.session.resolve_device",
                return_value=resolved,
            ) as mock_resolve,
            patch(
                "winml.modelkit.session.auto_detect_device",
                return_value="npu",
            ) as mock_global_auto,
            patch(
                "winml.modelkit.analyze.core.onnx_loader.ONNXLoader",
                side_effect=RuntimeError("stop after target resolution"),
            ),
            pytest.raises(RuntimeError, match="stop after target resolution"),
        ):
            ONNXStaticAnalyzer().analyze_from_proto(
                model_proto=MagicMock(spec=onnx.ModelProto),
                ep="openvino",
                device="auto",
                enable_information=False,
            )

        mock_resolve.assert_called_once_with(
            EPDeviceTarget(ep="OpenVINOExecutionProvider", device="auto")
        )
        mock_global_auto.assert_not_called()

    def test_analyze_from_proto_keeps_unpinned_auto_device_offline(self) -> None:
        from winml.modelkit.session import EPDeviceTarget

        resolved = EPDeviceTarget(ep="CPUExecutionProvider", device="cpu")
        with (
            patch(
                "winml.modelkit.session.resolve_device",
                return_value=resolved,
            ) as mock_resolve,
            patch(
                "winml.modelkit.session.auto_detect_device",
                return_value="cpu",
            ) as mock_global_auto,
            patch(
                "winml.modelkit.analyze.core.onnx_loader.ONNXLoader",
                side_effect=RuntimeError("stop after target resolution"),
            ),
            pytest.raises(RuntimeError, match="stop after target resolution"),
        ):
            ONNXStaticAnalyzer().analyze_from_proto(
                model_proto=MagicMock(spec=onnx.ModelProto),
                ep=None,
                device="auto",
                enable_information=False,
                run_unknown_op=False,
            )

        mock_global_auto.assert_called_once_with()
        mock_resolve.assert_not_called()

    @patch("winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep", return_value=True)
    @patch("winml.modelkit.analyze.core.onnx_loader.ONNXLoader")
    @patch("winml.modelkit.analyze.core.pattern_extractor.PatternExtractor")
    @patch("winml.modelkit.analyze.core.runtime_checker.RuntimeChecker")
    def test_analyze_from_proto_includes_runtime_debug_summary_when_debug_enabled(
        self,
        mock_runtime_checker_cls: Mock,
        mock_pattern_extractor_cls: Mock,
        mock_onnx_loader_cls: Mock,
        _mock_has_rule: Mock,
    ) -> None:
        """for_debug=True should add runtime_debug_details_summary to EP output."""
        mock_model = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_model
        mock_onnx_loader_cls.return_value = mock_loader

        mock_extractor = MagicMock()
        mock_extractor.summary.return_value = {
            "summary": ModelStats(
                model_path="test.onnx",
                opset_version=13,
                total_operators=2,
                operator_counts={"Conv": 1, "Relu": 1},
                unique_operator_types=2,
                detected_pattern_count={},
            ),
            "subgraph_patterns": [],
        }
        mock_pattern_extractor_cls.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker.summary.return_value = {
            "op_runtime_check_result": [
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Conv",
                    result=RuntimeTestResult(
                        compile=True,
                        run=True,
                        debug_details={
                            "node_stable_key": "node_conv",
                            "case_indices": ("case_7",),
                            "table_path": "rules/conv.parquet",
                            "table_file": "conv.parquet",
                            "match_status": "op_match",
                        },
                    ),
                ),
                PatternRuntime(
                    pattern_id="OP/ai.onnx/Relu",
                    result=RuntimeTestResult(
                        compile=True,
                        run=True,
                        no_data=True,
                        debug_details={
                            "node_stable_key": "node_unknown",
                            "case_indices": ["case_9"],
                            "table_path": "rules/relu.parquet",
                            "table_file": "relu.parquet",
                            "match_status": "pattern_match",
                        },
                    ),
                ),
            ],
        }
        mock_runtime_checker_cls.return_value = mock_checker

        analyzer = ONNXStaticAnalyzer()
        model_proto = MagicMock(spec=onnx.ModelProto)

        result = analyzer.analyze_from_proto(
            model_proto=model_proto,
            ep="QNNExecutionProvider",
            device="NPU",
            enable_information=False,
            for_debug=True,
        )

        assert isinstance(result, AnalysisResult)
        assert len(result.output.results) == 1

        ep_result = result.output.results[0]
        assert ep_result.runtime_debug_details_summary is not None
        node_conv_entry = ep_result.runtime_debug_details_summary["supported"]["node_conv"]
        assert node_conv_entry.case_indices == ["case_7"]
        assert node_conv_entry.table_path == "rules/conv.parquet"
        assert node_conv_entry.table_file == "conv.parquet"
        assert node_conv_entry.match_status == "op_match"
        assert ep_result.runtime_debug_details_summary["partial"] == {}
        assert ep_result.runtime_debug_details_summary["unsupported"] == {}
        assert ep_result.runtime_debug_details_summary["unknown"] == ["node_unknown"]
        assert "node_unknown" not in ep_result.runtime_debug_details_summary["supported"]

    @patch("winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep", return_value=True)
    @patch("winml.modelkit.analyze.core.onnx_loader.ONNXLoader")
    @patch("winml.modelkit.analyze.core.pattern_extractor.PatternExtractor")
    @patch("winml.modelkit.analyze.core.runtime_checker.RuntimeChecker")
    def test_analyze_from_proto_multi_ep(
        self,
        mock_runtime_checker_cls: Mock,
        mock_pattern_extractor_cls: Mock,
        mock_onnx_loader_cls: Mock,
        _mock_has_rule: Mock,
    ) -> None:
        """Test analyze_from_proto with multiple EPs (ep=None)."""
        # Setup mocks
        mock_model = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_model
        mock_onnx_loader_cls.return_value = mock_loader

        mock_extractor = MagicMock()
        pattern_counts_by_ep = {
            "QNNExecutionProvider": {"SUBGRAPH/GELU": 2},
            "OpenVINOExecutionProvider": {"SUBGRAPH/GELU": 1},
            "VitisAIExecutionProvider": {"SUBGRAPH/LayerNorm": 3},
        }

        def summary_for_ep(*, ep: str, **_kwargs: object) -> dict[str, object]:
            return {
                "summary": ModelStats(
                    model_path="test.onnx",
                    opset_version=13,
                    total_operators=10,
                    operator_counts={"Conv": 10},
                    unique_operator_types=1,
                    detected_pattern_count={ep: pattern_counts_by_ep[ep]},
                ),
                "subgraph_patterns": [],
            }

        mock_extractor.summary.side_effect = summary_for_ep
        mock_pattern_extractor_cls.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker.summary.return_value = {
            "op_runtime_check_result": [],
        }
        mock_runtime_checker_cls.return_value = mock_checker

        # Create analyzer
        analyzer = ONNXStaticAnalyzer()

        # Mock model proto
        model_proto = MagicMock(spec=onnx.ModelProto)

        # Analyze with ep=None (all EPs)
        result = analyzer.analyze_from_proto(
            model_proto=model_proto,
            ep=None,
            device="NPU",
            enable_information=False,
        )

        # Assertions
        assert isinstance(result, AnalysisResult)
        # Should have results for all NPU-capable EPs: QNN, OpenVINO, VitisAI
        # (NvTensorRTRTX only supports GPU, so it's excluded for device=NPU)
        assert len(result.output.results) == 3

        ep_types = {r.ep_type for r in result.output.results}
        assert "QNNExecutionProvider" in ep_types
        assert "OpenVINOExecutionProvider" in ep_types
        assert "VitisAIExecutionProvider" in ep_types

        assert result.output.metadata.detected_pattern_count == pattern_counts_by_ep
        assert (
            sum(
                result.output.metadata.detected_pattern_count[
                    "QNNExecutionProvider"
                ].values()
            )
            == 2
        )

        # Verify RuntimeChecker was called 3 times (once per NPU-capable EP)
        assert mock_runtime_checker_cls.call_count == 3

    @patch("winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep", return_value=True)
    @patch("winml.modelkit.analyze.core.onnx_loader.ONNXLoader")
    @patch("winml.modelkit.analyze.core.pattern_extractor.PatternExtractor")
    @patch("winml.modelkit.analyze.core.runtime_checker.RuntimeChecker")
    def test_analyze_from_proto_default_driver(
        self,
        mock_runtime_checker_cls: Mock,
        mock_pattern_extractor_cls: Mock,
        mock_onnx_loader_cls: Mock,
        _mock_has_rule: Mock,
    ) -> None:
        """Test analyze_from_proto uses NPU as default driver."""
        # Setup mocks
        mock_model = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_model
        mock_onnx_loader_cls.return_value = mock_loader

        mock_extractor = MagicMock()
        mock_extractor.summary.return_value = {
            "summary": ModelStats(
                model_path="test.onnx",
                opset_version=13,
                total_operators=10,
                operator_counts={"Conv": 10},
                unique_operator_types=1,
                detected_pattern_count={},
            ),
            "subgraph_patterns": [],
        }
        mock_pattern_extractor_cls.return_value = mock_extractor

        mock_checker = MagicMock()
        mock_checker.summary.return_value = {
            "op_runtime_check_result": [],
        }
        mock_runtime_checker_cls.return_value = mock_checker

        # Create analyzer
        analyzer = ONNXStaticAnalyzer()

        # Mock model proto
        model_proto = MagicMock(spec=onnx.ModelProto)

        # Analyze with device=None
        analyzer.analyze_from_proto(
            model_proto=model_proto,
            ep="QNNExecutionProvider",
            device=None,  # Should default to NPU
            enable_information=False,
        )

        # Verify RuntimeChecker was called with driver_version="NPU"
        call_args = mock_runtime_checker_cls.call_args
        assert call_args.kwargs["device"] == "NPU"

    @patch("winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep", return_value=True)
    @patch("winml.modelkit.analyze.core.onnx_loader.ONNXLoader")
    @patch("winml.modelkit.analyze.core.pattern_extractor.PatternExtractor")
    @patch("winml.modelkit.analyze.core.runtime_checker.RuntimeChecker")
    @patch("winml.modelkit.analyze.core.information_engine.InformationEngine")
    def test_analyze_from_proto_with_information(
        self,
        mock_info_engine_cls: Mock,
        mock_runtime_checker_cls: Mock,
        mock_pattern_extractor_cls: Mock,
        mock_onnx_loader_cls: Mock,
        _mock_has_rule: Mock,
    ) -> None:
        """Test analyze_from_proto with information enabled."""
        # Setup mocks
        mock_model = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_model
        mock_onnx_loader_cls.return_value = mock_loader

        mock_extractor = MagicMock()
        mock_extractor.summary.return_value = {
            "summary": ModelStats(
                model_path="test.onnx",
                opset_version=13,
                total_operators=10,
                operator_counts={"Conv": 10},
                unique_operator_types=1,
                detected_pattern_count={},
            ),
            "subgraph_patterns": [],
        }
        mock_pattern_extractor_cls.return_value = mock_extractor

        mock_checker = MagicMock()
        # Mock PatternRuntime with proper structure
        mock_pattern_runtime = MagicMock()
        mock_pattern_runtime.pattern_id = "OP/Conv"
        mock_pattern_runtime.result.classification = SupportLevel.SUPPORTED

        mock_checker.summary.return_value = {
            "op_runtime_check_result": [mock_pattern_runtime],  # Non-empty
        }
        mock_runtime_checker_cls.return_value = mock_checker

        mock_engine = MagicMock()
        # Create a proper Information object instead of MagicMock
        info = Information(
            explanation="Test recommendation",
            pattern_id="OP/Conv",
        )
        mock_engine.summary.return_value = [info]
        mock_info_engine_cls.return_value = mock_engine

        # Create analyzer
        analyzer = ONNXStaticAnalyzer()

        # Mock model proto
        model_proto = MagicMock(spec=onnx.ModelProto)

        # Analyze with information enabled
        result = analyzer.analyze_from_proto(
            model_proto=model_proto,
            ep="QNNExecutionProvider",
            device="NPU",
            enable_information=True,
        )

        # Assertions
        assert isinstance(result, AnalysisResult)

        # Verify InformationEngine was instantiated
        assert mock_info_engine_cls.called

    @patch("winml.modelkit.analyze.core.runtime_checker.RuntimeChecker")
    @patch("winml.modelkit.analyze.core.onnx_loader.ONNXLoader")
    @patch("winml.modelkit.analyze.core.pattern_extractor.PatternExtractor")
    def test_analyze_from_proto_always_runs_ep(
        self,
        mock_pattern_extractor_cls: Mock,
        mock_onnx_loader_cls: Mock,
        mock_runtime_checker_cls: Mock,
    ) -> None:
        """analyze_from_proto must always invoke RuntimeChecker regardless of rule data.

        Pattern extraction does not depend on rule data. RuntimeChecker is always
        instantiated; the rule-data check is deferred to op_support() where it is
        actually needed (not at the top-level EP loop).
        """
        mock_model = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_model
        mock_onnx_loader_cls.return_value = mock_loader

        mock_extractor = MagicMock()
        mock_extractor.summary.return_value = {
            "summary": ModelStats(
                model_path="test.onnx",
                opset_version=13,
                total_operators=10,
                operator_counts={"Conv": 10},
                unique_operator_types=1,
                detected_pattern_count={},
            ),
            "subgraph_patterns": [],
        }
        mock_pattern_extractor_cls.return_value = mock_extractor

        mock_runtime_checker = MagicMock()
        mock_runtime_checker.summary.return_value = {
            "op_runtime_check_result": [],
        }
        mock_runtime_checker_cls.return_value = mock_runtime_checker

        analyzer = ONNXStaticAnalyzer()
        model_proto = MagicMock(spec=onnx.ModelProto)

        result = analyzer.analyze_from_proto(
            model_proto=model_proto,
            ep="QNNExecutionProvider",
            device="NPU",
            enable_information=False,
        )

        assert isinstance(result, AnalysisResult)
        # Pattern extraction metadata is always present
        assert result.output.metadata.total_operators == 10
        # RuntimeChecker must always be instantiated — rule-data check is deferred
        mock_runtime_checker_cls.assert_called_once()
