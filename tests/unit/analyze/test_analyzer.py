# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ONNXStaticAnalyzer."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import onnx
import pytest

from winml.modelkit.analyze.analyzer import _build_runtime_debug_details_summary
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
    infer_ihv_from_ep_name,
)
from winml.modelkit.analyze.models.runtime_checks import PatternRuntime, RuntimeTestResult
from winml.modelkit.optim import WinMLOptimizationConfig


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
        assert repr(result) == "AnalysisResult(patterns=0)"

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

    def test_build_runtime_debug_details_summary_groups_and_filters_unknown(self) -> None:
        """Should group by support level and filter unknown results."""
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
                            "table_path": "/tmp/conv.parquet",
                            "table_file": "conv.parquet",
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
                            "table_path": "/tmp/resize.parquet",
                            "table_file": "resize.parquet",
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
                            "table_path": "/tmp/unknown.parquet",
                            "table_file": "unknown.parquet",
                        },
                    ),
                ),
            ],
            "subgraph_runtime_check_result": [
                PatternRuntime(
                    pattern_id="SUBGRAPH/TestPattern",
                    result=RuntimeTestResult(
                        compile=False,
                        run=False,
                        debug_details={
                            "node_stable_key": "node_subgraph",
                            "case_indices": ["case_5"],
                            "table_path": "/tmp/subgraph.parquet",
                            "table_file": "subgraph.parquet",
                        },
                    ),
                )
            ],
        }

        summary = _build_runtime_debug_details_summary(runtime_summary)

        assert summary is not None
        assert set(summary.keys()) == {"supported", "partial", "unsupported"}

        assert summary["supported"]["node_conv"]["case_indices"] == ["case_1", "case_2"]
        assert summary["supported"]["node_conv"]["table_path"] == "/tmp/conv.parquet"
        assert summary["supported"]["node_conv"]["table_file"] == "conv.parquet"

        assert summary["partial"]["node_resize"]["case_indices"] == ["case_3"]
        assert summary["partial"]["node_resize"]["table_path"] == "/tmp/resize.parquet"
        assert summary["partial"]["node_resize"]["table_file"] == "resize.parquet"

        assert summary["unsupported"]["node_subgraph"]["case_indices"] == ["case_5"]
        assert summary["unsupported"]["node_subgraph"]["table_path"] == "/tmp/subgraph.parquet"
        assert summary["unsupported"]["node_subgraph"]["table_file"] == "subgraph.parquet"

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
                            "table_path": "/tmp/conv.parquet",
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
                        },
                    ),
                ),
            ],
            "subgraph_runtime_check_result": [],
        }

        summary = _build_runtime_debug_details_summary(runtime_summary)

        assert summary is not None
        node_entry = summary["supported"]["node_conv"]
        assert node_entry["table_path"] == "/tmp/conv.parquet"
        assert node_entry["table_file"] == "conv.parquet"
        assert node_entry["case_indices"] == ["case_42"]


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

    def test_map_ep_to_ihv_qnn(self) -> None:
        """Test EP to IHV mapping for QNN."""
        assert infer_ihv_from_ep_name("QNNExecutionProvider") == IHVType.QC
        assert infer_ihv_from_ep_name("qnnexecutionprovider") == IHVType.QC
        assert infer_ihv_from_ep_name("QualcommProvider") == IHVType.QC

    def test_map_ep_to_ihv_openvino(self) -> None:
        """Test EP to IHV mapping for OpenVINO."""
        assert infer_ihv_from_ep_name("OpenVINOExecutionProvider") == IHVType.INTEL
        assert infer_ihv_from_ep_name("openvino") == IHVType.INTEL
        assert infer_ihv_from_ep_name("IntelProvider") == IHVType.INTEL

    def test_map_ep_to_ihv_vitisai(self) -> None:
        """Test EP to IHV mapping for VitisAI."""
        assert infer_ihv_from_ep_name("VitisAIExecutionProvider") == IHVType.AMD
        assert infer_ihv_from_ep_name("vitis") == IHVType.AMD
        assert infer_ihv_from_ep_name("AMDProvider") == IHVType.AMD

    def test_map_ep_to_ihv_nvidia(self) -> None:
        """Test EP to IHV mapping for NvTensorRTRTX."""
        assert infer_ihv_from_ep_name("NvTensorRTRTXExecutionProvider") == IHVType.NVIDIA
        assert infer_ihv_from_ep_name("nvtensorrtx") == IHVType.NVIDIA
        assert infer_ihv_from_ep_name("TensorRTProvider") == IHVType.NVIDIA

    def test_map_ep_to_ihv_invalid(self) -> None:
        """Test EP to IHV mapping with unrecognized EP resolves to MICROSOFT."""
        assert infer_ihv_from_ep_name("InvalidEP") == IHVType.MICROSOFT

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
            "subgraph_runtime_check_result": [],
        }
        mock_runtime_checker_cls.return_value = mock_checker

        # Create analyzer
        analyzer = ONNXStaticAnalyzer()

        # Mock model proto
        model_proto = MagicMock(spec=onnx.ModelProto)

        # Analyze
        result = analyzer.analyze_from_proto(
            model_proto=model_proto,
            ep="QNNExecutionProvider",
            device="NPU",
            enable_information=False,
        )

        # Assertions
        assert isinstance(result, AnalysisResult)
        assert len(result.output.results) == 1
        assert result.output.results[0].ihv_type == IHVType.QC

        # Verify RuntimeChecker was called once
        assert mock_runtime_checker_cls.call_count == 1

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
                            "table_path": "/tmp/conv.parquet",
                            "table_file": "conv.parquet",
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
                            "table_path": "/tmp/relu.parquet",
                            "table_file": "relu.parquet",
                        },
                    ),
                ),
            ],
            "subgraph_runtime_check_result": [],
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
        assert ep_result.runtime_debug_details_summary["supported"]["node_conv"] == {
            "case_indices": ["case_7"],
            "table_path": "/tmp/conv.parquet",
            "table_file": "conv.parquet",
        }
        assert ep_result.runtime_debug_details_summary["partial"] == {}
        assert ep_result.runtime_debug_details_summary["unsupported"] == {}
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
            "subgraph_runtime_check_result": [],
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
        # Should have results for all 4 EPs: QNN, OpenVINO, VitisAI, NvTensorRTRTX
        assert len(result.output.results) == 4

        ihv_types = {r.ihv_type for r in result.output.results}
        assert IHVType.QC in ihv_types
        assert IHVType.INTEL in ihv_types
        assert IHVType.AMD in ihv_types
        assert IHVType.NVIDIA in ihv_types

        # Verify RuntimeChecker was called 4 times (once per EP)
        assert mock_runtime_checker_cls.call_count == 4

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
            "subgraph_runtime_check_result": [],
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
            "subgraph_runtime_check_result": [],
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
            "subgraph_runtime_check_result": [],
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
