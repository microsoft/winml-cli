# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for analyze_onnx() wrapper API and AnalyzeResult dataclass."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from winml.modelkit.analyze.analyzer import (
    AnalysisResult,
    AnalyzeResult,
    LintResult,
    analyze_onnx,
)
from winml.modelkit.analyze.models.ihv_type import IHVType
from winml.modelkit.analyze.models.information import Action, ActionItem, Information
from winml.modelkit.analyze.models.output import AnalysisOutput, EPSupport, ModelStats
from winml.modelkit.analyze.models.support_level import SupportLevel
from winml.modelkit.optim import WinMLOptimizationConfig


# =============================================================================
# FIXTURES
# =============================================================================


def _make_lint_result(
    *,
    errors: int = 0,
    warnings: int = 0,
    info: int = 0,
    error_patterns: list[str] | None = None,
    warning_patterns: list[str] | None = None,
    optimization_config: WinMLOptimizationConfig | None = None,
) -> LintResult:
    """Build a LintResult for testing."""
    return LintResult(
        errors=errors,
        warnings=warnings,
        info=info,
        passed=errors == 0 and warnings == 0,
        error_patterns=error_patterns or [],
        warning_patterns=warning_patterns or [],
        information=[],
        optimization_config=optimization_config or WinMLOptimizationConfig(),
    )


def _make_mock_output(
    *,
    ep: str = "QNNExecutionProvider",
    ihv_type: IHVType = IHVType.QC,
    has_errors: bool = False,
    has_warnings: bool = False,
    unsupported_patterns: list[str] | None = None,
    partial_patterns: list[str] | None = None,
    information: list[Information] | None = None,
) -> AnalysisOutput:
    """Build a mock AnalysisOutput for testing."""
    metadata = ModelStats(
        model_path="test.onnx",
        opset_version=13,
        total_operators=10,
        operator_counts={"Conv": 5, "Relu": 5},
        unique_operator_types=2,
        detected_pattern_count={},
    )
    ep_support = EPSupport(
        ihv_type=ihv_type,
        ep_type=ep,
        runtime_support=not has_errors,
        has_errors=has_errors,
        has_warnings=has_warnings,
        classification={
            SupportLevel.SUPPORTED: ["Conv", "Relu"],
            SupportLevel.PARTIAL: partial_patterns or [],
            SupportLevel.UNSUPPORTED: unsupported_patterns or [],
            SupportLevel.UNKNOWN: [],
        },
        information=information or [],
    )
    return AnalysisOutput(
        analyzer_version="0.1.0",
        metadata=metadata,
        results=[ep_support],
    )


# =============================================================================
# AnalyzeResult DATACLASS TESTS
# =============================================================================


class TestAnalyzeResult:
    """Tests for the AnalyzeResult dataclass."""

    def test_has_errors_true_when_lint_has_errors(self) -> None:
        """has_errors reflects lint.errors > 0."""
        lint = _make_lint_result(errors=2, error_patterns=["Upsample", "NonZero"])
        result = AnalyzeResult(lint=lint, optimization_config=WinMLOptimizationConfig())
        assert result.has_errors is True

    def test_has_errors_false_when_no_lint_errors(self) -> None:
        """has_errors is False when lint has no errors."""
        lint = _make_lint_result(errors=0)
        result = AnalyzeResult(lint=lint, optimization_config=WinMLOptimizationConfig())
        assert result.has_errors is False

    def test_autoconf_truthy_with_nonempty_config(self) -> None:
        """autoconf is truthy when config has flags."""
        lint = _make_lint_result()
        config = WinMLOptimizationConfig(gelu_fusion=True)
        result = AnalyzeResult(lint=lint, optimization_config=config)
        assert result.autoconf  # truthy — has flags
        assert result.autoconf["gelu_fusion"] is True

    def test_autoconf_falsy_with_empty_config(self) -> None:
        """autoconf is falsy when config is empty."""
        lint = _make_lint_result()
        result = AnalyzeResult(lint=lint, optimization_config=WinMLOptimizationConfig())
        assert not result.autoconf  # falsy — empty dict

    def test_autoconf_falsy_when_none(self) -> None:
        """autoconf is falsy when optimization_config is None."""
        lint = _make_lint_result()
        result = AnalyzeResult(lint=lint, optimization_config=None)
        assert not result.autoconf  # falsy — None

    def test_lint_field_accessible(self) -> None:
        """lint field is directly accessible."""
        lint = _make_lint_result(errors=1, warnings=2, info=3)
        result = AnalyzeResult(lint=lint, optimization_config=None)
        assert result.lint.errors == 1
        assert result.lint.warnings == 2
        assert result.lint.info == 3

    def test_optimization_config_field_accessible(self) -> None:
        """optimization_config field is directly accessible."""
        config = WinMLOptimizationConfig(gelu_fusion=True, layer_norm_fusion=True)
        result = AnalyzeResult(lint=_make_lint_result(), optimization_config=config)
        assert result.optimization_config["gelu_fusion"] is True
        assert result.optimization_config["layer_norm_fusion"] is True


# =============================================================================
# analyze_onnx() FUNCTION TESTS
# =============================================================================


class TestAnalyzeOnnx:
    """Tests for the analyze_onnx() wrapper function."""

    def test_returns_analyze_result(self, tmp_path) -> None:
        """analyze_onnx returns an AnalyzeResult instance."""
        mock_output = _make_mock_output()
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.return_value = mock_analysis
            # Create a dummy model file
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            result = analyze_onnx(str(model_file), ep="qnn", device="NPU")

        assert isinstance(result, AnalyzeResult)

    def test_lint_populated(self, tmp_path) -> None:
        """result.lint has valid LintResult fields."""
        mock_output = _make_mock_output()
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            result = analyze_onnx(str(model_file), ep="qnn", device="NPU")

        assert isinstance(result.lint, LintResult)
        assert isinstance(result.lint.errors, int)
        assert isinstance(result.lint.warnings, int)
        assert isinstance(result.lint.passed, bool)

    def test_autoconf_enabled_by_default(self, tmp_path) -> None:
        """optimization_config is not None when autoconf=True (default)."""
        mock_output = _make_mock_output()
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            result = analyze_onnx(str(model_file), ep="qnn", device="NPU")

        assert result.optimization_config is not None
        assert isinstance(result.optimization_config, WinMLOptimizationConfig)

    def test_autoconf_disabled(self, tmp_path) -> None:
        """optimization_config is None when autoconf=False."""
        mock_output = _make_mock_output()
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            result = analyze_onnx(str(model_file), ep="qnn", device="NPU", autoconf=False)

        assert result.optimization_config is None

    def test_autoconf_disabled_skips_information_engine(self, tmp_path) -> None:
        """When autoconf=False, enable_information=False is passed to analyzer."""
        mock_output = _make_mock_output()
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_analyzer = mock_cls.return_value
            mock_analyzer.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            analyze_onnx(str(model_file), ep="qnn", device="NPU", autoconf=False)

        # Verify enable_information=False was passed
        call_kwargs = mock_analyzer.analyze.call_args.kwargs
        assert call_kwargs["enable_information"] is False

    def test_autoconf_enabled_enables_information_engine(self, tmp_path) -> None:
        """When autoconf=True, enable_information=True is passed to analyzer."""
        mock_output = _make_mock_output()
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_analyzer = mock_cls.return_value
            mock_analyzer.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            analyze_onnx(str(model_file), ep="qnn", device="NPU", autoconf=True)

        call_kwargs = mock_analyzer.analyze.call_args.kwargs
        assert call_kwargs["enable_information"] is True

    def test_accepts_path_object(self, tmp_path) -> None:
        """Path objects work as the model argument."""
        from pathlib import Path

        mock_output = _make_mock_output()
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_analyzer = mock_cls.return_value
            mock_analyzer.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            result = analyze_onnx(Path(model_file), ep="qnn", device="NPU")

        assert isinstance(result, AnalyzeResult)
        # Verify path was converted to string for the analyzer
        call_kwargs = mock_analyzer.analyze.call_args.kwargs
        assert call_kwargs["model_path"] == str(model_file)

    def test_file_not_found(self) -> None:
        """FileNotFoundError raised for non-existent model."""
        with pytest.raises(FileNotFoundError):
            analyze_onnx("nonexistent_model.onnx", ep="qnn", device="NPU")

    def test_ep_passed_through(self, tmp_path) -> None:
        """ep argument is passed through to the analyzer."""
        mock_output = _make_mock_output()
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_analyzer = mock_cls.return_value
            mock_analyzer.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            analyze_onnx(str(model_file), ep="qnn", device="NPU")

        call_kwargs = mock_analyzer.analyze.call_args.kwargs
        assert call_kwargs["ep"] == "qnn"

    def test_device_passed_through(self, tmp_path) -> None:
        """device argument is passed through to the analyzer."""
        mock_output = _make_mock_output()
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_analyzer = mock_cls.return_value
            mock_analyzer.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            analyze_onnx(str(model_file), ep="qnn", device="GPU")

        call_kwargs = mock_analyzer.analyze.call_args.kwargs
        assert call_kwargs["device"] == "GPU"

    def test_ep_none_logs_warning(self, tmp_path) -> None:
        """ep=None logs a warning about multi-EP aggregation."""
        mock_output = _make_mock_output()
        mock_analysis = AnalysisResult(output=mock_output)

        with (
            patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls,
            patch("winml.modelkit.analyze.analyzer.logger") as mock_logger,
        ):
            mock_cls.return_value.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            analyze_onnx(str(model_file), ep=None, device="NPU")

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "ep=None" in warning_msg

    def test_autoconf_with_gelu_pattern(self, tmp_path) -> None:
        """Autoconf picks up gelu_fusion from action items."""
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
        info = Information(
            pattern_id="SUBGRAPH/GeluPattern",
            explanation="GELU pattern detected",
            actions=[gelu_action],
        )
        mock_output = _make_mock_output(
            has_warnings=True,
            partial_patterns=["SUBGRAPH/GeluPattern"],
            information=[info],
        )
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            result = analyze_onnx(str(model_file), ep="QNNExecutionProvider", device="NPU")

        assert result.autoconf  # truthy — has flags
        assert result.autoconf["gelu_fusion"] is True

    def test_autoconf_with_multiple_patterns(self, tmp_path) -> None:
        """Autoconf picks up multiple fusion flags from action items."""
        gelu_action = Action(
            pattern_from_id="SUBGRAPH/GeluPattern",
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
            pattern_from_id="SUBGRAPH/LayerNormPattern",
            pattern_to_id="OP/ai.onnx/LayerNormalization",
            details="Replace LayerNorm pattern",
            action_items=[
                ActionItem(
                    type="GraphOptimization",
                    optimization_options={"layer_norm_fusion": True},
                )
            ],
        )
        mock_output = _make_mock_output(
            has_warnings=True,
            partial_patterns=["SUBGRAPH/GeluPattern", "SUBGRAPH/LayerNormPattern"],
            information=[
                Information(
                    pattern_id="SUBGRAPH/GeluPattern",
                    explanation="GELU detected",
                    actions=[gelu_action],
                ),
                Information(
                    pattern_id="SUBGRAPH/LayerNormPattern",
                    explanation="LayerNorm detected",
                    actions=[layernorm_action],
                ),
            ],
        )
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            result = analyze_onnx(str(model_file), ep="QNNExecutionProvider", device="NPU")

        assert result.optimization_config["gelu_fusion"] is True
        assert result.optimization_config["layer_norm_fusion"] is True

    def test_ep_alias_normalization(self, tmp_path) -> None:
        """ep='qnn' alias produces the same result shape as full EP name."""
        mock_output = _make_mock_output(ep="QNNExecutionProvider")
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.return_value = AnalysisResult(output=mock_output)
            result_alias = analyze_onnx(str(model_file), ep="qnn", device="NPU")

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.return_value = AnalysisResult(
                output=_make_mock_output(ep="QNNExecutionProvider")
            )
            result_full = analyze_onnx(str(model_file), ep="QNNExecutionProvider", device="NPU")

        assert result_alias.has_errors == result_full.has_errors
        assert result_alias.lint.errors == result_full.lint.errors
        assert result_alias.lint.warnings == result_full.lint.warnings

    def test_autoconf_disabled_lint_config_is_empty_not_none(self, tmp_path) -> None:
        """When autoconf=False, result.optimization_config is None but
        lint.optimization_config is an empty config (not None).
        Use result.optimization_config as the canonical 'disabled' signal."""
        mock_output = _make_mock_output()
        mock_analysis = AnalysisResult(output=mock_output)

        with patch("winml.modelkit.analyze.analyzer.ONNXStaticAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.return_value = mock_analysis
            model_file = tmp_path / "test.onnx"
            model_file.write_bytes(b"dummy")

            result = analyze_onnx(str(model_file), ep="qnn", device="NPU", autoconf=False)

        # Top-level: None signals "autoconf disabled"
        assert result.optimization_config is None
        # lint internal field: always present, empty when info engine was skipped
        assert result.lint.optimization_config is not None
        assert len(result.lint.optimization_config) == 0
