# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX Static Analyzer for NPU Runtime Support Validation.

This package provides static analysis capabilities for ONNX models to determine
runtime support across NPU execution providers (QNN, Intel OpenVINO, AMD Quark).
"""

from .analyzer import (
    AnalysisResult,
    AnalyzerConfig,
    AnalyzeResult,
    LintResult,
    ONNXStaticAnalyzer,
    analyze_onnx,
)
from .core.information_engine import InformationEngine
from .core.onnx_loader import ONNXLoader
from .core.output_aggregator import OutputAggregator
from .core.pattern_extractor import PatternExtractor
from .core.runtime_checker import RuntimeChecker
from .core.runtime_checker_query import QDQ_SUFFIX
from .models import (
    Action,
    ActionItem,
    ActionLevel,
    AlternativeType,
    AnalysisOutput,
    EPSupport,
    IHVType,
    Information,
    ModelStats,
    ModelTag,
    ONNXModel,
    ONNXOp,
    RuntimeCheckRule,
    RuntimeTestResult,
    SupportLevel,
)
from .utils import infer_ihv_from_ep_name
from .utils.rule_loader import RuleLoader


__all__ = [
    "QDQ_SUFFIX",
    "Action",
    "ActionItem",
    "ActionLevel",
    "AlternativeType",
    "AnalysisOutput",
    "AnalysisResult",
    "AnalyzeResult",
    "AnalyzerConfig",
    "EPSupport",
    "IHVType",
    "Information",
    "InformationEngine",
    "LintResult",
    "ModelStats",
    "ModelTag",
    "ONNXLoader",
    "ONNXModel",
    "ONNXOp",
    "ONNXStaticAnalyzer",
    "OutputAggregator",
    "PatternExtractor",
    "RuleLoader",
    "RuntimeCheckRule",
    "RuntimeChecker",
    "RuntimeTestResult",
    "SupportLevel",
    "analyze_onnx",
    "infer_ihv_from_ep_name",
]
