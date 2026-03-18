"""ONNX Static Analyzer for NPU Runtime Support Validation.

This package provides static analysis capabilities for ONNX models to determine
runtime support across NPU execution providers (QNN, Intel OpenVINO, AMD Quark).
"""

__version__ = "0.1.0"
__author__ = "WML Team"

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
from .models.output import AnalysisOutput


__all__ = [
    "AnalysisOutput",
    "AnalysisResult",
    "AnalyzeResult",
    "AnalyzerConfig",
    "InformationEngine",
    "LintResult",
    "ONNXLoader",
    "ONNXStaticAnalyzer",
    "OutputAggregator",
    "PatternExtractor",
    "RuntimeChecker",
    "__version__",
    "analyze_onnx",
]
