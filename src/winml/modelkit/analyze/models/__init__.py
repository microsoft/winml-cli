# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pydantic data models for ONNX Static Analyzer."""

from ...pattern.match import InputInfo, PatternMatchResult
from ...pattern.models import OperatorPattern, Pattern, PatternType, SubgraphPattern
from .ihv_type import IHVType
from .information import Action, ActionItem, ActionLevel, Information
from .onnx_model import ModelTag, ONNXModel
from .onnx_op import ONNXOp
from .output import AnalysisOutput, EPSupport, ModelStats, extract_model_stats
from .runtime_checks import AlternativeType, RuntimeCheckRule, RuntimeTestResult
from .support_level import SupportLevel


__all__ = [
    "Action",
    "ActionItem",
    "ActionLevel",
    "AlternativeType",
    "AnalysisOutput",
    "EPSupport",
    "IHVType",
    "Information",
    "InputInfo",
    "ModelStats",
    "ModelTag",
    "ONNXModel",
    "ONNXOp",
    "OperatorPattern",
    "Pattern",
    "PatternMatchResult",
    "PatternType",
    "RuntimeCheckRule",
    "RuntimeTestResult",
    "SubgraphPattern",
    "SupportLevel",
    "extract_model_stats",
]
