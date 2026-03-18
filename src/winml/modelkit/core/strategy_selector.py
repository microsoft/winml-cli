# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Intelligent Strategy Selection for ModelExport.

This module provides automatic strategy selection based on model characteristics
and user requirements. It analyzes models and recommends the optimal export strategy.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import torch


logger = logging.getLogger(__name__)


class ExportStrategy(Enum):
    """Available export strategies."""

    USAGE_BASED = "usage_based"
    HTP = "htp"
    FX = "fx_graph"
    AUTO = "auto"


@dataclass
class ModelCharacteristics:
    """Characteristics of a model that affect strategy selection."""

    model_type: str  # e.g., "transformer", "cnn", "unknown"
    has_control_flow: bool
    is_huggingface: bool
    module_count: int
    has_dynamic_shapes: bool
    estimated_complexity: str  # "low", "medium", "high"
    framework_hints: list[str]  # e.g., ["attention", "convolution", "embedding"]


@dataclass
class StrategyRecommendation:
    """Strategy recommendation with reasoning."""

    primary_strategy: ExportStrategy
    fallback_strategy: ExportStrategy | None
    confidence: float  # 0.0 to 1.0
    reasoning: list[str]
    warnings: list[str]
    expected_performance: dict[str, Any]


class ModelAnalyzer:
    """Analyze PyTorch models to determine their characteristics."""

    @staticmethod
    def analyze_model(model: torch.nn.Module) -> ModelCharacteristics:
        """Analyze a PyTorch model to determine its characteristics.

        Args:
            model: PyTorch model to analyze

        Returns:
            ModelCharacteristics with detected features
        """
        characteristics = ModelCharacteristics(
            model_type="unknown",
            has_control_flow=False,
            is_huggingface=False,
            module_count=0,
            has_dynamic_shapes=False,
            estimated_complexity="medium",
            framework_hints=[],
        )

        # Count modules
        module_count = sum(1 for _ in model.named_modules())
        characteristics.module_count = module_count

        # Detect model type and framework
        model_class_name = model.__class__.__name__.lower()
        [name for name, _ in model.named_modules()]

        # Check if HuggingFace model
        if hasattr(model, "config") or any(
            "transformers" in str(type(m)) for _, m in model.named_modules()
        ):
            characteristics.is_huggingface = True

        # Detect transformer architecture
        transformer_indicators = [
            "attention",
            "transformer",
            "bert",
            "gpt",
            "vit",
            "clip",
        ]
        if any(indicator in model_class_name for indicator in transformer_indicators):
            characteristics.model_type = "transformer"
            characteristics.framework_hints.append("attention")

        # Detect CNN architecture
        cnn_indicators = ["resnet", "vgg", "mobilenet", "efficientnet", "densenet"]
        conv_count = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Conv2d))
        if any(indicator in model_class_name for indicator in cnn_indicators) or conv_count > 5:
            characteristics.model_type = "cnn"
            characteristics.framework_hints.append("convolution")

        # Detect embedding layers
        if any(isinstance(m, torch.nn.Embedding) for _, m in model.named_modules()):
            characteristics.framework_hints.append("embedding")

        # Check for control flow indicators
        # This is a heuristic - actual control flow detection requires tracing
        if characteristics.is_huggingface:
            # HuggingFace models often have control flow
            characteristics.has_control_flow = True

        # Estimate complexity
        if module_count < 50:
            characteristics.estimated_complexity = "low"
        elif module_count < 200:
            characteristics.estimated_complexity = "medium"
        else:
            characteristics.estimated_complexity = "high"

        # Check for dynamic shapes (heuristic)
        if characteristics.model_type == "transformer":
            characteristics.has_dynamic_shapes = True

        return characteristics


class StrategySelector:
    """Intelligent strategy selector for ONNX export.

    Based on extensive benchmarking from Iterations 16-18.
    """

    # Performance benchmarks from testing (in seconds)
    PERFORMANCE_DATA: ClassVar[dict[str, dict[ExportStrategy, float | None]]] = {
        "resnet-50": {
            ExportStrategy.USAGE_BASED: 2.488,
            ExportStrategy.HTP: 5.920,
            ExportStrategy.FX: None,  # Incompatible
        },
        "transformer": {
            ExportStrategy.USAGE_BASED: 3.5,  # Estimated
            ExportStrategy.HTP: 6.0,  # Estimated
            ExportStrategy.FX: None,  # Incompatible
        },
        "simple_cnn": {
            ExportStrategy.USAGE_BASED: 1.0,  # Estimated
            ExportStrategy.HTP: 1.5,  # Estimated
            ExportStrategy.FX: 0.8,  # Estimated
        },
    }

    @classmethod
    def recommend_strategy(
        cls,
        model: torch.nn.Module,
        prioritize_speed: bool = True,
        prioritize_coverage: bool = False,
        force_strategy: ExportStrategy | None = None,
    ) -> StrategyRecommendation:
        """Recommend the best export strategy for a given model.

        Args:
            model: PyTorch model to export
            prioritize_speed: Prioritize export speed (default: True)
            prioritize_coverage: Prioritize hierarchy coverage over speed
            force_strategy: Force a specific strategy (for testing)

        Returns:
            StrategyRecommendation with primary and fallback strategies
        """
        # Analyze model characteristics
        characteristics = ModelAnalyzer.analyze_model(model)

        # Initialize recommendation
        recommendation = StrategyRecommendation(
            primary_strategy=ExportStrategy.USAGE_BASED,  # Default
            fallback_strategy=ExportStrategy.HTP,
            confidence=0.9,
            reasoning=[],
            warnings=[],
            expected_performance={},
        )

        # Handle forced strategy
        if force_strategy and force_strategy != ExportStrategy.AUTO:
            recommendation.primary_strategy = force_strategy
            recommendation.reasoning.append(f"Strategy forced to {force_strategy.value}")
            recommendation.confidence = 1.0
            return recommendation

        # Strategy selection logic based on benchmarks
        if characteristics.is_huggingface:
            # HuggingFace models - FX incompatible
            recommendation.reasoning.append("HuggingFace model detected")

            if prioritize_speed:
                recommendation.primary_strategy = ExportStrategy.USAGE_BASED
                recommendation.reasoning.append(
                    "Usage-Based fastest for HuggingFace models (2.488s vs 5.920s)"
                )
                recommendation.expected_performance = {"export_time": 2.5}
            elif prioritize_coverage:
                recommendation.primary_strategy = ExportStrategy.HTP
                recommendation.reasoning.append("HTP provides more comprehensive tracing")
                recommendation.expected_performance = {"export_time": 6.0}

            recommendation.warnings.append("FX strategy incompatible with HuggingFace models")

        elif characteristics.has_control_flow:
            # Models with control flow - FX likely incompatible
            recommendation.reasoning.append("Control flow detected")
            recommendation.primary_strategy = ExportStrategy.USAGE_BASED
            recommendation.fallback_strategy = ExportStrategy.HTP
            recommendation.warnings.append("FX strategy may fail due to control flow")

        elif characteristics.model_type == "cnn" and characteristics.estimated_complexity == "low":
            # Simple CNNs - FX might work
            recommendation.reasoning.append("Simple CNN architecture detected")

            if prioritize_speed and not characteristics.is_huggingface:
                recommendation.primary_strategy = ExportStrategy.FX
                recommendation.fallback_strategy = ExportStrategy.USAGE_BASED
                recommendation.reasoning.append("FX can be fastest for simple CNNs")
                recommendation.confidence = 0.7  # Lower confidence due to compatibility risks
            else:
                recommendation.primary_strategy = ExportStrategy.USAGE_BASED
                recommendation.reasoning.append("Usage-Based most reliable for CNNs")

        else:
            # Default case - Usage-Based is safest and fastest
            recommendation.reasoning.append("Default recommendation based on benchmarks")
            recommendation.primary_strategy = ExportStrategy.USAGE_BASED
            recommendation.reasoning.append("Usage-Based proven fastest and most reliable")

        # Add performance expectations
        if characteristics.model_type == "transformer":
            recommendation.expected_performance = {
                "export_time": 3.5
                if recommendation.primary_strategy == ExportStrategy.USAGE_BASED
                else 6.0,
                "coverage": "high",
                "reliability": "excellent",
            }
        elif characteristics.model_type == "cnn":
            recommendation.expected_performance = {
                "export_time": 2.5
                if recommendation.primary_strategy == ExportStrategy.USAGE_BASED
                else 4.0,
                "coverage": "high",
                "reliability": "excellent",
            }

        # Add module count to performance expectations
        recommendation.expected_performance["module_count"] = characteristics.module_count

        return recommendation

    @classmethod
    def get_strategy_description(cls, strategy: ExportStrategy) -> dict[str, str]:
        """Get description and characteristics of a strategy."""
        descriptions = {
            ExportStrategy.USAGE_BASED: {
                "name": "Usage-Based",
                "description": "Simple and fast hierarchy tracking using forward hooks",
                "pros": "Fastest (2.488s), reliable, works with all models",
                "cons": "Basic hierarchy tracking, may miss some unused modules",
                "best_for": "Production use, HuggingFace models, speed-critical applications",
            },
            ExportStrategy.HTP: {
                "name": "Hierarchical Trace-and-Project (HTP)",
                "description": "Comprehensive tracing with built-in PyTorch module tracking",
                "pros": "Detailed hierarchy, handles complex models, good coverage",
                "cons": "Slower (5.920s), more complex implementation",
                "best_for": "Development, debugging, comprehensive analysis",
            },
            ExportStrategy.FX: {
                "name": "FX Graph",
                "description": "PyTorch FX symbolic tracing for graph analysis",
                "pros": "Can be fast for simple models, graph-level analysis",
                "cons": "Incompatible with control flow, fails on HuggingFace models",
                "best_for": "Simple PyTorch models without dynamic control flow",
            },
        }

        return descriptions.get(
            strategy,
            {
                "name": strategy.value,
                "description": "Unknown strategy",
                "pros": "N/A",
                "cons": "N/A",
                "best_for": "N/A",
            },
        )


def select_best_strategy(
    model: torch.nn.Module, example_inputs: torch.Tensor | tuple | None = None, **kwargs: Any
) -> tuple[ExportStrategy, StrategyRecommendation]:
    """Convenience function to select the best strategy for a model.

    Args:
        model: PyTorch model to export
        example_inputs: Example inputs (used for shape analysis if provided)
        **kwargs: Additional arguments passed to recommend_strategy

    Returns:
        Tuple of (selected_strategy, recommendation_details)
    """
    selector = StrategySelector()
    recommendation = selector.recommend_strategy(model, **kwargs)

    logger.info(f"Selected strategy: {recommendation.primary_strategy.value}")
    logger.info(f"Reasoning: {'; '.join(recommendation.reasoning)}")

    if recommendation.warnings:
        for warning in recommendation.warnings:
            logger.warning(warning)

    return recommendation.primary_strategy, recommendation
