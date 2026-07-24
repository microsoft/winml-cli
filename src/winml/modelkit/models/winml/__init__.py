# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Inference Classes Package.

Contains task mapping system and inference wrappers for WinML models.

Components:
- TASK_TO_WINML_CLASS: Task -> Class name mapping (Level 1)
- WINML_MODEL_CLASS_MAPPING: (model_type, task) -> Specialized class (Level 2)
- get_winml_class(): Two-level class lookup function
- WinMLPreTrainedModel: Base class for all inference models
- WinMLModelFor*: Task-specific inference wrappers
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .base import WinMLPreTrainedModel

logger = logging.getLogger(__name__)


# =============================================================================
# Two-Level Task Mapping System
# =============================================================================

# Level 1: Task -> Universal WinML class name (lazy import)
TASK_TO_WINML_CLASS: dict[str, str] = {
    # Implemented
    "image-classification": "WinMLModelForImageClassification",
    "text-classification": "WinMLModelForSequenceClassification",
    "sequence-classification": "WinMLModelForSequenceClassification",
    "next-sentence-prediction": "WinMLModelForSequenceClassification",
    "image-segmentation": "WinMLModelForImageSegmentation",
    "semantic-segmentation": "WinMLModelForSemanticSegmentation",
    "object-detection": "WinMLModelForObjectDetection",
    "depth-estimation": "WinMLModelForDepthEstimation",
    # Not yet implemented — falls back to WinMLModelForGenericTask at runtime
    "token-classification": "WinMLModelForTokenClassification",
    "question-answering": "WinMLModelForQuestionAnswering",
    "text-generation": "WinMLModelForCausalLM",
    "text2text-generation": "WinMLModelForSeq2SeqLM",
    "fill-mask": "WinMLModelForMaskedLM",
    "feature-extraction": "WinMLModelForFeatureExtraction",
    "sentence-similarity": "WinMLModelForFeatureExtraction",
    "image-feature-extraction": "WinMLModelForFeatureExtraction",
}

# Level 2: (model_type, task) -> Specialized class (exceptions only)
WINML_MODEL_CLASS_MAPPING: dict[tuple[str, str], str] = {
    # Only add entries for models that need:
    # - Custom export logic (broken ONNX export)
    # - Specific OPSET requirements
    # - Input remapping (non-standard tensor names)
    # - Custom pre/post-processing
}


def _import_winml_class(class_name: str) -> type[WinMLPreTrainedModel]:
    """Lazy import WinML model class by name.

    Args:
        class_name: Name of the WinML model class to import

    Returns:
        The WinML model class

    Raises:
        ImportError: If class is not implemented yet
    """
    from .base import WinMLModelForGenericTask
    from .depth_estimation import WinMLModelForDepthEstimation
    from .feature_extraction import WinMLModelForFeatureExtraction
    from .image_classification import WinMLModelForImageClassification
    from .image_segmentation import (
        WinMLModelForImageSegmentation,
        WinMLModelForSemanticSegmentation,
    )
    from .object_detection import WinMLModelForObjectDetection
    from .question_answering import WinMLModelForQuestionAnswering
    from .sequence_classification import WinMLModelForSequenceClassification

    # Map class names to modules
    class_map: dict[str, type] = {
        "WinMLModelForDepthEstimation": WinMLModelForDepthEstimation,
        "WinMLModelForFeatureExtraction": WinMLModelForFeatureExtraction,
        "WinMLModelForImageClassification": WinMLModelForImageClassification,
        "WinMLModelForImageSegmentation": WinMLModelForImageSegmentation,
        "WinMLModelForObjectDetection": WinMLModelForObjectDetection,
        "WinMLModelForQuestionAnswering": WinMLModelForQuestionAnswering,
        "WinMLModelForSemanticSegmentation": WinMLModelForSemanticSegmentation,
        "WinMLModelForSequenceClassification": WinMLModelForSequenceClassification,
        "WinMLModelForGenericTask": WinMLModelForGenericTask,
    }

    if class_name not in class_map:
        raise ImportError(
            f"Cannot import {class_name} from winml.modelkit.models.winml. "
            f"Class may not be implemented yet."
        )

    return class_map[class_name]


def get_winml_class(model_type: str | None, task: str | None) -> type[WinMLPreTrainedModel]:
    """Get appropriate WinML class using three-level mapping.

    Level 1: Check class mapping (model_type, task) -> specialized class
    Level 2: Fallback to universal class by task
    Level 3: Fallback to WinMLModelForGenericTask for unknown tasks

    Args:
        model_type: Model type from config (e.g., "convnext"). Can be None
            for ONNX-only builds where HF config is unavailable.
        task: Canonical task name (e.g., "image-classification")

    Returns:
        WinMLPreTrainedModel subclass (never raises for unknown tasks)
    """
    model_type_normalized = model_type.lower().replace("_", "-") if model_type else None

    # Level 1: Check for (model_type, task) class mapping
    if model_type_normalized is not None and task is not None:
        specialized_name = WINML_MODEL_CLASS_MAPPING.get((model_type_normalized, task))
    else:
        specialized_name = None
    if specialized_name:
        logger.debug("Using specialized class: %s", specialized_name)
        return _import_winml_class(specialized_name)

    # Level 2: Universal class by task
    class_name = TASK_TO_WINML_CLASS.get(task) if task is not None else None
    if class_name is not None:
        # Try to import the class - if not implemented, fall through to Level 3
        try:
            logger.debug("Using universal class: %s", class_name)
            return _import_winml_class(class_name)
        except ImportError:
            logger.warning(
                "Class %s not implemented for task '%s', using generic fallback",
                class_name,
                task,
            )

    # Level 3: Generic fallback for unknown/unsupported tasks
    logger.info(
        "No specific class for task '%s', using WinMLModelForGenericTask",
        task,
    )
    return _import_winml_class("WinMLModelForGenericTask")


def get_supported_tasks() -> list[str]:
    """Get list of all supported tasks."""
    return sorted(TASK_TO_WINML_CLASS.keys())


def register_specialization(model_type: str, task: str, class_name: str) -> None:
    """Register a model class mapping.

    Args:
        model_type: Model type (e.g., "convnext")
        task: Task name (e.g., "image-classification")
        class_name: Name of the specialized WinML class
    """
    key = (model_type.lower().replace("_", "-"), task)
    WINML_MODEL_CLASS_MAPPING[key] = class_name
    logger.info("Registered class mapping: %s -> %s", key, class_name)


# =============================================================================
# Re-exports
# =============================================================================

from .base import WinMLModelForGenericTask, WinMLPreTrainedModel
from .composite_model import (
    COMPOSITE_MODEL_REGISTRY,
    WinMLCompositeModel,
    register_composite_model,
)
from .decoder_only import WinMLDecoderOnlyModel
from .depth_estimation import WinMLModelForDepthEstimation
from .encoder_decoder import WinMLEncoderDecoderModel
from .feature_extraction import WinMLModelForFeatureExtraction
from .genai_bundle import (
    GENAI_BUNDLE_REGISTRY,
    GenaiBundleRecipe,
    GenaiCompanionSpec,
    GenaiTarget,
    GenaiTransformerSpec,
    build_genai_bundle,
    register_genai_bundle,
    resolve_genai_bundle,
)
from .genai_causal_lm import CausalLMOutput, HFCausalLM, WinMLGenaiCausalLM
from .image_classification import WinMLModelForImageClassification
from .image_segmentation import (
    ImageSegmentationOutput,
    WinMLModelForImageSegmentation,
    WinMLModelForSemanticSegmentation,
)
from .kv_cache import (
    WinMLCache,
    WinMLSlidingWindowCache,
    WinMLStaticCache,
)
from .object_detection import WinMLModelForObjectDetection
from .sequence_classification import WinMLModelForSequenceClassification
from .zero_shot_image_classification import WinMLModelForZeroShotImageClassification


__all__ = [
    "COMPOSITE_MODEL_REGISTRY",
    "GENAI_BUNDLE_REGISTRY",
    "TASK_TO_WINML_CLASS",
    "WINML_MODEL_CLASS_MAPPING",
    "CausalLMOutput",
    "GenaiBundleRecipe",
    "GenaiCompanionSpec",
    "GenaiTarget",
    "GenaiTransformerSpec",
    "HFCausalLM",
    "ImageSegmentationOutput",
    "WinMLCache",
    "WinMLCompositeModel",
    "WinMLDecoderOnlyModel",
    "WinMLEncoderDecoderModel",
    "WinMLGenaiCausalLM",
    "WinMLModelForDepthEstimation",
    "WinMLModelForFeatureExtraction",
    "WinMLModelForGenericTask",
    "WinMLModelForImageClassification",
    "WinMLModelForImageSegmentation",
    "WinMLModelForObjectDetection",
    "WinMLModelForSemanticSegmentation",
    "WinMLModelForSequenceClassification",
    "WinMLModelForZeroShotImageClassification",
    "WinMLPreTrainedModel",
    "WinMLSlidingWindowCache",
    "WinMLStaticCache",
    "build_genai_bundle",
    "get_supported_tasks",
    "get_winml_class",
    "register_composite_model",
    "register_genai_bundle",
    "register_specialization",
    "resolve_genai_bundle",
]
