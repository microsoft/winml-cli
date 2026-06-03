# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""WinML Evaluation Module.

Provides accuracy evaluation using HuggingFace pipeline + evaluate library.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .base_evaluator import WinMLEvaluator
from .config import DatasetConfig, WinMLEvaluationConfig
from .evaluate import EvalResult, evaluate, get_evaluator_class


if TYPE_CHECKING:
    from .feature_extraction_evaluator import WinMLFeatureExtractionEvaluator
    from .fill_mask_evaluator import WinMLFillMaskEvaluator
    from .image_feature_extraction_evaluator import WinMLImageFeatureExtractionEvaluator
    from .image_segmentation_evaluator import WinMLImageSegmentationEvaluator
    from .image_to_text_evaluator import WinMLImageToTextEvaluator
    from .metrics.classification import ClassificationMetric
    from .metrics.knn_accuracy import KNNAccuracyMetric
    from .metrics.mean_average_precision import MAPMetric
    from .metrics.mean_iou import IGNORE_INDEX, MeanIoUMetric
    from .metrics.pseudo_perplexity import PseudoPerplexityMetric
    from .metrics.spearman_correlation import SpearmanCorrelationMetric
    from .metrics.top_k_accuracy import TopKAccuracyMetric
    from .object_detection_evaluator import WinMLObjectDetectionEvaluator
    from .question_answering_evaluator import WinMLQuestionAnsweringEvaluator
    from .tensor_similarity_evaluator import TensorSimilarityEvaluator
    from .text_classification_evaluator import WinMLTextClassificationEvaluator
    from .token_classification_evaluator import WinMLTokenClassificationEvaluator
    from .zero_shot_classification_evaluator import WinMLZeroShotClassificationEvaluator
    from .zero_shot_image_classification_evaluator import WinMLZeroShotImageClassificationEvaluator


_LAZY_ATTRS: dict[str, str] = {
    # Evaluators
    "WinMLFeatureExtractionEvaluator":
        ".feature_extraction_evaluator:WinMLFeatureExtractionEvaluator",
    "WinMLFillMaskEvaluator":
        ".fill_mask_evaluator:WinMLFillMaskEvaluator",
    "WinMLImageFeatureExtractionEvaluator":
        ".image_feature_extraction_evaluator:WinMLImageFeatureExtractionEvaluator",
    "WinMLImageSegmentationEvaluator":
        ".image_segmentation_evaluator:WinMLImageSegmentationEvaluator",
    "WinMLImageToTextEvaluator":
        ".image_to_text_evaluator:WinMLImageToTextEvaluator",
    "WinMLObjectDetectionEvaluator":
        ".object_detection_evaluator:WinMLObjectDetectionEvaluator",
    "WinMLQuestionAnsweringEvaluator":
        ".question_answering_evaluator:WinMLQuestionAnsweringEvaluator",
    "WinMLTextClassificationEvaluator":
        ".text_classification_evaluator:WinMLTextClassificationEvaluator",
    "WinMLTokenClassificationEvaluator":
        ".token_classification_evaluator:WinMLTokenClassificationEvaluator",
    "WinMLZeroShotClassificationEvaluator":
        ".zero_shot_classification_evaluator:WinMLZeroShotClassificationEvaluator",
    "WinMLZeroShotImageClassificationEvaluator":
        ".zero_shot_image_classification_evaluator:WinMLZeroShotImageClassificationEvaluator",
    "TensorSimilarityEvaluator":
        ".tensor_similarity_evaluator:TensorSimilarityEvaluator",
    # Metrics (defer numpy / scipy / torch / torchmetrics until first use)
    "ClassificationMetric":
        ".metrics.classification:ClassificationMetric",
    "IGNORE_INDEX":
        ".metrics.mean_iou:IGNORE_INDEX",
    "KNNAccuracyMetric":
        ".metrics.knn_accuracy:KNNAccuracyMetric",
    "MAPMetric":
        ".metrics.mean_average_precision:MAPMetric",
    "MeanIoUMetric":
        ".metrics.mean_iou:MeanIoUMetric",
    "PseudoPerplexityMetric":
        ".metrics.pseudo_perplexity:PseudoPerplexityMetric",
    "SpearmanCorrelationMetric":
        ".metrics.spearman_correlation:SpearmanCorrelationMetric",
    "TopKAccuracyMetric":
        ".metrics.top_k_accuracy:TopKAccuracyMetric",
}


def __getattr__(name: str) -> Any:
    """Lazy attribute loader (PEP 562)."""
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, symbol = spec.rsplit(":", 1)
    module = importlib.import_module(module_path, package=__name__)
    value = getattr(module, symbol)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = [
    "IGNORE_INDEX",
    "ClassificationMetric",
    "DatasetConfig",
    "EvalResult",
    "KNNAccuracyMetric",
    "MAPMetric",
    "MeanIoUMetric",
    "PseudoPerplexityMetric",
    "SpearmanCorrelationMetric",
    "TensorSimilarityEvaluator",
    "TopKAccuracyMetric",
    "WinMLEvaluationConfig",
    "WinMLEvaluator",
    "WinMLFeatureExtractionEvaluator",
    "WinMLFillMaskEvaluator",
    "WinMLImageFeatureExtractionEvaluator",
    "WinMLImageSegmentationEvaluator",
    "WinMLImageToTextEvaluator",
    "WinMLObjectDetectionEvaluator",
    "WinMLQuestionAnsweringEvaluator",
    "WinMLTextClassificationEvaluator",
    "WinMLTokenClassificationEvaluator",
    "WinMLZeroShotClassificationEvaluator",
    "WinMLZeroShotImageClassificationEvaluator",
    "evaluate",
    "get_evaluator_class",
]
