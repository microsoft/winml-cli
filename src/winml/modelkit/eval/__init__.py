# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""WinML Evaluation Module.

Provides accuracy evaluation using HuggingFace pipeline + evaluate library.
"""

from .base_evaluator import WinMLEvaluator
from .config import WinMLEvaluationConfig
from .evaluate import EvalResult, evaluate
from .feature_extraction_evaluator import WinMLFeatureExtractionEvaluator
from .image_segmentation_evaluator import WinMLImageSegmentationEvaluator
from .metrics.mean_average_precision import MAPMetric
from .metrics.mean_iou import IGNORE_INDEX, MeanIoUMetric
from .metrics.spearman_correlation import SpearmanCorrelationMetric
from .object_detection_evaluator import WinMLObjectDetectionEvaluator
from .text_classification_evaluator import WinMLTextClassificationEvaluator
from .token_classification_evaluator import WinMLTokenClassificationEvaluator


__all__ = [
    "IGNORE_INDEX",
    "EvalResult",
    "MAPMetric",
    "MeanIoUMetric",
    "SpearmanCorrelationMetric",
    "WinMLEvaluationConfig",
    "WinMLEvaluator",
    "WinMLFeatureExtractionEvaluator",
    "WinMLImageSegmentationEvaluator",
    "WinMLObjectDetectionEvaluator",
    "WinMLTextClassificationEvaluator",
    "WinMLTokenClassificationEvaluator",
    "evaluate",
]
