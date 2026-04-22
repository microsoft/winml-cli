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
from .fill_mask_evaluator import WinMLFillMaskEvaluator
from .image_feature_extraction_evaluator import WinMLImageFeatureExtractionEvaluator
from .image_segmentation_evaluator import WinMLImageSegmentationEvaluator
from .metrics.knn_accuracy import KNNAccuracyMetric
from .metrics.mean_average_precision import MAPMetric
from .metrics.mean_iou import IGNORE_INDEX, MeanIoUMetric
from .metrics.pseudo_perplexity import PseudoPerplexityMetric
from .metrics.spearman_correlation import SpearmanCorrelationMetric
from .metrics.top_k_accuracy import TopKAccuracyMetric
from .object_detection_evaluator import WinMLObjectDetectionEvaluator
from .question_answering_evaluator import WinMLQuestionAnsweringEvaluator
from .text_classification_evaluator import WinMLTextClassificationEvaluator
from .token_classification_evaluator import WinMLTokenClassificationEvaluator
from .zero_shot_image_classification_evaluator import WinMLZeroShotImageClassificationEvaluator


__all__ = [
    "IGNORE_INDEX",
    "EvalResult",
    "KNNAccuracyMetric",
    "MAPMetric",
    "MeanIoUMetric",
    "PseudoPerplexityMetric",
    "SpearmanCorrelationMetric",
    "TopKAccuracyMetric",
    "WinMLEvaluationConfig",
    "WinMLEvaluator",
    "WinMLFeatureExtractionEvaluator",
    "WinMLFillMaskEvaluator",
    "WinMLImageFeatureExtractionEvaluator",
    "WinMLImageSegmentationEvaluator",
    "WinMLObjectDetectionEvaluator",
    "WinMLQuestionAnsweringEvaluator",
    "WinMLTextClassificationEvaluator",
    "WinMLTokenClassificationEvaluator",
    "WinMLZeroShotImageClassificationEvaluator",
    "evaluate",
]
