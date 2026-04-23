# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Evaluation metrics."""

from .classification import ClassificationMetric
from .knn_accuracy import KNNAccuracyMetric
from .mean_average_precision import MAPMetric
from .mean_iou import IGNORE_INDEX, MeanIoUMetric
from .pseudo_perplexity import PseudoPerplexityMetric
from .spearman_correlation import SpearmanCorrelationMetric


__all__ = [
    "IGNORE_INDEX",
    "ClassificationMetric",
    "KNNAccuracyMetric",
    "MAPMetric",
    "MeanIoUMetric",
    "PseudoPerplexityMetric",
    "SpearmanCorrelationMetric",
]
