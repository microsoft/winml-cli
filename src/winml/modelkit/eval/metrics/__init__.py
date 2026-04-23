# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Evaluation metrics."""

from .classification import ClassificationMetric
from .mean_average_precision import MAPMetric
from .mean_iou import IGNORE_INDEX, MeanIoUMetric
from .spearman_correlation import SpearmanCorrelationMetric


__all__ = [
    "IGNORE_INDEX",
    "ClassificationMetric",
    "MAPMetric",
    "MeanIoUMetric",
    "SpearmanCorrelationMetric",
]
