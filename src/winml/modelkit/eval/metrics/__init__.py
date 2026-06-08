# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Evaluation metrics."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .classification import ClassificationMetric
    from .depth import DepthMetric
    from .knn_accuracy import KNNAccuracyMetric
    from .mean_average_precision import MAPMetric
    from .mean_iou import IGNORE_INDEX, MeanIoUMetric
    from .pseudo_perplexity import PseudoPerplexityMetric
    from .spearman_correlation import SpearmanCorrelationMetric
    from .top_k_accuracy import TopKAccuracyMetric


# All metric classes are loaded on first attribute access so that importing
# this package does not pull in numpy / scipy / torch / torchmetrics for callers
# that do not actually use the metric in question.
_LAZY_ATTRS: dict[str, str] = {
    "ClassificationMetric": ".classification:ClassificationMetric",
    "DepthMetric": ".depth:DepthMetric",
    "IGNORE_INDEX": ".mean_iou:IGNORE_INDEX",
    "KNNAccuracyMetric": ".knn_accuracy:KNNAccuracyMetric",
    "MAPMetric": ".mean_average_precision:MAPMetric",
    "MeanIoUMetric": ".mean_iou:MeanIoUMetric",
    "PseudoPerplexityMetric": ".pseudo_perplexity:PseudoPerplexityMetric",
    "SpearmanCorrelationMetric": ".spearman_correlation:SpearmanCorrelationMetric",
    "TopKAccuracyMetric": ".top_k_accuracy:TopKAccuracyMetric",
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
    "DepthMetric",
    "KNNAccuracyMetric",
    "MAPMetric",
    "MeanIoUMetric",
    "PseudoPerplexityMetric",
    "SpearmanCorrelationMetric",
    "TopKAccuracyMetric",
]
