# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Core inference package — shared by serve and CLI commands.

Public API:
    InferenceEngine       — stateful model loader + HF pipeline runner
    TASK_REGISTRY         — task → input schema + pipeline mapping
    InputField, PipelineMapping, TaskInputSpec — schema dataclasses
    BINARY_TYPES          — frozenset of binary input types
    Prediction, PredictionResult — inference result types
"""

from .engine import InferenceEngine
from .tasks import (
    BINARY_TYPES,
    TASK_REGISTRY,
    InputField,
    PipelineMapping,
    TaskInputSpec,
)
from .types import Prediction, PredictionResult


__all__ = [
    "BINARY_TYPES",
    "TASK_REGISTRY",
    "InferenceEngine",
    "InputField",
    "PipelineMapping",
    "Prediction",
    "PredictionResult",
    "TaskInputSpec",
]
