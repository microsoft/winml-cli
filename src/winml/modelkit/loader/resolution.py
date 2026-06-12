# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unified task resolution.

Single entry point ``resolve_task`` returns a structured ``TaskResolution``
consumed by every caller (inspect / config / build / eval / inference).
``resolve_composite`` decomposes a pipeline task into its sub-components.

This module owns ALL task-detection logic; ``loader.task`` keeps only the
data tables and boundary utilities (``to_optimum_task``, ``KNOWN_TASKS`` …).
optimum/transformers are imported lazily inside functions so the
``winml inspect --list-tasks`` fast path stays import-cheap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)

# Component-name -> sub-task, e.g. {"encoder": "feature-extraction",
# "decoder": "text2text-generation"} (the composite ``_SUB_MODEL_CONFIG`` shape).
CompositeComponents = dict[str, str]


class TaskSource(str, Enum):
    """How a task was decided. Surfaced by ``winml inspect`` as provenance."""

    USER_TASK = "user-task"  # user passed --task
    USER_CLASS = "user-class"  # user passed --model-class; task inferred
    MODEL_ID_DEFAULT = "model-id-default"  # MODEL_TASK_MAPPING model-id default
    SENTINEL_DEFAULT = "sentinel-default"  # (model_type, None) sentinel
    TASKS_MANAGER = "tasks-manager"  # Optimum inference (incl. fill-mask upgrade)
    WRAPPED_LIBRARY = "wrapped-library"  # no architectures -> first supported task
    HF_TASK_DEFAULT = "hf-task-default"  # last-resort default


@dataclass(frozen=True)
class TaskResolution:
    """Resolved task for a single model.

    ``task`` is WinML modality-aware (user-facing, dataset/eval key);
    ``optimum_task`` is Optimum-canonical (== ``to_optimum_task(task)``) and
    drives export-config + model-class lookup. ``composite`` is set when the
    resolved task bridges to a multi-component pipeline (else ``None``).
    """

    task: str
    optimum_task: str
    model_class: type
    source: TaskSource
    composite: CompositeComponents | None = None
