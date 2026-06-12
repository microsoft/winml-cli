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


def resolve_composite(model_type: str, task: str) -> CompositeComponents | None:
    """Sub-components of a composite *pipeline* task, else None.

    Exact registration-key lookup (summarization / translation /
    table-question-answering / image-to-text / text-generation). Returns None
    for granular tasks like ``text2text-generation`` — those resolve to a
    single model when requested explicitly. The seq2seq *bridge* (detected
    text2text-generation -> composite) lives in ``_composite_components_for_task``
    and is applied only on the auto-detection path.
    """
    import winml.modelkit.models.hf  # noqa: F401  # trigger composite registrations

    from ..models.winml.composite_model import COMPOSITE_MODEL_REGISTRY

    cls = COMPOSITE_MODEL_REGISTRY.get((model_type, task))
    return dict(cls._SUB_MODEL_CONFIG) if cls is not None else None


# Optimum-canonical generation task that detect-path seq2seq models surface;
# bridged to the model_type's composite. Universal taxonomy, not a model name.
_SEQ2SEQ_GENERATION_TASK = "text2text-generation"


def _composite_components_for_task(model_type: str, task: str) -> CompositeComponents | None:
    """Composite components serving a *detected* task, else None.

    Serves ``task`` when ``task`` is its registration task (qwen3 ->
    text-generation, blip -> image-to-text) OR the seq2seq generation task
    (text2text-generation, what detection yields for t5/bart/marian whose
    composites register under translation/summarization). Candidates deduped
    by export shape; >1 distinct shape -> ambiguous, require explicit --task.
    """
    import winml.modelkit.models.hf  # noqa: F401

    from ..models.winml import WinMLCompositeModel
    from ..models.winml.composite_model import COMPOSITE_MODEL_REGISTRY

    distinct: dict[tuple, type] = {}
    for (m_type, reg_task), cls in COMPOSITE_MODEL_REGISTRY.items():
        if m_type != model_type or not issubclass(cls, WinMLCompositeModel):
            continue
        if task in (reg_task, _SEQ2SEQ_GENERATION_TASK):
            distinct[tuple(sorted(cls._SUB_MODEL_CONFIG.items()))] = cls
    if not distinct:
        return None
    if len(distinct) == 1:
        return dict(next(iter(distinct.values()))._SUB_MODEL_CONFIG)
    tasks = sorted(t for (mt, t) in COMPOSITE_MODEL_REGISTRY if mt == model_type)
    raise ValueError(
        f"{model_type!r} has multiple composite exports; pass --task explicitly (one of: {tasks})."
    )
