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

import importlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast

from .task import (
    HF_TASK_DEFAULTS,
    get_default_task_for_model_id,
    get_supported_tasks,
    normalize_task,
    resolve_optimum_library,
    to_optimum_task,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


# =============================================================================
# Task-detection helpers (relocated from loader.task)
# =============================================================================


def _resolve_task_override(model_type_normalized: str, model_id: str | None = None) -> str | None:
    """Return the canonical default task for a model_type / model_id, or ``None``.

    Single source of truth for task overrides, consulted by every detection entry
    point so they all resolve the same default task. Priority:

    1. Model-id default (e.g. ``prajjwal1/bert-tiny`` -> ``feature-extraction``).
    2. ``(model_type, None)`` sentinel in ``MODEL_CLASS_MAPPING``: its value is the
       default *class*; the task is reverse-looked-up from the matching
       ``(model_type, task) -> same class`` entry. Covers families whose canonical
       export differs from the headless TasksManager default (SAM/SAM2 ->
       ``mask-generation``), structurally enforcing that the matching class entry exists.

    A default-task override is declared ONLY by an explicit sentinel — never inferred
    from a model_type happening to have a single ``(model_type, task)`` entry. Such an
    entry exists to fix the *class* for that task (e.g. ``segformer`` image-segmentation),
    not to declare it the default; without a sentinel the architecture head decides, so a
    fine-tuned checkpoint keeps its own task (a segformer classification checkpoint stays
    ``image-classification``).

    Returns ``None`` when there is no model-id default and no sentinel.
    """
    if model_id:
        model_id_task = get_default_task_for_model_id(model_id)
        if model_id_task is not None:
            return model_id_task

    from ..models.hf import MODEL_CLASS_MAPPING

    # (model_type, None) sentinel -> reverse-lookup the task sharing its class.
    default_class = MODEL_CLASS_MAPPING.get((model_type_normalized, None))
    if default_class is None:
        return None
    default_task = next(
        (
            t
            for (mt, t), cls in MODEL_CLASS_MAPPING.items()
            if mt == model_type_normalized and t is not None and cls is default_class
        ),
        None,
    )
    if default_task is None:
        raise ValueError(
            f"MODEL_CLASS_MAPPING has ({model_type_normalized!r}, None) sentinel "
            f"-> {default_class.__name__}, but no matching "
            f"({model_type_normalized!r}, <task>) entry maps to that class. "
            f"Add the corresponding (model_type, task) entry."
        )
    return default_task


def _resolve_model_class_from_config(config: PretrainedConfig) -> type:
    """Extract architecture class from config and import it from transformers.

    Reads ``config.architectures[0]`` and dynamically imports the corresponding
    class from the ``transformers`` package.

    Args:
        config: HuggingFace PretrainedConfig

    Returns:
        The model class (e.g., ``BertForSequenceClassification``)

    Raises:
        ValueError: If ``architectures`` is ``None`` or empty ``[]``,
            or if the class name is not importable from ``transformers``.
    """
    architectures = getattr(config, "architectures", None)
    if not architectures:
        raise ValueError(
            "Cannot detect task: config has no 'architectures' field. "
            "Please specify task explicitly."
        )

    arch_name = architectures[0]
    logger.debug("Resolving model class for architecture: %s", arch_name)

    try:
        transformers_module = importlib.import_module("transformers")
        return cast("type", getattr(transformers_module, arch_name))
    except AttributeError as e:
        raise ValueError(
            f"Cannot import {arch_name} from transformers. Please specify task explicitly."
        ) from e


def _detect_task_from_model_class(model_class: type) -> str:
    """Detect task from a model class via TasksManager.

    One-liner wrapper around ``TasksManager.infer_task_from_model()``.
    Avoids the ``class -> string -> reimport -> class`` round-trip when
    the class is already available.

    Args:
        model_class: A HuggingFace model class (e.g., ``BertForSequenceClassification``)

    Returns:
        Canonical task name (e.g., ``"text-classification"``)
    """
    from optimum.exporters.tasks import TasksManager

    return cast("str", TasksManager.infer_task_from_model(model_class))


def _upgrade_fill_mask_for_seq2seq(task: str, config: PretrainedConfig) -> str:
    """Correct Optimum's ``fill-mask`` mislabel for encoder-decoder generation heads.

    ``TasksManager`` maps some encoder-decoder ``*ForConditionalGeneration`` classes
    (e.g. ``BartForConditionalGeneration``) to ``fill-mask``. A real masked-LM is
    encoder-only, so a config that is ``is_encoder_decoder`` yet reported as
    ``fill-mask`` is actually a seq2seq generator -> ``text2text-generation``.
    Architecture-agnostic: keyed on the ``is_encoder_decoder`` flag, not model names.
    Requires the flag to be explicitly ``True`` (HF configs set a real bool) so a
    partial/duck-typed config without the field is never silently upgraded.
    """
    if task == "fill-mask" and getattr(config, "is_encoder_decoder", False) is True:
        return "text2text-generation"
    return task


# Modality-aware upgrade (D2) for the one modality-blind task, ``feature-extraction``.
# Keyed on the architecture class's ``main_input_name`` — an HF framework convention
# that is authoritative, offline, and architecture-agnostic (``pixel_values`` -> image,
# ``input_ids`` -> text, ``input_values``/``input_features`` -> audio). Only image has a
# downstream (dataset + evaluator) today, so text/audio/video deliberately stay
# ``feature-extraction`` (the Optimum-canonical export task). Extend this table — not the
# code — when a modality gains its downstream.
_FEATURE_MODALITY_BY_MAIN_INPUT: dict[str, str] = {
    "pixel_values": "image-feature-extraction",
}


def _resolve_task_modality(config: PretrainedConfig, task: str) -> str:
    """Upgrade a modality-blind ``feature-extraction`` to its modality-aware variant.

    Reads the *architecture* class's ``main_input_name`` and maps it via
    :data:`_FEATURE_MODALITY_BY_MAIN_INPUT`. Uses ``config.architectures`` (not a
    resolved Auto/wrapper class, whose ``main_input_name`` may be generic) so a ViT
    backbone resolving to a generic ``AutoModel`` still reads ``pixel_values``.

    Applied only to the surfaced/returned task — never to a task headed into an Optimum
    API, which does not recognise modality-aware names like ``image-feature-extraction``.
    Offline; a no-op for non-``feature-extraction`` tasks, for modalities with no
    downstream yet, and when the architecture class cannot be resolved.
    """
    if task != "feature-extraction":
        return task
    try:
        model_class = _resolve_model_class_from_config(config)
    except ValueError:
        return task
    main_input = getattr(model_class, "main_input_name", None)
    if main_input is None:
        return task
    return _FEATURE_MODALITY_BY_MAIN_INPUT.get(main_input, task)


def _get_custom_model_class(model_type: str, task: str) -> type | None:
    """Get model class for a (model_type, task) combination.

    Three-level lookup for model class overrides:

    1. ``MODEL_CLASS_MAPPING[(model_type, task)]`` from ``models/hf/``
       (CLIP, SAM2 specializations)
    2. ``HF_TASK_DEFAULTS[task]`` for unsupported tasks (e.g., NSP)
    3. Return ``None`` -> caller falls back to TasksManager

    Args:
        model_type: HuggingFace model type (e.g., ``"clip"``, ``"sam2_video"``).
        task: Task name (e.g., ``"feature-extraction"``, ``"image-segmentation"``).

    Returns:
        Model class, or ``None`` if TasksManager default should be used.
    """
    # Normalize model_type (handle underscores, case)
    model_type_normalized = model_type.lower().replace("_", "-")

    # Lazy import to avoid circular imports
    from ..models.hf import MODEL_CLASS_MAPPING

    key = (model_type_normalized, task)
    if key in MODEL_CLASS_MAPPING:
        return MODEL_CLASS_MAPPING[key]

    # Task defaults (for tasks TasksManager doesn't support, e.g., NSP)
    if task in HF_TASK_DEFAULTS:
        import transformers

        return cast("type", getattr(transformers, HF_TASK_DEFAULTS[task]))

    return None

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


def _infer_task_from_architecture(config: PretrainedConfig) -> str:
    """Optimum task inferred from ``config.architectures[0]``.

    Includes the encoder-decoder fill-mask -> text2text-generation correction.
    """
    return _upgrade_fill_mask_for_seq2seq(
        _detect_task_from_model_class(_resolve_model_class_from_config(config)),
        config,
    )


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

    distinct: dict[tuple, type[WinMLCompositeModel]] = {}
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


def resolve_task(
    config: PretrainedConfig,
    *,
    task: str | None = None,
    model_class: str | None = None,
) -> TaskResolution:
    """Resolve a single model's task + class from an HF config.

    Stages: 0 user override -> 1 detect (override / no-architectures /
    TasksManager / default) -> 2 model class -> 3 modality upgrade
    (detection path only) -> 4 composite tag.
    """
    from optimum.exporters.tasks import TasksManager

    model_type = getattr(config, "model_type", None)
    model_type_norm = model_type.lower().replace("_", "-") if model_type else ""
    model_id = getattr(config, "_name_or_path", "") or None

    # Declared once up front so the Stage-0 branches (which assign a concrete str)
    # and the Stage-1 detection (which starts at None) share one str | None type.
    opt_task: str | None = None

    # --- Stage 0: user override (short-circuits detection) ----------------
    if model_class is not None:
        if task is not None:
            # USER_CLASS with an explicit task does two separable things to the surfaced
            # task: (a) canonicalize the alias for the Optimum class lookup
            # (masked-lm -> fill-mask) and (b) re-apply modality so a WinML modality-aware
            # name survives (feature-extraction -> image-feature-extraction for a
            # pixel_values arch). (b) is a no-op for non-feature-extraction tasks, so (a)
            # is preserved. Consistent with the inferred branch below and USER_TASK —
            # adding --model-class must not collapse the modality.
            opt_task = normalize_task(task)
            surfaced = _resolve_task_modality(config, opt_task)
        else:
            # Task inferred from the architecture: surface it modality-aware, consistent
            # with the detection path (Stage 3), so e.g. a ViT backbone is
            # image-feature-extraction rather than the modality-blind feature-extraction.
            opt_task = _infer_task_from_architecture(config)
            surfaced = _resolve_task_modality(config, opt_task)
        try:
            resolved = TasksManager.get_model_class_for_task(
                opt_task, framework="pt", model_class_name=model_class
            )
        except (KeyError, AttributeError) as e:
            raise ValueError(
                f"Model class '{model_class}' not found for task '{opt_task}'. "
                f"Check that the class name is correct and available in transformers."
            ) from e
        return TaskResolution(
            surfaced, to_optimum_task(surfaced), resolved, TaskSource.USER_CLASS, None
        )

    if task is not None:
        original = task
        normalized = normalize_task(task)
        resolved = None
        if model_type_norm:
            resolved = _get_custom_model_class(
                model_type_norm, original
            ) or _get_custom_model_class(model_type_norm, normalized)
        if resolved is None:
            try:
                resolved = TasksManager.get_model_class_for_task(normalized, framework="pt")
            except KeyError as e:
                raise ValueError(
                    f"Task '{normalized}' not supported by TasksManager. "
                    f"Check optimum documentation for supported tasks."
                ) from e
        return TaskResolution(
            original, to_optimum_task(original), resolved, TaskSource.USER_TASK, None
        )

    # --- Stage 1: detection -----------------------------------------------
    # opt_task stays at its hoisted None until a detection sub-stage sets it.
    source: TaskSource | None = None
    resolved = None

    # 1a. canonical override (model-id default / sentinel)
    override = _resolve_task_override(model_type_norm, model_id)
    if override is not None:
        opt_task = override
        source = (
            TaskSource.MODEL_ID_DEFAULT
            if model_id and get_default_task_for_model_id(model_id) is not None
            else TaskSource.SENTINEL_DEFAULT
        )

    # 1b. no architectures -> first ONNX-exportable task
    #     (merges the old timm wrapped-library stage AND the --model-type fallback)
    if opt_task is None and not getattr(config, "architectures", None) and model_type:
        # Populate Optimum's ONNX export-config registry before querying it;
        # get_supported_tasks returns [] if this hasn't been imported.
        import optimum.exporters.onnx.model_configs  # noqa: F401

        supported = get_supported_tasks(model_type, resolve_optimum_library(model_type))
        if supported:
            opt_task = supported[0]
            source = TaskSource.WRAPPED_LIBRARY
            # The model class is resolved uniformly in Stage 2 (under its try/except), so a
            # lookup failure here — e.g. a wrapped library whose classes aren't registered
            # under framework="pt" — can't escape as a raw KeyError.

    # 1c. TasksManager (reads config.architectures)
    if opt_task is None:
        try:
            opt_task = _infer_task_from_architecture(config)
            source = TaskSource.TASKS_MANAGER
        except ValueError:
            opt_task = None

    # 1d. last-resort default
    if opt_task is None:
        opt_task = next(iter(HF_TASK_DEFAULTS))
        source = TaskSource.HF_TASK_DEFAULT

    # --- Stage 2: model class (if not already resolved in 1b) -------------
    if resolved is None:
        resolved = _get_custom_model_class(model_type_norm, opt_task)
        if resolved is None:
            try:
                resolved = TasksManager.get_model_class_for_task(opt_task, framework="pt")
            except Exception:
                resolved = _resolve_model_class_from_config(config)  # arch fallback

    # --- Stage 3: modality upgrade (surfaced task only) -------------------
    surfaced = _resolve_task_modality(config, opt_task)

    # --- Stage 4: composite tag (detection path) --------------------------
    composite = _composite_components_for_task(model_type, opt_task) if model_type else None

    if source is None:  # structural invariant: Stage 1d always sets a source
        raise RuntimeError("resolve_task: internal invariant violated — source was not set")
    return TaskResolution(surfaced, to_optimum_task(surfaced), resolved, source, composite)
