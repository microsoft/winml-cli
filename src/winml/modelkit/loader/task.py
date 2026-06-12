# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Task detection and utilities using Optimum's TasksManager.

Uses TasksManager.infer_task_from_model() as PRIMARY approach per design spec.

Public API:
    resolve_task_and_model_class  - Main orchestrator (3 resolution cases)
    resolve_optimum_library      - Route a model_type to the Optimum export library
    normalize_task               - Map task aliases to canonical names
    to_optimum_task              - Collapse a WinMLTask to its Optimum-canonical form
    get_task_abbrev              - Abbreviated task name for cache keys
    get_supported_tasks          - List ONNX-exportable tasks for a model type

Internal:
    _resolve_model_class_from_config  - Extract + import class from config.architectures
    _detect_task_from_model_class     - Infer task from a model class via TasksManager
    _detect_task_from_config          - Compose the two above
    _detect_task_and_class_from_config - Full auto-detection with specialization lookup
    _get_custom_model_class           - Three-level model class override lookup
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

# Task abbreviations for cache keys (47 tasks from HuggingFace Transformers)
TASK_ABBREV: dict[str, str] = {
    # Vision tasks
    "image-classification": "imgcls",
    "image-segmentation": "imgseg",
    "image-feature-extraction": "imgfeat",
    "image-to-image": "img2img",
    "image-to-text": "img2txt",
    "image-text-to-text": "imgtxt2t",
    "object-detection": "objdet",
    "depth-estimation": "depth",
    "instance-segmentation": "instseg",
    "semantic-segmentation": "semseg",
    "universal-segmentation": "uniseg",
    "keypoint-detection": "kptdet",
    "keypoint-matching": "kptmtch",
    "mask-generation": "maskgen",
    "masked-image-modeling": "mskim",
    "video-classification": "vidcls",
    "zero-shot-image-classification": "zsimg",
    "zero-shot-object-detection": "zsobj",
    # NLP tasks
    "text-classification": "txtcls",
    "sequence-classification": "seqcls",
    "token-classification": "tokcls",
    "question-answering": "qa",
    "text-generation": "txtgen",
    "text2text-generation": "txt2txt",
    "fill-mask": "mask",
    "feature-extraction": "feat",
    "text-encoding": "txtenc",
    "summarization": "summ",
    "translation": "transl",
    "multiple-choice": "mltchs",
    "next-sentence-prediction": "nsp",
    "pretraining": "pretrain",
    "table-question-answering": "tabqa",
    "document-question-answering": "docqa",
    "zero-shot-classification": "zscls",
    # Audio tasks
    "audio-classification": "audiocls",
    "audio-frame-classification": "audfrm",
    "audio-tokenization": "audtok",
    "audio-xvector": "audxvc",
    "automatic-speech-recognition": "asr",
    "text-to-audio": "txt2aud",
    "zero-shot-audio-classification": "zsaud",
    # Multimodal tasks
    "visual-question-answering": "vqa",
    "any-to-any": "a2a",
    "multimodal-lm": "mmlm",
    # Other tasks
    "backbone": "bkbone",
    "time-series-prediction": "tseries",
}


# Canonical set of task names recognized by `winml inspect`.
# Hand-coded so that `winml inspect --list-tasks` does not need to import
# optimum.exporters (which transitively imports transformers and costs ~10s).
# Synced with optimum.exporters.tasks.TasksManager.get_all_tasks() plus our
# own HF_TASK_DEFAULTS entries; add new tasks here when optimum gains them.
KNOWN_TASKS: frozenset[str] = frozenset(
    {
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
        "automatic-speech-recognition",
        "depth-estimation",
        "document-question-answering",
        "feature-extraction",
        "fill-mask",
        "image-classification",
        "image-feature-extraction",
        "image-segmentation",
        "image-text-to-text",
        "image-to-image",
        "image-to-text",
        "inpainting",
        "keypoint-detection",
        "mask-generation",
        "masked-im",
        "multiple-choice",
        "next-sentence-prediction",
        "object-detection",
        "question-answering",
        "reinforcement-learning",
        "semantic-segmentation",
        "sentence-similarity",
        "text-classification",
        "text-generation",
        "text-to-audio",
        "text-to-image",
        "text2text-generation",
        "time-series-forecasting",
        "token-classification",
        "visual-question-answering",
        "zero-shot-image-classification",
        "zero-shot-object-detection",
    }
)


# =============================================================================
# Model Class Resolution
# Lookup: MODEL_CLASS_MAPPING -> HF_TASK_DEFAULTS -> TasksManager default
# =============================================================================

# Task defaults for tasks NOT in TasksManager
# task -> HuggingFace model class NAME
HF_TASK_DEFAULTS: dict[str, str] = {
    # Tasks not supported by optimum.exporters.tasks.TasksManager
    "next-sentence-prediction": "AutoModelForNextSentencePrediction",
}

# Model-specific task defaults for known model IDs that need explicit routing
# when users do not pass --task or model_class.
# Sentinel key (model_id, None) mirrors MODEL_CLASS_MAPPING's default pattern.
MODEL_TASK_MAPPING: dict[tuple[str, str | None], str] = {
    ("prajjwal1/bert-tiny", None): "feature-extraction",
}

# Some transformers model_types are generic wrappers that expose an entire other
# library through a single type (e.g. timm via "timm_wrapper"). Such configs
# carry no `architectures` field, and their Optimum ONNX export config is
# registered under the wrapped library, not "transformers". This is a
# library-routing concern handled at the common resolution layer (the loader
# below and export.io._get_onnx_config), not a per-model OnnxConfig.
#
# Only the library is recorded here -- it is the irreducible Optimum-taxonomy
# fact. The export task is derived from Optimum's task list for that library
# (get_supported_tasks), not hardcoded.
# model_type -> optimum_library
WRAPPED_LIBRARY_MODEL_TYPES: dict[str, str] = {
    "timm_wrapper": "timm",
}


def resolve_optimum_library(model_type: str | None, library_name: str = "transformers") -> str:
    """Route a transformers model_type to the Optimum library that owns its export.

    Most models export under the library they were requested with. A few
    transformers model_types are thin wrappers whose Optimum OnnxConfig lives in
    another library (see :data:`WRAPPED_LIBRARY_MODEL_TYPES`); route those so the
    OnnxConfig lookup succeeds without an explicit ``--library`` flag.

    Only the ``"transformers"`` library is rerouted, so an explicit
    non-``"transformers"`` library is returned unchanged. (An explicit
    ``--library transformers`` is indistinguishable from the default and is
    still rerouted for wrapped types -- harmless, since those types have no
    OnnxConfig registered under transformers anyway.)
    """
    if library_name == "transformers" and model_type in WRAPPED_LIBRARY_MODEL_TYPES:
        return WRAPPED_LIBRARY_MODEL_TYPES[model_type]
    return library_name


# =============================================================================
# Internal Helpers
# =============================================================================


def get_default_task_for_model_id(model_name_or_path: str) -> str | None:
    """Get model-specific default task for a model ID/path if configured."""
    model_id = model_name_or_path.strip().lower()
    return MODEL_TASK_MAPPING.get((model_id, None))


def _resolve_task_override(model_type_normalized: str, model_id: str | None = None) -> str | None:
    """Return the canonical default task for a model_type / model_id, or ``None``.

    Single source of truth for task overrides, consulted by every detection entry
    point (``detect_task``, ``_detect_task_and_class_from_config``,
    ``resolve_loader_config``) so they all resolve the same default task. Priority:

    1. Model-id default (e.g. ``prajjwal1/bert-tiny`` -> ``feature-extraction``).
    2. ``(model_type, None)`` sentinel in ``MODEL_CLASS_MAPPING``: its value is the
       default *class*; the task is reverse-looked-up from the matching
       ``(model_type, task) -> same class`` entry. Covers multi-task families whose
       canonical export differs from the headless TasksManager default
       (SAM/SAM2 -> ``mask-generation``), structurally enforcing that the matching
       class entry exists.
    3. Exactly one real (non-``None``) task for the model_type -> that task.

    Returns ``None`` when the model_type is unregistered or maps to several real
    tasks with no sentinel (ambiguous -> the architecture head decides).
    """
    if model_id:
        model_id_task = get_default_task_for_model_id(model_id)
        if model_id_task is not None:
            return model_id_task

    from ..models.hf import MODEL_CLASS_MAPPING

    # (model_type, None) sentinel -> reverse-lookup the task sharing its class.
    default_class = MODEL_CLASS_MAPPING.get((model_type_normalized, None))
    if default_class is not None:
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

    # Exactly one real task -> unambiguous default (the former detect_task short-circuit).
    distinct_tasks = {
        mapped
        for (mt, mapped) in MODEL_CLASS_MAPPING
        if mt == model_type_normalized and mapped is not None
    }
    if len(distinct_tasks) == 1:
        return next(iter(distinct_tasks))

    return None


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
        return getattr(transformers_module, arch_name)
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

    return TasksManager.infer_task_from_model(model_class)


def _detect_task_from_config(config: PretrainedConfig) -> str:
    """Detect task from HF config using TasksManager PUBLIC API.

    Composes ``_resolve_model_class_from_config`` + ``_detect_task_from_model_class``.

    Args:
        config: HuggingFace PretrainedConfig

    Returns:
        Canonical task name (e.g., ``"image-classification"``)

    Raises:
        ValueError: If task cannot be detected
    """
    model_class = _resolve_model_class_from_config(config)
    task = _detect_task_from_model_class(model_class)
    task = _upgrade_fill_mask_for_seq2seq(task, config)
    logger.info("Detected task: %s (from %s)", task, model_class.__name__)
    return task


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


# Data-driven task-modality disambiguation (D2). Maps a modality-blind task to its
# modality-aware variants, each keyed by the top-level config fields that signal that
# modality. Extend this table — not the code — to add new modalities. First match wins.
_TASK_MODALITY_DISAMBIGUATION: dict[str, dict[str, tuple[str, ...]]] = {
    "feature-extraction": {
        # Vision backbones (ViT, DINOv2, ConvNeXt, …) carry image_size/patch_size at the
        # config root; multimodal models (CLIP) nest them under vision_config, so the
        # top-level check does not fire for those.
        "image-feature-extraction": ("image_size", "patch_size"),
        # Future, when supported: "audio-feature-extraction": ("sampling_rate", ...),
    },
}


def _top_level_config_keys(config: PretrainedConfig) -> set[str]:
    """Top-level field names of an HF config (nested sub-configs are not flattened)."""
    try:
        return set(config.to_dict().keys())
    except Exception:
        return set(vars(config).keys())


def _resolve_task_modality(config: PretrainedConfig, task: str) -> str:
    """Disambiguate a modality-blind task using top-level config fields (D2).

    Data-driven via :data:`_TASK_MODALITY_DISAMBIGUATION`. Applied only to surfaced/
    returned tasks — never to a task headed into an Optimum API, which does not
    recognise modality-aware names like ``image-feature-extraction``.
    """
    candidates = _TASK_MODALITY_DISAMBIGUATION.get(task)
    if not candidates:
        return task
    keys = _top_level_config_keys(config)
    for modality_task, signal_fields in candidates.items():
        if any(field in keys for field in signal_fields):
            return modality_task
    return task


def detect_task(config: PretrainedConfig) -> tuple[str, str]:
    """Single offline detection entry. Returns ``(WinMLTask, source)``.

    ``WinMLTask`` is HF modality-aware (e.g. ``image-feature-extraction``) — the
    only behavioural difference from :func:`_detect_task_from_config`, which stays
    Optimum-canonical for internal model-class resolution.

    Dispatch order mirrors the historical inspect resolver::

        HF_MODEL_CLASS_MAPPING -> wrapped-library -> TasksManager -> HF_TASK_DEFAULTS

    The D2 vision-modality upgrade is applied to the **returned** task only; no
    Optimum API ever receives ``image-feature-extraction``. Offline / config-only —
    no network.
    """
    model_type = getattr(config, "model_type", "unknown")
    model_type_normalized = model_type.lower().replace("_", "-")
    model_id = getattr(config, "_name_or_path", "") or None

    task: str | None = None
    source = "none"

    # 1. Canonical task override — model-id default, the (model_type, None) sentinel, or a
    #    model_type that maps to exactly one real task. Single source of truth shared with
    #    the build path (_resolve_task_override), so inspect/eval and config/build resolve
    #    the same task. A multi-task model_type with no sentinel returns None here, and the
    #    architecture head decides in step 3.
    override_task = _resolve_task_override(model_type_normalized, model_id)
    if override_task is not None:
        task, source = override_task, "HF_MODEL_CLASS_MAPPING"

    # 2. Wrapped-library model types (e.g. timm via "timm_wrapper") carry no
    #    `architectures`; resolve through their wrapped library instead of the
    #    HF_TASK_DEFAULTS mislabel below. Use the raw model_type for the lookup.
    if task is None and (
        model_type in WRAPPED_LIBRARY_MODEL_TYPES and not getattr(config, "architectures", None)
    ):
        try:
            task, _ = _detect_task_and_class_from_config(config)
            source = "wrapped-library"
        except Exception:
            logger.debug("wrapped-library task detection failed for %s", model_type, exc_info=True)

    # 3. TasksManager (Optimum) detection.
    if task is None:
        try:
            task = _detect_task_from_config(config)
            source = "TasksManager"
        except ValueError:
            # TasksManager can't infer a task (e.g. no recognizable architecture);
            # leave task unset and fall through to the HF_TASK_DEFAULTS fallback below.
            pass

    # 4. Fallback to task defaults.
    if task is None:
        if not HF_TASK_DEFAULTS:
            return "unknown", "none"
        task, source = next(iter(HF_TASK_DEFAULTS.keys())), "HF_TASK_DEFAULTS"

    # D2 — vision modality upgrade, applied to the surfaced task only.
    return _resolve_task_modality(config, task), source


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

        return getattr(transformers, HF_TASK_DEFAULTS[task])

    return None


def _detect_task_and_class_from_config(config: PretrainedConfig) -> tuple[str, type]:
    """Detect both task and model class from HF config.

    Full auto-detection with specialization lookup.
    Called by ``resolve_task_and_model_class`` Case 1.

    Resolution flow:
    1. ``_resolve_model_class_from_config(config)`` -> arch_model_class
    2. ``_detect_task_from_model_class(arch_model_class)`` -> task
    3. ``_get_custom_model_class(model_type, task)`` -> specialization check
    4. If specialization found -> return (task, specialized_class)
    5. Else ``TasksManager.get_model_class_for_task(task)`` -> tm_class
    6. If TasksManager fails -> fallback to arch_model_class

    Args:
        config: HuggingFace PretrainedConfig

    Returns:
        Tuple of (task, model_class)

    Raises:
        ValueError: If task cannot be detected or model_type is missing
    """
    from optimum.exporters.tasks import TasksManager

    # [0] Canonical task override — model-id default, the (model_type, None) sentinel, or a
    # single-real-task model_type. Single source of truth shared with detect_task so
    # config/build and inspect/eval resolve the same task. Applies before architecture
    # resolution, so models with no architectures (model-id default) and multi-task families
    # whose canonical export differs from the headless default (SAM/SAM2 -> mask-generation)
    # both resolve here without guessing from the arch head.
    model_id = getattr(config, "_name_or_path", "") or None
    model_type_for_override = getattr(config, "model_type", None)
    model_type_normalized = (
        model_type_for_override.lower().replace("_", "-") if model_type_for_override else ""
    )
    override_task = _resolve_task_override(model_type_normalized, model_id)
    if override_task is not None:
        logger.info(
            "Using task override %s for %s", override_task, model_id or model_type_for_override
        )
        return resolve_task_and_model_class(config, task=override_task)

    # [1] Resolve architecture class from config.
    # Some model_types (e.g. timm via "timm_wrapper") are generic library
    # wrappers that carry no `architectures` field. Resolve those through their
    # wrapped library: the task comes from Optimum's task list for that library
    # (not hardcoded), and the class from get_model_class_for_task (a generic
    # Auto* class that transformers dispatches to the wrapper at load).
    if not getattr(config, "architectures", None):
        model_type = getattr(config, "model_type", None)
        library = WRAPPED_LIBRARY_MODEL_TYPES.get(model_type) if model_type else None
        if library is not None:
            # Populate Optimum's exporter registry (incl. the wrapped library's
            # task list) before querying it; scoped to this rare branch so normal
            # model loading never pays for the import.
            import optimum.exporters.onnx.model_configs  # noqa: F401

            supported = get_supported_tasks(model_type, library_name=library)
            if supported:
                # A wrapped library exposes a single ONNX export task today
                # (timm -> "image-classification"), so supported[0] is the right
                # default. If one ever exposes multiple, supported[0] is an
                # arbitrary pick -- warn (listing the tasks) but still proceed;
                # pass --task to choose a different one.
                # No _upgrade_fill_mask_for_seq2seq here: this task comes from the
                # wrapped library's ONNX export list (get_supported_tasks), not from
                # class->task inference, so the optimum fill-mask mislabel cannot
                # arise on this path; rewriting could also yield a task outside that
                # list. The correction is intentionally scoped to the class->task paths.
                task = supported[0]
                if len(supported) > 1:
                    logger.warning(
                        "config has no 'architectures' and the %s library exposes "
                        "multiple export tasks for %s %s; defaulting to %r "
                        "(pass --task to choose another).",
                        library,
                        model_type,
                        supported,
                        task,
                    )
                model_class = TasksManager.get_model_class_for_task(task, framework="pt")
                logger.info(
                    "config has no 'architectures'; resolved %s via %s library (task=%s, class=%s)",
                    model_type,
                    library,
                    task,
                    model_class.__name__,
                )
                return task, model_class
    # If config.architectures is still missing/empty, this raises ValueError and
    # the caller should provide task explicitly.
    arch_model_class = _resolve_model_class_from_config(config)
    arch_name = arch_model_class.__name__

    # [2] Infer task from model class
    task = _detect_task_from_model_class(arch_model_class)
    task = _upgrade_fill_mask_for_seq2seq(task, config)
    logger.info("Detected task: %s (from %s)", task, arch_name)

    # [3] Get model_type - REQUIRED for specialization lookup
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        raise ValueError(
            "Cannot resolve model class: config has no 'model_type' field. "
            "Please specify model_class explicitly."
        )

    # [4] Check specializations first (CLIP, SAM2, etc.) - highest priority
    model_class = _get_custom_model_class(model_type, task)
    if model_class:
        logger.info("Using specialized model class: %s", model_class.__name__)
        return task, model_class

    # [5] Try TasksManager, fallback to arch_model_class on failure
    try:
        model_class = TasksManager.get_model_class_for_task(task)

        # TasksManager may return a generic AutoModel* class that differs from
        # config.architectures. Surface that choice because it can affect export.
        if model_class.__name__ != arch_name:
            logger.info(
                "TasksManager returned %s, but config.architectures specifies %s. "
                "Honoring TasksManager's choice.",
                model_class.__name__,
                arch_name,
            )
        else:
            logger.debug("Using TasksManager model class: %s", model_class.__name__)

    except Exception:
        # [6] TasksManager failed - fallback to architecture from config
        logger.info(
            "TasksManager does not support %s/%s, falling back to architecture class: %s",
            model_type,
            task,
            arch_name,
        )
        model_class = arch_model_class

    return task, model_class


# =============================================================================
# Public API
# =============================================================================


def normalize_task(task: str) -> str:
    """Normalize task name using TasksManager's synonym mapping.

    Handles aliases like:

    - ``"causal-lm"`` -> ``"text-generation"``
    - ``"seq2seq-lm"`` -> ``"text2text-generation"``
    - ``"masked-lm"`` -> ``"fill-mask"``

    Unknown task names are returned unchanged (passthrough behavior).
    ``normalize_task("my-custom-task")`` returns ``"my-custom-task"``.

    Args:
        task: User-provided task name (may be alias)

    Returns:
        Canonical task name
    """
    from optimum.exporters.tasks import TasksManager

    return TasksManager.map_from_synonym(task)


# WinML task-synonym extensions — extend Optimum's ``TasksManager.map_from_synonym``
# for tasks it does not recognize or mis-maps. Entries here take priority over Optimum.
TASK_SYNONYM_EXTENSIONS: dict[str, str] = {
    # next-sentence-prediction has the same I/O as text-classification: input_ids -> logits
    "next-sentence-prediction": "text-classification",
    # mask-generation is registered via register_onnx_overwrite for SAM2.
    # Optimum incorrectly maps it to "feature-extraction"; preserve as-is.
    "mask-generation": "mask-generation",
}


def to_optimum_task(task: str) -> str:
    """Map a task name to its Optimum-canonical form, extending Optimum's synonyms.

    This is the single WinML -> Optimum boundary translation: call it only at the
    moment of an Optimum API call (e.g. ``TasksManager.get_exporter_config_constructor``).
    The result is lossy — modality-aware names collapse
    (``image-feature-extraction`` -> ``feature-extraction``).

    WinML extensions in ``TASK_SYNONYM_EXTENSIONS`` take priority and short-circuit
    before Optimum, which may otherwise mis-normalize custom-registered tasks such as
    ``mask-generation``.

    Args:
        task: Task name (a WinMLTask or an alias).

    Returns:
        Optimum-canonical task name.
    """
    if task in TASK_SYNONYM_EXTENSIONS:
        return TASK_SYNONYM_EXTENSIONS[task]

    from optimum.exporters.tasks import TasksManager

    return TasksManager.map_from_synonym(task)


def get_task_abbrev(task: str) -> str:
    """Get abbreviated task name for cache keys.

    Tasks not in ``TASK_ABBREV`` are truncated to first 8 characters.
    ``get_task_abbrev("my-custom-task")`` returns ``"my-custo"``.

    Args:
        task: Canonical task name (e.g., ``"image-classification"``)

    Returns:
        Abbreviated task name (e.g., ``"imgcls"``)
    """
    return TASK_ABBREV.get(task, task[:8])


def resolve_task_and_model_class(
    config: PretrainedConfig,
    task: str | None = None,
    model_class: str | None = None,
) -> tuple[str, type]:
    """Resolve task and model class based on user inputs.

    The main orchestrator. Three resolution cases:

    1. ``task=None, model_class=None``:
       Auto-detect both from config via ``_detect_task_and_class_from_config``

    2. ``task!=None, model_class=None``:
       User specified task only, resolve model class for that task.
       Checks specializations (CLIP, NSP, etc.) first with double lookup
       (original_task then normalized_task) to prevent task collapsing.

    3. ``model_class!=None``:
       Honor user override. Detect task if not provided.

    Args:
        config: HuggingFace PretrainedConfig
        task: Optional task name (auto-detected if None)
        model_class: Optional model class name to override auto-detection

    Returns:
        Tuple of (task, resolved_class)

    Raises:
        ValueError: If task cannot be detected or model_class not found

    Example:
        >>> config = AutoConfig.from_pretrained("microsoft/resnet-50")
        >>> task, resolved_class = resolve_task_and_model_class(config)
        >>> # task = "image-classification", resolved_class = AutoModelForImageClassification

        >>> # With explicit task for CLIP
        >>> config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
        >>> task, resolved_class = resolve_task_and_model_class(
        ...     config, task="image-feature-extraction"
        ... )
        >>> # resolved_class = CLIPVisionModelWithProjection (from specializations)

        >>> # With explicit model_class
        >>> task, resolved_class = resolve_task_and_model_class(
        ...     config, model_class="CLIPTextModel"
        ... )
        >>> # resolved_class = CLIPTextModel (user override honored)
    """
    from optimum.exporters.tasks import TasksManager

    model_type = getattr(config, "model_type", None)

    # Case 1: Auto-detect both task and model class
    if task is None and model_class is None:
        # Resolve task + class from config, then surface the modality-aware task
        # (D2). The class was resolved from the pre-upgrade Optimum task, so model
        # loading is unchanged. Case 2/3 are intentionally left untouched.
        detected_task, resolved_class = _detect_task_and_class_from_config(config)
        return _resolve_task_modality(config, detected_task), resolved_class

    # Case 2: User specified task only -> resolve model class for that task
    if task is not None and model_class is None:
        # Store original task before normalization (for specialization lookup)
        original_task = task
        task = normalize_task(task)

        # Check our specializations first (CLIP, SAM2, etc.)
        # Double lookup: original_task then normalized_task
        # Prevents "image-feature-extraction" from collapsing to "feature-extraction"
        resolved_class = None
        if model_type:
            resolved_class = _get_custom_model_class(model_type, original_task)
            if resolved_class is None:
                resolved_class = _get_custom_model_class(model_type, task)

        try:
            if resolved_class:
                logger.info(
                    "Using specialized model class: %s (for %s/%s)",
                    resolved_class.__name__,
                    model_type,
                    task,
                )
            else:
                resolved_class = TasksManager.get_model_class_for_task(task, framework="pt")
                logger.debug("Using TasksManager default: %s", resolved_class.__name__)
        except KeyError as e:
            raise ValueError(
                f"Task '{task}' not supported by TasksManager. "
                f"Check optimum documentation for supported tasks."
            ) from e

        # Return original_task (not normalized) to preserve user intent
        # e.g., "image-feature-extraction" stays as-is for dataset lookup
        return original_task, resolved_class

    # Case 3: User specified model_class -> honor it!
    # model_class is not None
    # If task not provided, detect it; otherwise normalize
    task = _detect_task_from_config(config) if task is None else normalize_task(task)

    try:
        resolved_class = TasksManager.get_model_class_for_task(
            task,
            framework="pt",
            model_class_name=model_class,
        )
        logger.info("Using user-specified model class: %s", model_class)
    except (KeyError, AttributeError) as e:
        raise ValueError(
            f"Model class '{model_class}' not found for task '{task}'. "
            f"Check that the class name is correct and available in transformers."
        ) from e

    # TODO: Validate task + model_class compatibility
    # inferred_task = _detect_task_from_model_class(resolved_class)
    # if inferred_task != task:
    #     logger.warning(
    #         "model_class '%s' infers task '%s' but requested task is '%s'.",
    #         model_class, inferred_task, task,
    #     )

    return task, resolved_class


def get_supported_tasks(
    model_type: str,
    library_name: str = "transformers",
) -> list[str]:
    """Get list of ONNX-exportable tasks for a model type.

    Queries ``TasksManager.get_supported_tasks_for_model_type()`` directly.
    No network access needed — works with just a model type string.

    Args:
        model_type: HuggingFace model type (e.g., ``"bert"``, ``"segformer"``,
            ``"gpt2"``). This is the ``model_type`` field from HF configs.
        library_name: Source library (default: ``"transformers"``).
            Also supports ``"diffusers"``, ``"timm"``, ``"sentence_transformers"``.

    Returns:
        List of supported task names, or empty list if model type is unknown.

    Example:
        >>> get_supported_tasks("segformer")
        ['feature-extraction', 'image-classification', 'image-segmentation', ...]
        >>> get_supported_tasks("bert")
        ['feature-extraction', 'fill-mask', 'multiple-choice', ...]
        >>> get_supported_tasks("nonexistent")
        []
    """
    from optimum.exporters.tasks import TasksManager

    try:
        tasks = TasksManager.get_supported_tasks_for_model_type(
            model_type,
            exporter="onnx",
            library_name=library_name,
        )
        return list(tasks.keys()) if isinstance(tasks, dict) else list(tasks)
    except Exception:
        return []
