# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Task-detection data tables and boundary utilities.

Holds the static task taxonomy (``KNOWN_TASKS``, ``TASK_ABBREV``,
``HF_TASK_DEFAULTS``, ``MODEL_TASK_MAPPING``, ``WRAPPED_LIBRARY_MODEL_TYPES``,
``TASK_SYNONYM_EXTENSIONS``) plus the WinML <-> Optimum boundary helpers. The
actual task-resolution logic lives in ``loader.resolution``.

Public API:
    resolve_optimum_library      - Route a model_type to the Optimum export library
    normalize_task               - Map task aliases to canonical names
    to_optimum_task              - Collapse a WinMLTask to its Optimum-canonical form
    get_task_abbrev              - Abbreviated task name for cache keys
    get_supported_tasks          - List ONNX-exportable tasks for a model type
    get_default_task_for_model_id - Model-id default task lookup
"""

from __future__ import annotations

import logging
from typing import cast


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

    return cast("str", TasksManager.map_from_synonym(task))


# WinML task-synonym extensions — extend Optimum's ``TasksManager.map_from_synonym``
# for tasks it does not recognize or mis-maps. Entries here take priority over Optimum.
TASK_SYNONYM_EXTENSIONS: dict[str, str] = {
    # NOTE: do NOT add "image-feature-extraction" here. This set is also consulted by
    # commands.build._validate_task_supported_for_model (its "WinML extension" branch),
    # so adding it would silence the cross-modality visibility warning. Its Optimum-synonym
    # collapse to feature-extraction is guarded by tests/unit/loader/test_task_boundary.py.
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

    return cast("str", TasksManager.map_from_synonym(task))


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
