# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Task detection and utilities using Optimum's TasksManager.

Uses TasksManager.infer_task_from_model() as PRIMARY approach per design spec.

Public API:
    resolve_task_and_model_class  - Main orchestrator (3 resolution cases)
    normalize_task               - Map task aliases to canonical names
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


# =============================================================================
# Internal Helpers
# =============================================================================


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
    logger.info("Detected task: %s (from %s)", task, model_class.__name__)
    return task


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

    # [1] Resolve architecture class from config
    arch_model_class = _resolve_model_class_from_config(config)
    arch_name = arch_model_class.__name__

    # [2] Infer task from model class
    task = _detect_task_from_model_class(arch_model_class)
    logger.info("Detected task: %s (from %s)", task, arch_name)

    # [3] Get model_type - REQUIRED for specialization lookup
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        raise ValueError(
            "Cannot resolve model class: config has no 'model_type' field. "
            "Please specify model_class explicitly."
        )

    # [3a] Per-model-type default task override.
    # Some model families (e.g., SAM/SAM2) have an architecture class whose
    # default TasksManager mapping ("feature-extraction") differs from the
    # canonical export target ("mask-generation"). Honor MODEL_TASK_DEFAULTS
    # to bias auto-detection toward the right export configuration.
    from ..models.hf import MODEL_TASK_DEFAULTS

    model_type_normalized = model_type.lower().replace("_", "-")
    default_task = MODEL_TASK_DEFAULTS.get(model_type_normalized)
    if default_task is not None and default_task != task:
        logger.info(
            "Overriding auto-detected task %r with model-type default %r for %s",
            task,
            default_task,
            model_type_normalized,
        )
        task = default_task

    # [4] Check specializations first (CLIP, SAM2, etc.) - highest priority
    model_class = _get_custom_model_class(model_type, task)
    if model_class:
        logger.info("Using specialized model class: %s", model_class.__name__)
        return task, model_class

    # [5] Try TasksManager, fallback to arch_model_class on failure
    try:
        model_class = TasksManager.get_model_class_for_task(task)

        # Informational: TasksManager may return a generic AutoModel* class
        # that differs from config.architectures — this is expected behavior.
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
        return _detect_task_and_class_from_config(config)

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
