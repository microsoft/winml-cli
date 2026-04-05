# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Inspect module for analyzing HuggingFace models.

Provides the inspect_model() function to analyze model compatibility
with ModelKit and display loader/exporter/WinML configurations.

Usage:
    from winml.modelkit.inspect import inspect_model

    result = inspect_model("openai/clip-vit-base-patch32")
    print(result.model_type)  # "clip"
    print(result.task)  # "feature-extraction"
    print(result.loader.hf_model_class)  # "CLIPTextModelWithProjection"
"""

from __future__ import annotations

import logging

from transformers import AutoConfig

from .resolver import (
    build_tensor_infos_from_io_specs,
    compile_support_status,
    detect_task,
    get_build_config,
    get_known_tasks,
    resolve_cache,
    resolve_exporter,
    resolve_io_config,
    resolve_loader,
    resolve_processor,
    resolve_winml,
    validate_task,
)
from .types import (
    CacheInfo,
    CacheStageInfo,
    ExporterInfo,
    HierarchyInfo,
    InspectResult,
    IOConfigInfo,
    LoaderInfo,
    ModuleInfo,
    ProcessorInfo,
    SupportLevel,
    TensorInfo,
    WinMLInfo,
)


logger = logging.getLogger(__name__)


class InspectError(Exception):
    """Base exception for inspect command."""


class ModelNotFoundError(InspectError):
    """Model not found on HuggingFace Hub."""


class NetworkError(InspectError):
    """Network error while fetching model config."""


def inspect_model(
    model_id: str,
    include_hierarchy: bool = False,
    task_override: str | None = None,
) -> InspectResult:
    """Inspect a HuggingFace model and return configuration details.

    Args:
        model_id: HuggingFace model identifier (e.g., "openai/clip-vit-base-patch32")
        include_hierarchy: If True, load model and extract HF module hierarchy
        task_override: If provided, use this task instead of auto-detection

    Returns:
        InspectResult with all configuration details

    Raises:
        ModelNotFoundError: If model doesn't exist on HuggingFace
        NetworkError: If unable to fetch model config

    Example:
        >>> result = inspect_model("openai/clip-vit-base-patch32")
        >>> print(result.model_type)
        "clip"
        >>> print(result.loader.hf_model_class)
        "CLIPTextModelWithProjection"
    """
    logger.info("Inspecting model: %s", model_id)

    # Step 1: Fetch HF config (no model download)
    try:
        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except OSError as e:
        if "404" in str(e) or "not found" in str(e).lower():
            raise ModelNotFoundError(f"Model '{model_id}' not found on HuggingFace Hub") from e
        raise NetworkError(f"Unable to fetch model config: {e}") from e

    model_type = getattr(hf_config, "model_type", "unknown")
    architectures = getattr(hf_config, "architectures", [])

    logger.debug("Model type: %s, Architectures: %s", model_type, architectures)

    # Step 2: Detect or override task
    if task_override:
        try:
            validate_task(task_override)
        except ValueError as e:
            raise InspectError(str(e)) from e
        task = task_override
        task_source = "user_override"
        logger.debug("Task override: %s", task)
    else:
        task, task_source = detect_task(hf_config)
        logger.debug("Detected task: %s (source: %s)", task, task_source)

    # Step 3: Resolve loader configuration
    loader_info = resolve_loader(model_type, task)
    logger.debug(
        "Loader: %s (source: %s)",
        loader_info.hf_model_class,
        loader_info.hf_model_class_source,
    )

    # Step 4: Resolve exporter configuration (pass model_id for correct image sizes)
    exporter_info = resolve_exporter(model_type, task, hf_config=hf_config, model_id=model_id)
    logger.debug(
        "Exporter: %s (source: %s)",
        exporter_info.onnx_config_class,
        exporter_info.onnx_config_source,
    )

    # Step 5: Resolve WinML class
    winml_info = resolve_winml(model_type, task)
    logger.debug("WinML: %s (source: %s)", winml_info.winml_class, winml_info.winml_class_source)

    # Step 5.5: Extract HF module hierarchy (if requested)
    hierarchy_info = None
    if include_hierarchy:
        from .hierarchy import extract_hierarchy

        hierarchy_info = extract_hierarchy(model_id)
        logger.debug("Hierarchy: %d HF modules", hierarchy_info.hf_module_count)

    # Step 6: Compile overall support status
    overall_support, support_notes = compile_support_status(loader_info, exporter_info, winml_info)
    logger.info("Overall support: %s", overall_support.value)

    # Step 7: Get full build config (for verbose output)
    build_config = get_build_config(model_type)

    # Step 8: Check cache status
    cache_info = resolve_cache(model_id)
    logger.debug("Cache: %d/%d stages cached", cache_info.total_cached, len(cache_info.stages))

    # Step 9: Resolve processor classes
    processor_info = resolve_processor(model_id, model_type=model_type)
    logger.debug(
        "Processor: %s, Tokenizer: %s",
        processor_info.processor_class,
        processor_info.tokenizer_class,
    )

    # Step 10: Extract IO config (dynamically discovers attrs from OnnxConfig)
    io_config_info = resolve_io_config(
        hf_config,
        model_id=model_id,
        model_type=model_type,
        task=task,
    )
    logger.debug(
        "IO Config: max_pos=%s, vocab=%s, img_size=%s",
        io_config_info.max_position_embeddings,
        io_config_info.vocab_size,
        io_config_info.image_size,
    )

    return InspectResult(
        model_id=model_id,
        model_type=model_type,
        architectures=architectures,
        task=task,
        task_source=task_source,
        loader=loader_info,
        exporter=exporter_info,
        winml=winml_info,
        overall_support=overall_support,
        support_notes=support_notes,
        build_config=build_config,
        hierarchy=hierarchy_info,
        cache=cache_info,
        processor=processor_info,
        io_config=io_config_info,
    )


__all__ = [
    "CacheInfo",
    "CacheStageInfo",
    "ExporterInfo",
    "HierarchyInfo",
    "IOConfigInfo",
    "InspectError",
    "InspectResult",
    "LoaderInfo",
    "ModelNotFoundError",
    "ModuleInfo",
    "NetworkError",
    "ProcessorInfo",
    "SupportLevel",
    "TensorInfo",
    "WinMLInfo",
    "build_tensor_infos_from_io_specs",
    "get_known_tasks",
    "inspect_model",
]
