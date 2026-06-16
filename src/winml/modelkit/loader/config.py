# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLLoaderConfig - Configuration and resolution for HuggingFace model loading.

This module provides:
- WinMLLoaderConfig: Dataclass for model loading configuration
- resolve_loader_config(): Resolve raw user inputs into a complete loader config
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from .resolution import TaskResolution

logger = logging.getLogger(__name__)


@dataclass
class WinMLLoaderConfig:
    """Configuration for HuggingFace model loading.

    Resolution priority:
    1. user_script + model_class → load from custom script
    2. model_class → load from transformers (overrides auto-discovery)
    3. task only → auto-discover model class from task

    Attributes:
        task: Task name (e.g., "image-classification", "feature-extraction").
            Auto-detected from model config if not provided.
        model_class: Model class name to override auto-detection.
            Examples: "AutoModelForCTC", "CLIPTextModelWithProjection"
        model_type: HuggingFace model type (e.g., "bert", "clip").
            Resolved from hf_config.model_type or overridden explicitly.
        module_path: Dotted path to a specific submodule. Used by build for
            get_submodule(). Example: "encoder.layer.0.attention"
        user_script: Path to Python script defining custom model class.
            The script must define a class matching `model_class` at module level.
            Requires trust_remote_code=True for security.
        trust_remote_code: Whether to trust remote/custom code.
            Required when using user_script.

    Example:
        # Standard usage with auto-detection
        config = WinMLLoaderConfig()

        # Override model class
        config = WinMLLoaderConfig(
            task="automatic-speech-recognition",
            model_class="AutoModelForCTC",
        )

        # Custom script
        config = WinMLLoaderConfig(
            task="image-classification",
            model_class="PatchedConvNext",
            user_script="scripts/custom.py",
            trust_remote_code=True,
        )
    """

    task: str | None = None
    model_class: str | None = None
    model_type: str | None = None
    module_path: str | None = None
    user_script: str | None = None
    trust_remote_code: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of config.
        """
        result: dict[str, Any] = {}
        if self.task is not None:
            result["task"] = self.task
        if self.model_class is not None:
            result["model_class"] = self.model_class
        if self.model_type is not None:
            result["model_type"] = self.model_type
        if self.module_path is not None:
            result["module_path"] = self.module_path
        if self.user_script is not None:
            result["user_script"] = self.user_script
        if self.trust_remote_code:
            result["trust_remote_code"] = self.trust_remote_code
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WinMLLoaderConfig:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with config values.

        Returns:
            WinMLLoaderConfig instance.
        """
        return cls(
            task=data.get("task"),
            model_class=data.get("model_class"),
            model_type=data.get("model_type"),
            module_path=data.get("module_path"),
            user_script=data.get("user_script"),
            trust_remote_code=data.get("trust_remote_code", False),
        )


def resolve_loader_config(
    model_id: str | None = None,
    *,
    task: str | None = None,
    model_class: str | None = None,
    model_type: str | None = None,
    trust_remote_code: bool = False,
    library_name: str = "transformers",
    hf_config: PretrainedConfig | None = None,
) -> tuple[WinMLLoaderConfig, PretrainedConfig, type, TaskResolution]:
    """Resolve all loader concerns from raw user inputs.

    Encapsulates hf_config loading/creation, model_type override,
    task auto-detection, and task/model_class resolution into a single call.

    Internal call flow (step numbers match code comments)::

        1. Load hf_config (depends on: model_id, model_type, or model_class)
           - model_id → AutoConfig.from_pretrained(model_id)
           - model_type → AutoConfig.for_model(model_type)
           - model_class → create_hf_config_from_model_class(cls)
        2-3. Unified task + model_class resolution (depends on: hf_config + task)
           → resolve_task(hf_config, task, model_class)
           Subsumes the old --model-type fallback (first supported task) and the
           legacy task/model-class orchestrator; returns a TaskResolution.
        4. Resolve hf_config + model_type (depends on: resolved_class)
           → _resolve_hf_config_for_class(hf_config, resolved_class)
           Uses config_class.base_config_key to extract sub-config for multimodal
        5. Build WinMLLoaderConfig from resolved values

    Args:
        model_id: HuggingFace model ID or local path. Optional when model_type
            is provided (uses default HF config via AutoConfig.for_model).
        task: Override auto-detected task (e.g., "text-classification").
        model_class: Override auto-detected model class name.
        model_type: Override auto-detected model type (e.g., "bert", "resnet").
            When provided without model_id, creates a default HF config.
            When provided without task, the first supported task is used.
        trust_remote_code: Whether to trust remote/custom code.
        library_name: Source library for TasksManager lookup.
        hf_config: Pre-loaded HF config to reuse instead of fetching again.
            When supplied, step 1's ``AutoConfig.from_pretrained`` is skipped.
            Use this when the caller already has the parent config (e.g.,
            inspect needs the un-narrowed config for I/O introspection).

    Returns:
        Tuple of:
        - loader_config: Resolved WinMLLoaderConfig with task, model_class,
          model_type populated.
        - hf_config: Resolved HF config for the model class. For multimodal
          models, this is the sub-config (e.g., CLIPTextConfig), not the parent.
        - resolved_class: Actual model class type for instantiation.
        - resolution: Full TaskResolution with provenance (source, optimum_task,
          composite). Consumed by inspect and future callers for richer output.

    Raises:
        ValueError: If neither model_id nor model_type is provided, model_type
            is unknown, task detection fails, or no supported tasks found.
    """
    from transformers import AutoConfig

    from .resolution import resolve_task

    if trust_remote_code:
        from ..utils.cli import warn_trust_remote_code

        warn_trust_remote_code()

    # 1. Load hf_config (depends on: model_id, model_type, or model_class)
    if hf_config is not None:
        # Caller supplied a pre-loaded config — skip the round-trip.
        pass
    elif model_id is not None:
        hf_config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
    elif model_type is not None:
        try:
            hf_config = AutoConfig.for_model(model_type)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Use a valid HuggingFace model type (e.g., 'bert', 'resnet', 'gpt2') "
                f"or provide --model with a model ID instead."
            ) from e
        logger.info("Created default HF config for model_type='%s'", model_type)
    elif model_class is not None:
        from .hf import resolve_hf_model_class

        try:
            cls = resolve_hf_model_class(model_class)
        except ImportError as e:
            raise ValueError(str(e)) from e
        hf_config = _create_hf_config_from_model_class(cls)
        logger.info("Created HF config from model_class='%s'", model_class)
    else:
        raise ValueError("At least one of model_id, model_type, or model_class must be provided.")

    if getattr(hf_config, "model_type", None) is None:
        raise ValueError(
            f"Config for '{model_id or model_type}' does not have 'model_type' "
            f"attribute. Cannot proceed with config generation."
        )

    # 2-3. Unified resolution. Task detection — including the no-architectures
    # --model-type fallback (first supported task) — now lives in resolve_task.
    resolution = resolve_task(hf_config, task=task, model_class=model_class)
    resolved_task, resolved_class = resolution.task, resolution.model_class
    logger.info("Resolved: task=%s, model_class=%s", resolved_task, resolved_class.__name__)

    # 4. Resolve hf_config + model_type (depends on: resolved_class)
    resolved_hf_config, resolved_model_type = _resolve_hf_config_for_class(
        hf_config,
        resolved_class,
    )

    # 5. Build loader config
    loader_config = WinMLLoaderConfig(
        task=resolved_task,
        model_class=resolved_class.__name__,
        model_type=resolved_model_type,
        trust_remote_code=trust_remote_code,
    )

    return loader_config, resolved_hf_config, resolved_class, resolution


def _create_hf_config_from_model_class(model_class: type) -> PretrainedConfig:
    """Create a default HF config from a model class (no network access).

    Uses ``model_class.config_class()`` to instantiate a default config and
    sets ``architectures`` so that task detection works.

    Args:
        model_class: A HuggingFace model class (e.g., BertForMaskedLM).

    Returns:
        PretrainedConfig with ``architectures`` set.

    Raises:
        TypeError: If model_class lacks config_class or it's not callable.
    """
    config_cls = getattr(model_class, "config_class", None)
    if config_cls is None or not callable(config_cls):
        raise TypeError(
            f"'{getattr(model_class, '__name__', model_class)}' does not have a "
            f"'config_class' attribute. Expected a HuggingFace model class."
        )
    hf_config = config_cls()
    hf_config.architectures = [model_class.__name__]
    return hf_config


def _resolve_hf_config_for_class(
    hf_config: PretrainedConfig,
    resolved_class: type,
) -> tuple[PretrainedConfig, str]:
    """Extract the correct hf_config and model_type for a resolved model class.

    For multimodal models (CLIP, etc.), the parent config contains sub-configs
    for each modality. The resolved model class (e.g., CLIPTextModelWithProjection)
    expects the sub-config (CLIPTextConfig), not the parent (CLIPConfig).

    Uses HF's ``config_class.base_config_key`` to find the sub-config attribute
    on the parent. For example::

        CLIPTextConfig.base_config_key = "text_config"
        → getattr(CLIPConfig_instance, "text_config") → CLIPTextConfig instance

    For single-model architectures (BERT, ResNet), ``base_config_key`` is empty
    and the parent config is returned unchanged.

    Args:
        hf_config: Parent HuggingFace config (e.g., CLIPConfig).
        resolved_class: Model class (e.g., CLIPTextModelWithProjection).

    Returns:
        Tuple of (resolved_hf_config, resolved_model_type).
    """
    config_cls = getattr(resolved_class, "config_class", None)
    base_key = getattr(config_cls, "base_config_key", "") if config_cls else ""

    if isinstance(base_key, str) and base_key and hasattr(hf_config, base_key):
        hf_config = getattr(hf_config, base_key)
        logger.info(
            "Using sub-config '%s' (model_type='%s') for %s",
            base_key,
            hf_config.model_type,
            resolved_class.__name__,
        )

    return hf_config, hf_config.model_type


__all__ = ["WinMLLoaderConfig", "resolve_loader_config"]
