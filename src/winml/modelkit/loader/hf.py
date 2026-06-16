# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HuggingFace Model Loader.

Provides unified HF model loading with automatic task detection.
Model patches for ONNX export are applied via Optimum's ModelPatcher /
PATCHING_SPECS mechanism during export, not at load time.

Example:
    >>> from winml.modelkit.loader import load_hf_model
    >>> model, config, task = load_hf_model("microsoft/resnet-50")
    >>> # model is ready for ONNX export

    >>> # With explicit model class
    >>> model, config, task = load_hf_model(
    ...     "openai/clip-vit-base-patch32",
    ...     task="feature-extraction",
    ...     model_class="CLIPTextModelWithProjection",
    ... )
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from transformers import AutoConfig


if TYPE_CHECKING:
    import torch.nn as nn
    from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

# Priority-ordered list of HF ecosystem modules to search for model classes.
# transformers first (most common), then specialized libraries.
_HF_MODEL_MODULES = [
    "transformers",
    "timm",
    "diffusers",
    "sentence_transformers",
]


def resolve_hf_model_class(class_name: str) -> type:
    """Resolve a model class name to an actual class from HF ecosystem modules.

    Searches ``_HF_MODEL_MODULES`` in priority order. Returns the first match.

    Args:
        class_name: Model class name (e.g., ``"AutoModelForImageClassification"``,
            ``"CLIPTextModelWithProjection"``, ``"ConvNextForImageClassification"``).

    Returns:
        The resolved model class.

    Raises:
        ImportError: If class_name not found in any known module.

    Example:
        >>> from winml.modelkit.loader import resolve_hf_model_class
        >>> cls = resolve_hf_model_class("AutoModelForImageClassification")
        >>> model = cls.from_pretrained("microsoft/resnet-50")
    """
    for module_name in _HF_MODEL_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue  # Module not installed, skip
        cls = getattr(module, class_name, None)
        if cls is not None:
            logger.debug("Resolved '%s' from '%s'", class_name, module_name)
            return cast("type", cls)

    raise ImportError(
        f"Model class '{class_name}' not found in any of: {', '.join(_HF_MODEL_MODULES)}"
    )


def _load_class_from_script(script_path: str, class_name: str) -> type:
    """Load a model class from a user-provided Python script.

    Args:
        script_path: Path to Python script (.py file)
        class_name: Name of the class to load from the script

    Returns:
        The model class from the script

    Raises:
        FileNotFoundError: If script doesn't exist
        AttributeError: If class not found in script
        ValueError: If script path is invalid
    """
    path = Path(script_path)

    if not path.exists():
        raise FileNotFoundError(f"User script not found: {script_path}")

    if not path.suffix == ".py":
        raise ValueError(f"User script must be a .py file, got: {script_path}")

    logger.info("Loading model class '%s' from script: %s", class_name, script_path)

    # Load module from file
    spec = importlib.util.spec_from_file_location("user_model_script", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to load script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class from the module
    if not hasattr(module, class_name):
        available = [name for name in dir(module) if not name.startswith("_")]
        raise AttributeError(
            f"Class '{class_name}' not found in script '{script_path}'. "
            f"Available names: {available}"
        )

    model_class = getattr(module, class_name)

    # Validate it looks like a model class
    if not isinstance(model_class, type):
        raise TypeError(f"'{class_name}' in script is not a class, got {type(model_class)}")

    if not hasattr(model_class, "from_pretrained"):
        raise TypeError(f"Class '{class_name}' must have 'from_pretrained' method")

    logger.debug("Successfully loaded class: %s", model_class)
    return model_class


def load_hf_model(
    model_name_or_path: str,
    task: str | None = None,
    model_class: str | None = None,
    user_script: str | None = None,
    trust_remote_code: bool = False,
    hf_config: PretrainedConfig | None = None,
) -> tuple[nn.Module, PretrainedConfig, str]:
    """Load, detect task, and prepare HuggingFace model.

    Pipeline:
    1. Load HF config
    2. Detect task (if not provided) or normalize (if provided)
    3. Resolve model class (user_script > model_class > auto-detect)
    4. Instantiate model from pretrained
    5. Prepare for export (eval mode)

    Note:
        Model patches for ONNX export compatibility (ConvNeXT LayerNorm,
        SAM2 window partition, etc.) are applied via Optimum's ModelPatcher /
        PATCHING_SPECS mechanism during export, not at load time.

    Args:
        model_name_or_path: HuggingFace model identifier or local path
        task: Optional task name (auto-detected if None)
        model_class: Optional model class name to override auto-detection.
            Examples: "AutoModelForCTC", "CLIPTextModelWithProjection"
        user_script: Optional path to Python script defining custom model class.
            The script must define a class matching `model_class` at module level.
            Requires trust_remote_code=True for security.
        trust_remote_code: Whether to trust remote code (required for user_script)
        hf_config: Optional pre-loaded HF config. When supplied, the
            ``AutoConfig.from_pretrained`` round-trip is skipped — same dedup
            pattern as ``resolve_loader_config(hf_config=...)`` from PR #719.

    Returns:
        Tuple of (model, hf_config, task)
        - model: PyTorch model ready for export
        - hf_config: HuggingFace PretrainedConfig
        - task: Canonical task name (e.g., "image-classification")

    Raises:
        ValueError: If task cannot be detected or is not supported
        ValueError: If user_script provided without trust_remote_code=True

    Example:
        >>> model, config, task = load_hf_model("microsoft/resnet-50")
        >>> # task = "image-classification"
        >>> # model is patched and in eval mode

        >>> # With explicit model class
        >>> model, config, task = load_hf_model(
        ...     "openai/clip-vit-base-patch32",
        ...     task="feature-extraction",
        ...     model_class="CLIPTextModelWithProjection",
        ... )
    """
    logger.info("Loading HF model: %s", model_name_or_path)

    if trust_remote_code:
        from ..utils.cli import warn_trust_remote_code

        warn_trust_remote_code()

    # Validate user_script requirements before any network calls
    if user_script is not None:
        if not trust_remote_code:
            raise ValueError(
                "user_script requires trust_remote_code=True for security. "
                "Loading arbitrary Python code is potentially dangerous."
            )
        if model_class is None:
            raise ValueError("model_class must be specified when using user_script")

    # [1] Load HF Config
    if hf_config is None:
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

    # [2] Task & Model Class Resolution
    from .resolution import resolve_task

    if user_script is not None:
        # model_class is guaranteed non-None here: the validation block above
        # raises when user_script is set without a model_class.
        assert model_class is not None
        resolved_class = _load_class_from_script(user_script, model_class)
        logger.info("Using custom model class from script: %s", model_class)
        # Surfaced modality-aware task (consistent with the non-script branch).
        task = resolve_task(hf_config, task=task).task
    else:
        try:
            resolution = resolve_task(hf_config, task=task, model_class=model_class)
            task, resolved_class = resolution.task, resolution.model_class
        except ValueError as e:
            raise ValueError(
                f"Cannot resolve task/model for {model_name_or_path}. Original error: {e}"
            ) from e

    # [4] Model Instantiation
    logger.debug("Loading model with class: %s", resolved_class.__name__)
    # resolved_class is a dynamically-resolved model class (transformers, timm,
    # diffusers, ...); from_pretrained is a duck-typed boundary across these libs.
    model = cast("Any", resolved_class).from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    # [5] Export Preparation
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Model loaded and prepared for export: %s", model.__class__.__name__)

    return model, hf_config, task
