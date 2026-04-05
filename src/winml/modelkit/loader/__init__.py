# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HuggingFace Model Loader Module.

Provides unified HF model loading with automatic task detection.

Example:
    >>> from winml.modelkit.loader import load_hf_model
    >>> model, config, task = load_hf_model("microsoft/resnet-50")

    >>> # With explicit model class
    >>> model, config, task = load_hf_model(
    ...     "openai/clip-vit-base-patch32",
    ...     task="feature-extraction",
    ...     model_class="CLIPTextModelWithProjection",
    ... )

Note:
    This module consolidates config loading, task detection,
    and model instantiation into a single workflow. Model patches for
    ONNX export are applied via Optimum's ModelPatcher / PATCHING_SPECS
    during export, not at load time.
    See docs/design/loader/hf.md for design details.
"""

from .config import WinMLLoaderConfig, resolve_loader_config
from .task import (
    HF_TASK_DEFAULTS,
    get_supported_tasks,
    get_task_abbrev,
    normalize_task,
    resolve_task_and_model_class,
)


__all__ = [
    "HF_TASK_DEFAULTS",
    "WinMLLoaderConfig",
    "get_supported_tasks",
    "get_task_abbrev",
    "load_hf_model",
    "normalize_task",
    "resolve_hf_model_class",
    "resolve_loader_config",
    "resolve_task_and_model_class",
]


def __getattr__(name: str):
    """Lazy-load heavy exports (hf.py imports transformers)."""
    if name in ("load_hf_model", "resolve_hf_model_class"):
        from .hf import load_hf_model, resolve_hf_model_class

        globals().update(
            {
                "load_hf_model": load_hf_model,
                "resolve_hf_model_class": resolve_hf_model_class,
            }
        )
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
