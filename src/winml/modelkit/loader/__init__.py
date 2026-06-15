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
from .onnx_hub import resolve_hf_onnx_path
from .task import (
    HF_TASK_DEFAULTS,
    KNOWN_TASKS,
    TASK_SYNONYM_EXTENSIONS,
    detect_task,
    get_supported_tasks,
    get_task_abbrev,
    normalize_task,
    resolve_optimum_library,
    resolve_task_and_model_class,
    to_optimum_task,
)


__all__ = [
    "HF_TASK_DEFAULTS",
    "KNOWN_TASKS",
    "TASK_SYNONYM_EXTENSIONS",
    "WinMLLoaderConfig",
    "detect_task",
    "get_supported_tasks",
    "get_task_abbrev",
    "load_hf_model",
    "normalize_task",
    "resolve_hf_model_class",
    "resolve_hf_onnx_path",
    "resolve_loader_config",
    "resolve_optimum_library",
    "resolve_task_and_model_class",
    "to_optimum_task",
]


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "load_hf_model": (".hf", "load_hf_model"),
    "resolve_hf_model_class": (".hf", "resolve_hf_model_class"),
}


def __getattr__(name: str):
    """Lazy-load heavy exports (hf.py imports transformers)."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(set(list(globals()) + __all__))
