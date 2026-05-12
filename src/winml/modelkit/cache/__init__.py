# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Cache management for WinML CLI.

Provides deterministic path computation for cached build artifacts.
Both ``from_pretrained()`` and ``winml build --use-cache`` use these
functions to guarantee identical paths for the same model+config.

Usage::

    from winml.modelkit.cache import get_cache_dir, get_model_dir, get_cache_key

    output_dir = get_model_dir(model_id)
    cache_key  = get_cache_key(task_abbrev, config_hash)
"""

from .model import get_model_dir, list_cached_models, model_id_to_slug
from .path import get_artifact_path, get_artifacts_dir, get_cache_dir, get_cache_key


__all__ = [
    "get_artifact_path",
    "get_artifacts_dir",
    "get_cache_dir",
    "get_cache_key",
    "get_model_dir",
    "list_cached_models",
    "model_id_to_slug",
]
