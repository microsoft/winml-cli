"""Cache path primitives for ModelKit.

Pure functions that compute deterministic cache paths. This module
knows about cache directories and key assembly — nothing about models.

Dependencies: stdlib only (os, pathlib). ZERO internal imports.
"""

from __future__ import annotations

import os
from pathlib import Path


__all__ = [
    "get_artifact_path",
    "get_artifacts_dir",
    "get_cache_dir",
    "get_cache_key",
]

# Default cache directory name
_DEFAULT_CACHE_DIR_NAME = "winml"


def get_cache_dir(override: str | Path | None = None) -> Path:
    """Resolve the root cache directory.

    Priority:
        1. ``override`` parameter (caller-specified)
        2. ``WMK_CACHE_DIR`` environment variable
        3. ``~/.cache/winml/``

    Args:
        override: Explicit cache directory path. Takes highest priority.

    Returns:
        Resolved cache directory path (not created — caller decides).
    """
    if override is not None:
        return Path(override)
    env_dir = os.environ.get("WMK_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".cache" / _DEFAULT_CACHE_DIR_NAME


def get_artifacts_dir(cache_dir: Path | None = None) -> Path:
    """Get the artifacts subdirectory within the cache.

    Args:
        cache_dir: Cache root. If ``None``, resolves via :func:`get_cache_dir`.

    Returns:
        ``{cache_dir}/artifacts/``
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    return cache_dir / "artifacts"


def get_cache_key(task_abbrev: str, config_hash: str) -> str:
    """Assemble the cache key prefix from task abbreviation and config hash.

    Args:
        task_abbrev: Abbreviated task name (e.g., ``"imgcls"``).
            Obtained from ``modelkit.loader.task.get_task_abbrev()``.
        config_hash: Deterministic config hash (e.g., ``"a1b2c3d4e5f67890"``).
            Obtained from ``WinMLBuildConfig.generate_cache_key()``.

    Returns:
        Cache key string: ``"{task_abbrev}_{config_hash}"``
    """
    return f"{task_abbrev}_{config_hash}"


def get_artifact_path(
    model_dir: Path,
    cache_key: str,
    stage: str,
    ext: str = ".onnx",
) -> Path:
    """Compute the full path for a cached artifact.

    Path template::

        {model_dir}/{cache_key}_{stage}{ext}

    Args:
        model_dir: Model-specific directory (from :func:`get_model_dir`).
        cache_key: Cache key prefix (from :func:`get_cache_key`).
        stage: Pipeline stage (``"export"``, ``"optimized"``, ``"model"``, etc.).
        ext: File extension (default ``".onnx"``).

    Returns:
        Full path to the artifact file.
    """
    return model_dir / f"{cache_key}_{stage}{ext}"
