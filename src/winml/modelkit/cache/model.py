# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Model-aware cache operations for ModelKit.

Adds model_id to slug mapping, model directory resolution, and
directory scanning for cached artifact enumeration.

Dependencies: imports from ``modelkit.cache.path`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .path import get_artifacts_dir, get_cache_dir


if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "get_model_dir",
    "list_cached_models",
    "model_id_to_slug",
]

# Length of the config hash produced by WinMLBuildConfig.generate_cache_key()
_CONFIG_HASH_LENGTH = 16


def model_id_to_slug(model_id: str) -> str:
    r"""Convert a model identifier to a filesystem-safe slug.

    Replaces ``/`` and ``\\`` with ``_``.
    Falls back to ``"random-init"`` for empty string.

    Args:
        model_id: HuggingFace model ID (e.g., ``"facebook/convnext-tiny-224"``).

    Returns:
        Filesystem-safe slug (e.g., ``"facebook_convnext-tiny-224"``).
    """
    if not model_id:
        return "random-init"
    return model_id.replace("/", "_").replace("\\", "_")


def get_model_dir(
    model_id: str,
    cache_dir: Path | None = None,
) -> Path:
    """Get the model-specific directory within the artifacts folder.

    Args:
        model_id: HuggingFace model ID.
        cache_dir: Cache root. If ``None``, resolves via :func:`get_cache_dir`.

    Returns:
        ``{cache_dir}/artifacts/{model_slug}/``
    """
    return get_artifacts_dir(cache_dir) / model_id_to_slug(model_id)


def list_cached_models(cache_dir: Path | None = None) -> list[dict[str, str]]:
    """Enumerate cached models from the directory structure.

    Scans ``{cache_dir}/artifacts/*/`` and parses filenames matching
    the ``{task_abbrev}_{config_hash}_{stage}.onnx`` pattern.

    Args:
        cache_dir: Cache root. If ``None``, resolves via :func:`get_cache_dir`.

    Returns:
        List of dicts with keys: ``model_slug``, ``task_abbrev``,
        ``config_hash``, ``stage``, ``filename``, ``path``.
        Returns empty list if artifacts dir does not exist.
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    artifacts_dir = get_artifacts_dir(cache_dir)
    if not artifacts_dir.exists():
        return []

    results: list[dict[str, str]] = []
    for model_dir in sorted(artifacts_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_slug = model_dir.name
        for artifact in sorted(model_dir.glob("*.onnx")):
            parsed = _parse_artifact_filename(artifact.stem)
            if parsed is None:
                continue
            task_abbrev, config_hash, stage = parsed
            results.append({
                "model_slug": model_slug,
                "task_abbrev": task_abbrev,
                "config_hash": config_hash,
                "stage": stage,
                "filename": artifact.name,
                "path": str(artifact),
            })
    return results


def _parse_artifact_filename(
    stem: str,
) -> tuple[str, str, str] | None:
    """Parse an artifact filename stem into components.

    Expected format: ``{task_abbrev}_{config_hash}_{stage}``
    where config_hash is exactly :data:`_CONFIG_HASH_LENGTH` hex chars.

    Returns:
        ``(task_abbrev, config_hash, stage)`` or ``None`` if unparseable.
    """
    # Split off stage (last segment after final "_")
    last_sep = stem.rfind("_")
    if last_sep < 0:
        return None
    stage = stem[last_sep + 1:]
    prefix = stem[:last_sep]

    # prefix = "{task_abbrev}_{config_hash}"
    # config_hash is always _CONFIG_HASH_LENGTH hex chars at the end
    min_prefix_len = 1 + 1 + _CONFIG_HASH_LENGTH  # "x_" + 16 chars
    if len(prefix) < min_prefix_len:
        return None

    config_hash = prefix[-_CONFIG_HASH_LENGTH:]
    # Validate hex
    try:
        int(config_hash, 16)
    except ValueError:
        return None

    # task_abbrev is everything before "_{config_hash}"
    task_abbrev = prefix[: -(_CONFIG_HASH_LENGTH + 1)]
    if not task_abbrev:
        return None

    return task_abbrev, config_hash, stage
