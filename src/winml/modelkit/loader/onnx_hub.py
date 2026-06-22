# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Download pre-exported ONNX files hosted on the HuggingFace Hub.

This module is the **download** half of Hub-hosted ONNX support.
Classification (deciding whether a ``-m/--model`` value is a Hub ONNX
ref, a local ``.onnx`` file, an HF model ID, or a build directory) lives
in :mod:`winml.modelkit.utils.model_input` and is the single entry point
that all CLI commands and library APIs should go through.

The function exposed here, :func:`resolve_hf_onnx_path`, is called by
``resolve_model_input`` for the ``hub_onnx`` case and downloads the
``.onnx`` file (plus any ``.onnx_data`` sidecar) via
``huggingface_hub.hf_hub_download``.
"""

from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def resolve_hf_onnx_path(
    model_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> Path:
    """Download a Hub-hosted ONNX file and return the local path.

    Splits ``model_id`` into ``(repo_id, filename)``, downloads the
    ``.onnx`` file, and best-effort fetches an optional
    ``<filename>.onnx_data`` sidecar so the ONNX loader can find external
    initializers.

    Args:
        model_id: A Hub ONNX reference such as
            ``"onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx"``.
        revision: Optional Hub revision (branch, tag, or commit SHA).
        cache_dir: Optional override for the ``huggingface_hub`` cache directory.
        token: Optional auth token forwarded to ``hf_hub_download``.

    Returns:
        The local path to the downloaded ``.onnx`` file.

    Raises:
        ValueError: If ``model_id`` does not have at least three ``/``-separated
            components.
        FileNotFoundError: If the referenced ``.onnx`` file does not exist in
            the repo. The error message lists the ``.onnx`` files that *are*
            present so the user can correct the path.
        huggingface_hub.utils.RepositoryNotFoundError: If the repo itself does
            not exist (re-raised unchanged).
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    repo_id, filename = _split_hf_onnx_path(model_id)
    logger.info("Downloading ONNX from Hub: repo=%s file=%s", repo_id, filename)

    try:
        local_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
        )
    except EntryNotFoundError as e:
        # The repo exists but ``filename`` does not. Surface the available
        # ``.onnx`` files so the user can pick the right one without leaving
        # the terminal. Re-raise as ``FileNotFoundError`` so callers that
        # already handle local-file-missing errors get a consistent type.
        hint = _format_available_onnx_files(
            repo_id, revision=revision, token=token
        )
        raise FileNotFoundError(
            f"ONNX file '{filename}' not found in Hub repo '{repo_id}'.\n{hint}"
        ) from e

    # External-data sidecars (used for >2 GiB models) live next to the .onnx
    # file with a ``.onnx_data`` suffix. The main download above just
    # succeeded for the same repo, so the only expected reason the sidecar
    # is missing is that the model inlined its weights -- catch
    # ``EntryNotFoundError`` quietly. Any other failure (disk full,
    # permissions, network blip, etc.) is real and surfaced as a warning
    # so the user is not left with a half-downloaded model that fails
    # later at load time with a confusing error.
    sidecar_filename = f"{filename}_data"
    try:
        sidecar_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=sidecar_filename,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
        )
        logger.info("Downloaded external-data sidecar: %s", sidecar_path.name)
    except EntryNotFoundError:
        # Expected: model has no separate weights file (weights are inlined).
        logger.debug("No external-data sidecar at %s (weights inlined)", sidecar_filename)
    except OSError as e:
        # Unexpected: disk/permission/network problem. Warn loudly --
        # silent failure here would make the model unloadable later.
        logger.warning(
            "Failed to download external-data sidecar %s for %s: %s. "
            "If the model uses external weights, loading will fail.",
            sidecar_filename,
            repo_id,
            e,
        )

    return local_path


def _split_hf_onnx_path(model_id: str) -> tuple[str, str]:
    """Split a Hub ONNX reference into ``(repo_id, filename)``."""
    parts = [p for p in model_id.split("/") if p]
    if len(parts) < 3:
        raise ValueError(
            f"Hub ONNX reference must have form 'org/repo/path/to/file.onnx', got: {model_id!r}"
        )
    return "/".join(parts[:2]), "/".join(parts[2:])


def _format_available_onnx_files(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
) -> str:
    """Build a human-readable hint listing ``.onnx`` files in a Hub repo.

    Used to enrich ``EntryNotFoundError`` messages so users who guessed the
    wrong filename can see the available options without leaving the
    terminal. Best-effort: if listing fails for any reason (network,
    auth, gated repo) we return a generic fallback hint instead of
    masking the original error.
    """
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(repo_id, revision=revision, token=token)
    except Exception as list_err:
        logger.debug("Could not list files for %s: %s", repo_id, list_err)
        return (
            f"Could not list available .onnx files in '{repo_id}' "
            f"(see https://huggingface.co/{repo_id}/tree/main)."
        )

    onnx_files = sorted(f for f in files if f.lower().endswith(".onnx"))
    if not onnx_files:
        return (
            f"No .onnx files were found in '{repo_id}'. "
            f"This repo may not host pre-exported ONNX weights; "
            f"see https://huggingface.co/{repo_id}/tree/main."
        )

    listing = "\n".join(f"  - {repo_id}/{f}" for f in onnx_files)
    return f"Available .onnx files in '{repo_id}':\n{listing}"


__all__ = [
    "resolve_hf_onnx_path",
]
