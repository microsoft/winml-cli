# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Download pre-exported ONNX files hosted on the HuggingFace Hub.

ModelKit accepts two model input forms today: a HuggingFace model ID
(``org/name``) for the standard ``transformers`` + ``optimum-onnx`` export
path, and a local ``.onnx`` file path for the Scenario D pipeline in
``modelkit.build.build_onnx_model``.

This module recognizes a third form -- a path-style reference to a
pre-exported ONNX artifact in a Hub repo, e.g.::

    onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx

The first two ``/``-separated components are interpreted as the repo ID;
everything that follows is the file path inside the repo. The file is
downloaded once via ``huggingface_hub.hf_hub_download`` and the local
path is then handed to the existing Scenario D code path. This is the
supported route for models like SAM 3 whose ``transformers`` requirement
exceeds what ``optimum-onnx`` currently pins.

Any sibling ``<file>.onnx_data`` external-data sidecar is fetched
best-effort so the ONNX loader can resolve external initializers.
"""

from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def is_hf_onnx_path(model_id: str | None) -> bool:
    """Check whether ``model_id`` is a Hub-style reference to a pre-exported ONNX file.

    Returns True only when the value has at least three ``/``-separated
    components, ends with ``.onnx``, and does not point at an existing
    local file or directory. Local paths always win over the Hub
    interpretation so users can keep working with paths that happen to
    look like repo IDs.
    """
    if not model_id:
        return False
    if not model_id.endswith(".onnx"):
        return False
    if Path(model_id).exists():
        return False
    parts = [p for p in model_id.split("/") if p]
    return len(parts) >= 3


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
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    repo_id, filename = _split_hf_onnx_path(model_id)
    logger.info("Downloading ONNX from Hub: repo=%s file=%s", repo_id, filename)

    local_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
    )

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


def maybe_resolve_hf_onnx_path(
    model_id: str | None,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> str | None:
    """Resolve ``model_id`` to a local ONNX path if it is a Hub ONNX reference.

    Convenience wrapper that combines :func:`is_hf_onnx_path` and
    :func:`resolve_hf_onnx_path`. Non-Hub inputs (HF model IDs, local
    paths, ``None``) are returned unchanged so callers can use this as a
    transparent normalization step before dispatching to existing code.

    Args:
        model_id: HF model ID, local path, Hub ONNX ref, or ``None``.
        revision: Optional Hub revision (forwarded when downloading).
        cache_dir: Optional cache override (forwarded when downloading).
        token: Optional auth token (forwarded when downloading).

    Returns:
        Local ``.onnx`` path string when ``model_id`` was a Hub ref; the
        original ``model_id`` otherwise.
    """
    if not is_hf_onnx_path(model_id):
        return model_id
    return str(
        resolve_hf_onnx_path(
            model_id,  # type: ignore[arg-type]  # is_hf_onnx_path() rejects None
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
    )


__all__ = [
    "is_hf_onnx_path",
    "maybe_resolve_hf_onnx_path",
    "resolve_hf_onnx_path",
]
