# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Single classifier + resolver for ``-m/--model`` input values.

ModelKit accepts four shapes for a model reference:

* a HuggingFace model ID (``org/name``);
* a local ``.onnx`` file path;
* a Hub-hosted ONNX artifact (``org/repo/path/file.onnx``); and
* a local build-output directory (containing a ModelKit manifest + cached ONNX).

Historically the codebase had separate detectors for each form
(``is_hub_model``, ``is_hf_onnx_path``, ``is_onnx_file_path``, plus
scattered ``path.suffix == ".onnx"`` checks). This module replaces them
with a single classifier (:func:`classify_model_input`) and a single
resolver (:func:`resolve_model_input`) so adding a fourth input form
later means editing one function, not seven.

The classifier is pure (no I/O beyond the shared ``_is_local_path``
existence check). The resolver downloads Hub-hosted ONNX artifacts via
:func:`~winml.modelkit.loader.onnx_hub.resolve_hf_onnx_path` and
populates ``local_path`` on the returned :class:`ModelInput`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .hub_utils import _is_local_path


ModelInputKind = Literal[
    "local_onnx",   # an existing .onnx file on the local filesystem
    "build_dir",    # an existing directory (typically a ModelKit build output)
    "hub_onnx",     # org/repo/path/file.onnx -> needs Hub download
    "hf_id",        # HuggingFace model id (org/name or model-name)
    "invalid",      # empty or unparsable
]


@dataclass(frozen=True)
class ModelInput:
    """Discriminated classification of a ``-m/--model`` value.

    Attributes:
        kind: One of :data:`ModelInputKind`.
        raw: The original user-supplied string (unchanged).
        local_path: Local filesystem path for ``local_onnx`` / ``build_dir``,
            or the cached download path for ``hub_onnx`` after
            :func:`resolve_model_input`. ``None`` for ``hf_id`` /
            ``invalid``.
        hf_id: HuggingFace repo id (``org/name``) for ``hf_id`` /
            ``hub_onnx``. ``None`` otherwise.
    """

    kind: ModelInputKind
    raw: str
    local_path: str | None = None
    hf_id: str | None = None


def classify_model_input(value: str) -> ModelInput:
    """Classify a ``-m/--model`` value without any network I/O.

    Resolution order (first match wins):

    1. Empty / falsy → ``invalid``.
    2. Looks like a local path (``_is_local_path``) →
       ``local_onnx`` if the suffix is ``.onnx``, else ``build_dir`` if
       it is an existing directory, else ``invalid``.
    3. Has ≥ 3 ``/``-separated components and ends with ``.onnx`` →
       ``hub_onnx``.
    4. Otherwise → ``hf_id``.

    Local-path rejection in step 2 reuses ``_is_local_path`` (existing
    path, ``./``/``../``/``/``/``~/`` prefixes, Windows drive letters),
    the same logic used by ``is_hub_model``, so all four forms apply
    identical rejection rules.
    """
    if not value:
        return ModelInput(kind="invalid", raw=value or "")

    raw = value

    if _is_local_path(value):
        path = Path(value)
        if path.suffix.lower() == ".onnx":
            return ModelInput(kind="local_onnx", raw=raw, local_path=str(path))
        if path.is_dir():
            return ModelInput(kind="build_dir", raw=raw, local_path=str(path))
        # A local path that is neither .onnx nor a directory (e.g. a
        # mistyped file). Leave it to the caller to surface a friendly
        # error from its own context.
        return ModelInput(kind="invalid", raw=raw, local_path=str(path))

    # Case-insensitive .onnx match keeps parity with the rest of the
    # CLI, which lowercases suffixes when sniffing file types.
    if value.lower().endswith(".onnx"):
        parts = [p for p in value.split("/") if p]
        if len(parts) >= 3:
            repo_id = "/".join(parts[:2])
            return ModelInput(kind="hub_onnx", raw=raw, hf_id=repo_id)
        return ModelInput(kind="invalid", raw=raw)

    return ModelInput(kind="hf_id", raw=raw, hf_id=value)


def resolve_model_input(
    value: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> ModelInput:
    """Classify + download Hub-hosted ONNX refs in one call.

    Equivalent to :func:`classify_model_input` for every kind except
    ``hub_onnx``, where the file is fetched via ``huggingface_hub`` and
    the returned :class:`ModelInput` has ``local_path`` populated with
    the cached path.

    Args:
        value: ``-m/--model`` value (HF id, local path, Hub ONNX ref).
        revision: Optional Hub revision forwarded for ``hub_onnx``.
        cache_dir: Optional cache override forwarded for ``hub_onnx``.
        token: Optional auth token forwarded for ``hub_onnx``.

    Returns:
        A :class:`ModelInput` with ``local_path`` populated whenever the
        kind implies a filesystem path.
    """
    mi = classify_model_input(value)
    if mi.kind != "hub_onnx":
        return mi

    # Lazy import: keeps huggingface_hub off the CLI startup path for
    # commands that never touch the Hub. Tests patch the downloader on
    # the loader package so the lookup picks up the mock at call time.
    from ..loader.onnx_hub import resolve_hf_onnx_path

    local = resolve_hf_onnx_path(
        value, revision=revision, cache_dir=cache_dir, token=token,
    )
    return replace(mi, local_path=str(local))


__all__ = [
    "ModelInput",
    "ModelInputKind",
    "classify_model_input",
    "resolve_model_input",
]
