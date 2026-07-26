# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Single classifier + resolver for ``-m/--model`` input values.

ModelKit accepts four shapes for a model reference:

* a HuggingFace model ID (``org/name``);
* a local ``.onnx`` file path;
* a Hub-hosted ONNX artifact (``org/repo/path/file.onnx``); and
* a local directory (a HuggingFace source folder or a ModelKit build output).

Historically the codebase had separate detectors for each form
(``is_hub_model``, ``is_hf_onnx_path``, ``is_onnx_file_path``, plus
scattered ``path.suffix == ".onnx"`` checks) *and* a second, CLI-only
classifier living in :mod:`winml.modelkit.utils.cli`. This module is now the
single classifier (:func:`classify_model_input`) and single resolver
(:func:`resolve_model_input`) for every layer, so adding a fifth input form
later means editing one function, not seven.

The classifier is **pure**: it performs no network I/O and never raises. An
input that cannot be used is returned with ``kind == ModelInputKind.INVALID``
and a human-readable :attr:`ModelInput.error` message that CLI callers can
surface via ``click.UsageError``. The resolver downloads Hub-hosted ONNX
artifacts via :func:`~winml.modelkit.loader.onnx_hub.resolve_hf_onnx_path` and
populates ``local_path`` on the returned :class:`ModelInput`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from .hub_utils import _is_local_path


# Matches a bare HuggingFace identifier: ``name`` or ``org/name`` where each
# component starts alphanumeric and may contain ``.`` ``_`` ``-``.
_HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


class ModelInputKind(str, Enum):
    """Discriminated classification of a ``-m/--model`` value.

    Subclasses ``str`` so members compare equal to their string value
    (``ModelInputKind.HUB_ONNX == "hub_onnx"``) while still supporting
    identity checks against members (``kind is ModelInputKind.ONNX_FILE``).
    This keeps both call-site styles working from a single enum.
    """

    ONNX_FILE = "local_onnx"  # an existing .onnx file on the local filesystem
    FOLDER = "build_dir"  # an existing directory (HF source or ModelKit build output)
    HUB_ONNX = "hub_onnx"  # org/repo/path/file.onnx -> needs Hub download
    HF_ID = "hf_id"  # HuggingFace model id (org/name or model-name)
    INVALID = "invalid"  # empty or unparsable

    def __str__(self) -> str:
        """Render as the bare value (``"hub_onnx"``) rather than ``ModelInputKind.HUB_ONNX``."""
        return self.value


@dataclass(frozen=True)
class ModelInput:
    """Classification result for a ``-m/--model`` value.

    Attributes:
        kind: One of :class:`ModelInputKind`.
        raw: The original user-supplied string (unchanged).
        local_path: Local filesystem path for ``ONNX_FILE`` / ``FOLDER``,
            or the cached download path for ``HUB_ONNX`` after
            :func:`resolve_model_input`. May also be populated for an
            ``INVALID`` local path (e.g. a mistyped file). ``None`` for
            ``HF_ID``.
        hf_id: HuggingFace repo id (``org/name``) for ``HF_ID`` /
            ``HUB_ONNX``. ``None`` otherwise.
        artifact_path: Path inside the Hub repository for ``HUB_ONNX``.
        revision: Immutable Hub commit used to download a discovered repository
            artifact, or the caller-provided revision for an explicit artifact.
        error: Human-readable reason the value is ``INVALID``. ``None`` for
            every valid kind.
    """

    kind: ModelInputKind
    raw: str
    local_path: str | None = None
    hf_id: str | None = None
    artifact_path: str | None = None
    revision: str | None = None
    error: str | None = None


def _classify_local_dir(path: Path, raw: str) -> ModelInput:
    """Classify an existing directory as a ``FOLDER`` input."""
    return ModelInput(kind=ModelInputKind.FOLDER, raw=raw, local_path=str(path))


def classify_model_input(value: str) -> ModelInput:
    r"""Classify a ``-m/--model`` value without any network I/O.

    Resolution order (first match wins):

    1. Empty / falsy → ``INVALID``.
    2. Looks like a local path (``_is_local_path``):
       * existing ``.onnx`` file → ``ONNX_FILE``;
       * existing non-``.onnx`` file → ``INVALID`` (unsupported file);
       * existing directory → ``FOLDER`` (with folder metadata);
       * non-existing ``.onnx`` path → ``ONNX_FILE`` (the loader surfaces the
         missing-file error later, keeping classification pure);
       * any other non-existing path → ``INVALID`` (path does not exist).
    3. Ends with ``.onnx`` and has >= 3 ``/``-separated components →
       ``HUB_ONNX``; a shorter ``.onnx`` ref → ``INVALID``.
    4. Path-shaped (``\\`` or >= 2 ``/`` components) → ``INVALID``.
    5. Valid HuggingFace identifier → ``HF_ID``; otherwise ``INVALID``.

    Local-path rejection in step 2 reuses ``_is_local_path`` (existing path,
    ``./``/``../``/``/``/``~/`` prefixes, Windows drive letters), the same
    logic used by ``is_hub_model``, so all forms apply identical rules.
    """
    if not value or not str(value).strip():
        return ModelInput(
            kind=ModelInputKind.INVALID,
            raw=value or "",
            error="Model input cannot be empty.",
        )

    raw = value

    if _is_local_path(value):
        path = Path(value).expanduser()
        if path.exists():
            if path.is_file():
                if path.suffix.lower() != ".onnx":
                    return ModelInput(
                        kind=ModelInputKind.INVALID,
                        raw=raw,
                        local_path=str(path),
                        error=(
                            f"Unsupported model file: '{value}'. Only .onnx files are supported."
                        ),
                    )
                return ModelInput(kind=ModelInputKind.ONNX_FILE, raw=raw, local_path=str(path))
            if path.is_dir():
                return _classify_local_dir(path, raw)

        # A local-path-shaped value that does not exist. Treat a ``.onnx``
        # suffix as a (missing) local ONNX file so downstream loaders raise a
        # precise error; anything else is an invalid path.
        if path.suffix.lower() == ".onnx":
            return ModelInput(kind=ModelInputKind.ONNX_FILE, raw=raw, local_path=str(path))
        return ModelInput(
            kind=ModelInputKind.INVALID,
            raw=raw,
            local_path=str(path),
            error=f"Model path does not exist: {value}",
        )

    # Case-insensitive .onnx match keeps parity with the rest of the CLI,
    # which lowercases suffixes when sniffing file types.
    if value.lower().endswith(".onnx"):
        parts = [p for p in value.split("/") if p]
        if len(parts) >= 3:
            repo_id = "/".join(parts[:2])
            return ModelInput(kind=ModelInputKind.HUB_ONNX, raw=raw, hf_id=repo_id)
        return ModelInput(
            kind=ModelInputKind.INVALID,
            raw=raw,
            error=f"ONNX file not found: {value}",
        )

    # Path-shaped but not an existing local path and not a Hub ONNX ref: a
    # backslash or three-plus segments means the user meant a filesystem path
    # that does not exist, not a HuggingFace id.
    if "\\" in value or value.count("/") >= 2:
        return ModelInput(
            kind=ModelInputKind.INVALID,
            raw=raw,
            error=f"Model path does not exist: {value}",
        )

    if not _HF_ID_RE.match(value):
        return ModelInput(
            kind=ModelInputKind.INVALID,
            raw=raw,
            error=(
                f"'{value}' is not a valid HuggingFace model identifier "
                "(expected 'name' or 'org/name')."
            ),
        )

    return ModelInput(kind=ModelInputKind.HF_ID, raw=raw, hf_id=value)


def resolve_model_input(
    value: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    discover_repo_onnx: bool = False,
) -> ModelInput:
    """Classify + download Hub-hosted ONNX refs in one call.

    Equivalent to :func:`classify_model_input` for every kind except
    ``HUB_ONNX``, where the file is fetched via ``huggingface_hub`` and
    the returned :class:`ModelInput` has ``local_path`` populated with
    the cached path.

    Args:
        value: ``-m/--model`` value (HF id, local path, Hub ONNX ref).
        revision: Optional Hub revision forwarded for ``HUB_ONNX``.
        cache_dir: Optional cache override forwarded for ``HUB_ONNX``.
        token: Optional auth token forwarded for ``HUB_ONNX``.

    Returns:
        A :class:`ModelInput` with ``local_path`` populated whenever the
        kind implies a filesystem path.
    """
    mi = classify_model_input(value)
    if mi.kind is ModelInputKind.HF_ID and discover_repo_onnx:
        from ..loader.onnx_hub import resolve_hf_repo_onnx

        discovered = resolve_hf_repo_onnx(
            value,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
        if discovered is not None:
            return ModelInput(
                kind=ModelInputKind.HUB_ONNX,
                raw=value,
                local_path=str(discovered.local_path),
                hf_id=discovered.repo_id,
                artifact_path=discovered.filename,
                revision=discovered.revision,
            )
        return mi
    if mi.kind is not ModelInputKind.HUB_ONNX:
        return mi

    # Lazy import: keeps huggingface_hub off the CLI startup path for
    # commands that never touch the Hub. Tests patch the downloader on
    # the loader package so the lookup picks up the mock at call time.
    from ..loader.onnx_hub import resolve_hf_onnx_path

    local = resolve_hf_onnx_path(value, revision=revision, cache_dir=cache_dir, token=token)
    parts = [part for part in value.split("/") if part]
    return replace(
        mi,
        local_path=str(local),
        artifact_path="/".join(parts[2:]),
        revision=revision,
    )


__all__ = [
    "ModelInput",
    "ModelInputKind",
    "classify_model_input",
    "resolve_model_input",
]
