# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Acquire versioned model archives declared by Hub ``release_assets.json`` files.

The resolver is intentionally metadata-driven.  It does not know model IDs or
archive member names: callers select a precision/format tuple from the manifest,
then inspect the safely extracted artifacts by their graph contracts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

_MANIFEST_NAME = "release_assets.json"
_PROVENANCE_NAME = "winml_release_provenance.json"
_RUNTIME_METADATA_NAME = "winml_release_metadata.json"
_MAX_EXTRACTED_BYTES = 20 * 1024**3


@dataclass(frozen=True)
class AcquiredReleaseAsset:
    """A safely extracted immutable release asset and its provenance."""

    root: Path
    manifest_path: Path
    metadata_path: Path | None
    provenance_path: Path
    provenance: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_asset(manifest: dict[str, Any], precision: str, format_name: str) -> dict[str, Any]:
    """Select exactly one release tuple, using float source for fp16 conversion."""
    source_precision = "float" if precision in {"fp32", "fp16"} else precision
    try:
        asset = manifest["precisions"][source_precision]["universal_assets"][format_name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Release manifest has no {format_name!r} asset for requested precision "
            f"{precision!r} (source precision {source_precision!r})."
        ) from error
    if not isinstance(asset, dict) or not isinstance(asset.get("download_url"), str):
        raise TypeError(
            f"Release manifest entry for {source_precision!r}/{format_name!r} must "
            "contain one string download_url."
        )
    return asset


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / "winml-release-assets"


def _download_archive(url: str, destination: Path) -> None:
    """Stream an archive to a temporary file and atomically publish it."""
    import httpx

    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Release asset download_url must be an absolute HTTPS URL: {url!r}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    stream.write(chunk)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    """Validate one ZIP member and return a normalized relative path."""
    raw = info.filename.replace("\\", "/")
    member = PurePosixPath(raw)
    mode = info.external_attr >> 16
    if (
        not raw
        or member.is_absolute()
        or any(part in {"", ".", ".."} for part in member.parts)
        or (member.parts and ":" in member.parts[0])
        or stat.S_ISLNK(mode)
    ):
        raise ValueError(f"Unsafe ZIP member path or type: {info.filename!r}")
    return member


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP without ``extractall`` and fail closed on unsafe members."""
    seen: set[PurePosixPath] = set()
    total_size = 0
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for info in members:
            member = _safe_member_path(info)
            if member in seen:
                raise ValueError(f"ZIP contains duplicate member {info.filename!r}.")
            seen.add(member)
            total_size += info.file_size
            if total_size > _MAX_EXTRACTED_BYTES:
                raise ValueError(
                    f"ZIP expands beyond the {_MAX_EXTRACTED_BYTES}-byte safety limit."
                )

        destination.mkdir(parents=True, exist_ok=False)
        root = destination.resolve()
        for info in members:
            member = _safe_member_path(info)
            target = destination.joinpath(*member.parts)
            resolved = target.resolve()
            if resolved != root and root not in resolved.parents:
                raise ValueError(f"ZIP member escapes extraction root: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def validate_onnx_external_data(path: Path, root: Path) -> tuple[Path, ...]:
    """Require every ONNX external-data location to be a present in-root file."""
    import onnx

    model = onnx.load(path, load_external_data=False)
    sidecars: list[Path] = []
    root_resolved = root.resolve()
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        fields = {field.key: field.value for field in initializer.external_data}
        location = fields.get("location")
        if not location:
            raise ValueError(f"External initializer in {path.name!r} has no location.")
        relative = PurePosixPath(location.replace("\\", "/"))
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or (relative.parts and ":" in relative.parts[0])
        ):
            raise ValueError(
                f"ONNX graph {path.name!r} has unsafe external-data location {location!r}."
            )
        sidecar = path.parent.joinpath(*relative.parts).resolve()
        if root_resolved not in sidecar.parents or not sidecar.is_file():
            raise FileNotFoundError(
                f"ONNX graph {path.name!r} requires missing or out-of-root external data "
                f"{location!r}."
            )
        sidecars.append(sidecar)
    return tuple(dict.fromkeys(sidecars))


def copy_release_contract_files(source_graph: Path, destination: Path) -> None:
    """Copy immutable release provenance/runtime metadata beside a built graph."""
    release_root = next(
        (
            parent
            for parent in (source_graph.parent, *source_graph.parents)
            if (parent / _PROVENANCE_NAME).is_file()
        ),
        None,
    )
    if release_root is None:
        return
    metadata_files = sorted(release_root.rglob("metadata.json"))
    if len(metadata_files) != 1:
        raise ValueError(
            "Release-backed ONNX graph requires exactly one metadata.json; "
            f"found {[str(path.relative_to(release_root)) for path in metadata_files]}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(release_root / _PROVENANCE_NAME, destination / _PROVENANCE_NAME)
    shutil.copy2(metadata_files[0], destination / _RUNTIME_METADATA_NAME)


def acquire_hf_release_asset(
    model_id: str,
    *,
    revision: str | None = None,
    precision: str = "fp32",
    format_name: str = "onnx",
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> AcquiredReleaseAsset | None:
    """Resolve and safely acquire a Hub repository's release archive.

    ``None`` means the repository does not publish ``release_assets.json``.
    A present but malformed manifest/archive always raises; it never falls back
    to another loader and thereby hides ambiguous or incomplete provenance.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    info = HfApi().model_info(model_id, revision=revision, token=token)
    resolved_revision = info.sha
    try:
        manifest_path = Path(
            hf_hub_download(
                repo_id=model_id,
                filename=_MANIFEST_NAME,
                revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
            )
        )
    except EntryNotFoundError:
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {_MANIFEST_NAME} for {model_id!r}: {error}") from error
    if not isinstance(manifest, dict):
        raise TypeError(f"{_MANIFEST_NAME} for {model_id!r} must contain a JSON object.")

    asset = _manifest_asset(manifest, precision, format_name)
    url = asset["download_url"]
    key_payload = json.dumps(
        [model_id, resolved_revision, precision, format_name, url], separators=(",", ":")
    )
    key = hashlib.sha256(key_payload.encode()).hexdigest()
    asset_root = _cache_root(cache_dir) / key
    archive_name = Path(urlparse(url).path).name or "release.zip"
    archive_path = asset_root / archive_name
    extracted = asset_root / "extracted"
    provenance_path = extracted / _PROVENANCE_NAME

    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("archive_sha256") == _sha256(archive_path):
            metadata_files = sorted(extracted.rglob("metadata.json"))
            return AcquiredReleaseAsset(
                root=extracted,
                manifest_path=manifest_path,
                metadata_path=metadata_files[0] if len(metadata_files) == 1 else None,
                provenance_path=provenance_path,
                provenance=provenance,
            )

    asset_root.mkdir(parents=True, exist_ok=True)
    if not archive_path.is_file():
        _download_archive(url, archive_path)

    staging = Path(tempfile.mkdtemp(prefix="extract-", dir=asset_root))
    try:
        shutil.rmtree(staging)
        safe_extract_zip(archive_path, staging)
        graph_paths = sorted(staging.rglob("*.onnx"))
        if not graph_paths:
            raise ValueError(f"Release archive {url!r} contains no ONNX graphs.")
        external_data = {
            str(path.relative_to(staging)): [
                str(sidecar.relative_to(staging))
                for sidecar in validate_onnx_external_data(path, staging)
            ]
            for path in graph_paths
        }
        provenance = {
            "schema_version": 1,
            "repo_id": model_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "manifest_sha256": _sha256(manifest_path),
            "manifest_version": manifest.get("version"),
            "requested_precision": precision,
            "source_precision": "float" if precision in {"fp32", "fp16"} else precision,
            "format": format_name,
            "download_url": url,
            "archive_sha256": _sha256(archive_path),
            "tool_versions": asset.get("tool_versions", {}),
            "graphs": {str(path.relative_to(staging)): _sha256(path) for path in graph_paths},
            "external_data": external_data,
        }
        (staging / _PROVENANCE_NAME).write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        if extracted.exists():
            shutil.rmtree(extracted)
        staging.replace(extracted)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    metadata_files = sorted(extracted.rglob("metadata.json"))
    if len(metadata_files) > 1:
        raise ValueError(
            f"Release archive contains multiple metadata.json files: "
            f"{[str(path.relative_to(extracted)) for path in metadata_files]}"
        )
    return AcquiredReleaseAsset(
        root=extracted,
        manifest_path=manifest_path,
        metadata_path=metadata_files[0] if metadata_files else None,
        provenance_path=provenance_path,
        provenance=provenance,
    )


__all__ = [
    "AcquiredReleaseAsset",
    "acquire_hf_release_asset",
    "copy_release_contract_files",
    "safe_extract_zip",
    "validate_onnx_external_data",
]
