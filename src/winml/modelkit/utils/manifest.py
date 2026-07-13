# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML manifest: machine-readable provenance for build and export outputs.

``build_manifest.json`` sits alongside every ONNX artifact produced by
``winml build`` or the Python-API equivalents.  It records *what* was
built, *how* (which pipeline stages ran), and *when*, so downstream
tools (``winml inspect``, ``winml serve``, the inference engine) can
discover model metadata without re-running the pipeline.

The :class:`WinMLManifest` dataclass is the single source of truth for the
manifest schema.  All producers **must** construct a ``WinMLManifest``
instance and call :meth:`save`; all consumers **should** use
:meth:`load` / :meth:`find` instead of hand-parsing the JSON.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "build_manifest.json"
"""Bare manifest filename (no cache-key prefix)."""

SCHEMA_VERSION = 1


def _sanitize_value(value: Any) -> Any:
    """Coerce non-JSON-native types to JSON-safe primitives.

    Handles ``Path`` (→ ``str``) and numpy scalars (→ native Python
    numbers/bools) so that numeric metrics are never accidentally
    serialised as strings by the ``default=str`` fallback.
    """
    from pathlib import PurePath

    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    # numpy scalars → native Python types (int, float, bool, etc.)
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    return value


def _sanitize_stage_dict(stage: dict[str, Any]) -> dict[str, Any]:
    """Compact a stage dict: drop ``None``, merge ``extras``, sanitize values."""
    extras = stage.pop("extras", {})
    d = {k: _sanitize_value(v) for k, v in stage.items() if v is not None}
    d.update({k: _sanitize_value(v) for k, v in extras.items()})
    return d


@dataclass
class ManifestStage:
    """One pipeline stage entry inside the manifest."""

    name: str
    status: str  # "completed" | "skipped"
    filename: str | None = None
    elapsed_seconds: float | None = None

    # Optional quantize metrics (populated only for the "quantize" stage).
    nodes_quantized: int | None = None
    nodes_skipped: int | None = None
    calibration_time_seconds: float | None = None
    qdq_insertion_time_seconds: float | None = None

    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class WinMLManifest:
    """Machine-readable provenance for a single ONNX artifact.

    Attributes:
        source: How the artifact was produced (``"hf"``, ``"onnx"``,
            ``"export"``).
        final_artifact: Filename of the deployment-ready ONNX
            (e.g. ``"model.onnx"``).
        stages: Ordered list of pipeline stages that ran (or were skipped).
        model_id: HuggingFace model ID or user-supplied label.
        task: Pipeline task (e.g. ``"image-classification"``).
        cache_key: Build-cache key (build pipelines only).
        config_hash: Trailing hash portion of *cache_key*.
        input_onnx: Path to the source ONNX file (``source="onnx"`` only).
        elapsed_seconds: Total wall-clock time in seconds.
        timestamp: ISO-8601 UTC timestamp of manifest creation.
        analyze_iterations: Number of optimize-analyze loop iterations.
        analyze_unsupported_node_count: Unsupported-node count from the
            final analysis pass.
        analyze_details: Free-form analysis metadata.
        export_stats: Statistics dict from HTPExporter (``source="export"``
            only).
        extras: Catch-all for forward-compatible fields that this version
            of the code does not model explicitly.
    """

    source: str
    final_artifact: str
    stages: list[ManifestStage] = field(default_factory=list)

    model_id: str | None = None
    task: str | None = None
    cache_key: str | None = None
    config_hash: str | None = None
    input_onnx: str | None = None
    elapsed_seconds: float | None = None
    timestamp: str | None = None

    # Analyze loop metadata
    analyze_iterations: int | None = None
    analyze_unsupported_node_count: int | None = None
    analyze_details: dict[str, Any] | None = None

    # Export-only
    export_stats: dict[str, Any] | None = None

    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (``schema_version`` injected)."""
        d: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        raw = asdict(self)
        extras = raw.pop("extras", {})
        # Drop None values for a compact JSON representation.
        d.update({k: _sanitize_value(v) for k, v in raw.items() if v is not None})
        # Stage entries: also drop None fields inside each stage, merge extras.
        if "stages" in d:
            d["stages"] = [_sanitize_stage_dict(s) for s in d["stages"]]
        d.update({k: _sanitize_value(v) for k, v in extras.items()})
        return d

    def save(self, path: Path) -> None:
        """Write the manifest as indented JSON to *path*."""
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        logger.debug("Manifest persisted: %s", path)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def manifest_path_for(
        output_dir: Path,
        prefix: str | None = None,
    ) -> Path:
        """Return the canonical manifest path in *output_dir*.

        When *prefix* is given (e.g. a cache key or composite-component
        stem), the filename becomes ``{prefix}_build_manifest.json``.
        """
        name = f"{prefix}_{MANIFEST_FILENAME}" if prefix else MANIFEST_FILENAME
        return output_dir / name

    @classmethod
    def load(cls, path: Path) -> WinMLManifest:
        """Deserialise from a JSON file on disk."""
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WinMLManifest:
        """Construct from a raw JSON dict (as stored on disk)."""
        data = dict(data)  # shallow copy — don't mutate caller's dict
        data.pop("schema_version", None)

        stages_raw = data.pop("stages", [])
        stage_known = set(ManifestStage.__dataclass_fields__) - {"extras"}
        stages = []
        for s in stages_raw:
            s = dict(s)  # shallow copy
            known_stage = {k: s.pop(k) for k in list(s) if k in stage_known}
            stages.append(ManifestStage(**known_stage, extras=s))

        known = set(cls.__dataclass_fields__) - {"stages", "extras"}
        known_kwargs = {k: data.pop(k) for k in list(data) if k in known}
        extras = data  # anything left is forward-compat extras

        return cls(stages=stages, extras=extras, **known_kwargs)

    @classmethod
    def find(cls, directory: Path) -> list[WinMLManifest]:
        """Discover and load all ``*build_manifest.json`` in *directory*."""
        results: list[WinMLManifest] = []
        for p in sorted(directory.glob(f"*{MANIFEST_FILENAME}")):
            m = cls._try_load(p)
            if m is not None:
                results.append(m)
        return results

    @classmethod
    def _try_load(cls, path: Path) -> WinMLManifest | None:
        """Load a manifest, returning ``None`` on read/parse failure."""
        try:
            return cls.load(path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable manifest %s: %s", path, exc)
            return None
