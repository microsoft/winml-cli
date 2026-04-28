# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Expand runtime rule zip snapshots into full JSON payloads."""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SNAPSHOT_TYPE_KEY = "__snapshot_type__"
SNAPSHOT_TYPE_DELTA = "delta_v1"
SNAPSHOT_BASE_OPSET_KEY = "__base_opset__"
SNAPSHOT_CHANGED_KEY = "__changed__"
SNAPSHOT_DELETED_KEY = "__deleted__"
EXPANDED_MARKER_FILE = "expanded"

_OPSET_TOKEN_PATTERN = re.compile(r"_opset\d+")


@dataclass
class ExpandSummary:
    """Summary of an expand run."""

    zip_files_processed: int
    zip_files_with_delta: int
    json_entries_processed: int
    delta_entries_materialized: int
    output_mode: str
    per_zip: list[tuple[str, int, int]]


def _replace_opset_token(name: str, new_opset: int) -> str:
    """Replace the first ``_opset<digits>`` token in a name."""
    return _OPSET_TOKEN_PATTERN.sub(f"_opset{new_opset}", name, count=1)


def _is_delta_snapshot(payload: Any) -> bool:
    """Check whether payload is a delta snapshot dict."""
    return isinstance(payload, dict) and payload.get(SNAPSHOT_TYPE_KEY) == SNAPSHOT_TYPE_DELTA


def _sorted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return payload sorted by top-level keys for stable output."""
    return dict(sorted(payload.items(), key=lambda item: item[0]))


def _apply_delta(base_payload: dict[str, Any], delta_payload: dict[str, Any]) -> dict[str, Any]:
    """Apply changed/deleted entries from delta payload onto base payload."""
    changed = delta_payload.get(SNAPSHOT_CHANGED_KEY, {})
    if not isinstance(changed, dict):
        raise TypeError(f"Delta snapshot field '{SNAPSHOT_CHANGED_KEY}' must be a dict.")

    deleted = delta_payload.get(SNAPSHOT_DELETED_KEY, [])
    if not isinstance(deleted, list):
        raise TypeError(f"Delta snapshot field '{SNAPSHOT_DELETED_KEY}' must be a list.")

    merged = dict(base_payload)
    merged.update(changed)

    for key in deleted:
        if not isinstance(key, str):
            raise TypeError(
                f"Delta snapshot field '{SNAPSHOT_DELETED_KEY}' contains non-string key: {key!r}"
            )
        merged.pop(key, None)

    return _sorted_payload(merged)


class SnapshotExpander:
    """Resolve and expand snapshot payloads stored in runtime rules zip files."""

    def __init__(self, rules_dir: Path) -> None:
        self.rules_dir = rules_dir
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _read_json_payload(self, zip_name: str, entry_name: str) -> dict[str, Any]:
        zip_path = self.rules_dir / zip_name
        if not zip_path.exists():
            raise FileNotFoundError(f"Base zip not found: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            try:
                raw = zf.read(entry_name)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Base entry '{entry_name}' not found in zip '{zip_name}'"
                ) from exc

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON entry '{entry_name}' in '{zip_name}'") from exc

        if not isinstance(payload, dict):
            raise TypeError(
                f"JSON entry '{entry_name}' in '{zip_name}' must be a dict, got {type(payload)}"
            )
        return payload

    def resolve_payload(
        self,
        zip_name: str,
        entry_name: str,
        stack: set[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Resolve an entry payload to full expanded contents."""
        token = (zip_name, entry_name)
        if token in self._cache:
            return self._cache[token]

        if stack is None:
            stack = set()
        if token in stack:
            raise ValueError(f"Detected cyclic snapshot dependency at {zip_name}::{entry_name}")

        stack.add(token)
        payload = self._read_json_payload(zip_name, entry_name)

        if _is_delta_snapshot(payload):
            base_opset = payload.get(SNAPSHOT_BASE_OPSET_KEY)
            if not isinstance(base_opset, int):
                raise ValueError(
                    f"Delta snapshot in {zip_name}::{entry_name} is missing integer "
                    f"'{SNAPSHOT_BASE_OPSET_KEY}'"
                )

            base_zip_name = _replace_opset_token(zip_name, base_opset)
            base_entry_name = _replace_opset_token(entry_name, base_opset)
            if base_zip_name == zip_name and base_entry_name == entry_name:
                raise ValueError(
                    "Could not derive base snapshot location for "
                    f"{zip_name}::{entry_name} (opset token not replaced)."
                )

            base_payload = self.resolve_payload(base_zip_name, base_entry_name, stack)
            resolved = _apply_delta(base_payload, payload)
        else:
            resolved = _sorted_payload(payload)

        stack.remove(token)
        self._cache[token] = resolved
        return resolved


def _expand_single_zip(
    zip_path: Path,
    dest_path: Path,
    expander: SnapshotExpander,
) -> tuple[int, int]:
    """Expand all delta JSON entries in one zip.

    Returns:
        Tuple of ``(json_entry_count, materialized_delta_count)``.
    """
    json_entries = 0
    materialized_entries = 0

    with zipfile.ZipFile(zip_path, "r") as src, zipfile.ZipFile(
        dest_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as dst:
        for entry in src.infolist():
            raw = src.read(entry.filename)
            out_raw = raw

            if entry.filename.endswith(".json"):
                json_entries += 1
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse JSON entry '{entry.filename}' in '{zip_path.name}'"
                    ) from exc

                if _is_delta_snapshot(payload):
                    materialized = expander.resolve_payload(zip_path.name, entry.filename)
                    out_raw = (json.dumps(materialized, indent=2) + "\n").encode("utf-8")
                    materialized_entries += 1

            dst.writestr(entry, out_raw)

    return json_entries, materialized_entries


def expand_rules_zip_dir(
    rules_dir: Path,
    *,
    output_dir: Path | None = None,
    glob_pattern: str = "*.zip",
    marker_filename: str = EXPANDED_MARKER_FILE,
) -> ExpandSummary:
    """Expand delta snapshots in rule zips.

    Args:
        rules_dir: Directory containing runtime rule zip files.
        output_dir: Optional output directory. If omitted, files are rewritten in place.
        glob_pattern: Zip filename pattern to process.
        marker_filename: Empty marker file name created after successful expand.

    Returns:
        ExpandSummary with per-zip stats.
    """
    rules_dir = rules_dir.resolve()
    if not rules_dir.exists() or not rules_dir.is_dir():
        raise FileNotFoundError(f"Rules directory not found: {rules_dir}")

    # Ignore stale temp artifacts from prior interrupted in-place runs.
    zip_files = [
        path
        for path in sorted(rules_dir.glob(glob_pattern), key=lambda p: p.name)
        if ".materialized." not in path.name
    ]
    if not zip_files:
        return ExpandSummary(
            zip_files_processed=0,
            zip_files_with_delta=0,
            json_entries_processed=0,
            delta_entries_materialized=0,
            output_mode=(
                f"in-place ({rules_dir})" if output_dir is None else f"copied to {output_dir}"
            ),
            per_zip=[],
        )

    output_mode = ""
    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_mode = f"copied to {output_dir}"
    else:
        output_mode = f"in-place ({rules_dir})"

    expander = SnapshotExpander(rules_dir)

    total_json = 0
    total_materialized = 0
    changed_zip_count = 0
    per_zip: list[tuple[str, int, int]] = []

    for zip_path in zip_files:
        if output_dir is None:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".zip",
                prefix=f"{zip_path.stem}.materialized.",
                dir=str(zip_path.parent),
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
            try:
                json_count, materialized_count = _expand_single_zip(
                    zip_path,
                    tmp_path,
                    expander,
                )
                tmp_path.replace(zip_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        else:
            dest = output_dir / zip_path.name
            json_count, materialized_count = _expand_single_zip(zip_path, dest, expander)

        per_zip.append((zip_path.name, json_count, materialized_count))
        total_json += json_count
        total_materialized += materialized_count
        if materialized_count > 0:
            changed_zip_count += 1

    target_dir = output_dir if output_dir is not None else rules_dir
    marker_path = target_dir / marker_filename
    marker_path.touch(exist_ok=True)

    return ExpandSummary(
        zip_files_processed=len(zip_files),
        zip_files_with_delta=changed_zip_count,
        json_entries_processed=total_json,
        delta_entries_materialized=total_materialized,
        output_mode=output_mode,
        per_zip=per_zip,
    )


# Backward-compatible aliases for existing imports.
MaterializeSummary = ExpandSummary
SnapshotMaterializer = SnapshotExpander


def materialize_rules_zip_dir(
    rules_dir: Path,
    *,
    output_dir: Path | None = None,
    glob_pattern: str = "*.zip",
) -> ExpandSummary:
    """Backward-compatible wrapper for previous API name."""
    return expand_rules_zip_dir(
        rules_dir,
        output_dir=output_dir,
        glob_pattern=glob_pattern,
    )
