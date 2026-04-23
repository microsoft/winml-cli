# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Materialize runtime rule zip snapshots into full JSON payloads.

This script resolves ``delta_v1`` snapshot chains and rewrites each JSON payload
as a full dictionary without snapshot metadata keys.

Usage:
    uv run python scripts/materialize_rules_zip.py --rules-dir <dir>
    uv run python scripts/materialize_rules_zip.py --rules-dir <dir> --output-dir <out>
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any


SNAPSHOT_TYPE_KEY = "__snapshot_type__"
SNAPSHOT_TYPE_DELTA = "delta_v1"
SNAPSHOT_BASE_OPSET_KEY = "__base_opset__"
SNAPSHOT_CHANGED_KEY = "__changed__"
SNAPSHOT_DELETED_KEY = "__deleted__"


_OPSET_TOKEN_PATTERN = re.compile(r"_opset\d+")


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
        raise ValueError(f"Delta snapshot field '{SNAPSHOT_CHANGED_KEY}' must be a dict.")

    deleted = delta_payload.get(SNAPSHOT_DELETED_KEY, [])
    if not isinstance(deleted, list):
        raise ValueError(f"Delta snapshot field '{SNAPSHOT_DELETED_KEY}' must be a list.")

    merged = dict(base_payload)
    merged.update(changed)

    for key in deleted:
        if not isinstance(key, str):
            raise ValueError(
                f"Delta snapshot field '{SNAPSHOT_DELETED_KEY}' contains non-string key: {key!r}"
            )
        merged.pop(key, None)

    return _sorted_payload(merged)


class SnapshotMaterializer:
    """Resolve and materialize snapshot payloads stored in runtime rules zip files."""

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
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to parse JSON entry '{entry_name}' in '{zip_name}'") from exc

        if not isinstance(payload, dict):
            raise ValueError(
                f"JSON entry '{entry_name}' in '{zip_name}' must be a dict, got {type(payload)}"
            )
        return payload

    def resolve_payload(
        self,
        zip_name: str,
        entry_name: str,
        stack: set[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Resolve an entry payload to full materialized contents."""
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


def _materialize_single_zip(
    zip_path: Path,
    dest_path: Path,
    materializer: SnapshotMaterializer,
) -> tuple[int, int]:
    """Materialize all delta JSON entries in one zip.

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
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(
                        f"Failed to parse JSON entry '{entry.filename}' in '{zip_path.name}'"
                    ) from exc

                if _is_delta_snapshot(payload):
                    materialized = materializer.resolve_payload(zip_path.name, entry.filename)
                    out_raw = (json.dumps(materialized, indent=2) + "\n").encode("utf-8")
                    materialized_entries += 1

            dst.writestr(entry, out_raw)

    return json_entries, materialized_entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize delta snapshots in runtime rule zips into full JSON payloads "
            "(remove baseline dependencies)."
        )
    )
    parser.add_argument(
        "--rules-dir",
        type=Path,
        required=True,
        help="Directory containing runtime rule zip files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for materialized zips. "
            "When omitted, zips are overwritten in-place."
        ),
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.zip",
        help="Filename glob used to select zip files (default: *.zip).",
    )
    args = parser.parse_args()

    rules_dir = args.rules_dir.resolve()
    if not rules_dir.exists() or not rules_dir.is_dir():
        raise FileNotFoundError(f"Rules directory not found: {rules_dir}")

    zip_files = sorted(rules_dir.glob(args.glob), key=lambda path: path.name)
    if not zip_files:
        print(f"No zip files matched '{args.glob}' in {rules_dir}")
        return

    output_dir: Path | None = None
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    materializer = SnapshotMaterializer(rules_dir)

    total_json = 0
    total_materialized = 0
    changed_zip_count = 0

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
                json_count, materialized_count = _materialize_single_zip(
                    zip_path,
                    tmp_path,
                    materializer,
                )
                tmp_path.replace(zip_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        else:
            dest = output_dir / zip_path.name
            json_count, materialized_count = _materialize_single_zip(zip_path, dest, materializer)

        total_json += json_count
        total_materialized += materialized_count
        if materialized_count > 0:
            changed_zip_count += 1

        print(
            f"[{zip_path.name}] json_entries={json_count}, "
            f"materialized_delta_entries={materialized_count}"
        )

    print("\nDone.")
    print(f"  zip_files_processed: {len(zip_files)}")
    print(f"  zip_files_with_delta: {changed_zip_count}")
    print(f"  json_entries_processed: {total_json}")
    print(f"  delta_entries_materialized: {total_materialized}")
    if output_dir is None:
        print(f"  output_mode: in-place ({rules_dir})")
    else:
        print(f"  output_mode: copied to {output_dir}")


if __name__ == "__main__":
    main()
