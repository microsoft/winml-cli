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
from pathlib import Path
import sys


def _load_materializer():
    """Load materializer utility from package, with src fallback in repo mode."""
    try:
        from winml.modelkit.analyze.utils.rule_expander import expand_rules_zip_dir

        return expand_rules_zip_dir
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent
        src_path = repo_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from winml.modelkit.analyze.utils.rule_expander import expand_rules_zip_dir

        return expand_rules_zip_dir


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
    expand_rules_zip_dir = _load_materializer()
    summary = expand_rules_zip_dir(
        args.rules_dir,
        output_dir=args.output_dir,
        glob_pattern=args.glob,
    )

    if not summary.per_zip:
        print(f"No zip files matched '{args.glob}' in {args.rules_dir.resolve()}")
        return

    for zip_name, json_count, materialized_count in summary.per_zip:
        print(
            f"[{zip_name}] json_entries={json_count}, "
            f"materialized_delta_entries={materialized_count}"
        )

    print("\nDone.")
    print(f"  zip_files_processed: {summary.zip_files_processed}")
    print(f"  zip_files_with_delta: {summary.zip_files_with_delta}")
    print(f"  json_entries_processed: {summary.json_entries_processed}")
    print(f"  delta_entries_materialized: {summary.delta_entries_materialized}")
    print(f"  output_mode: {summary.output_mode}")


if __name__ == "__main__":
    main()
