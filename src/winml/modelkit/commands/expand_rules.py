# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""Expand runtime rule zips in-place for faster loading.

Resolves rules directories via ``_resolve_env_rules_dir_entry`` and, when
each directory exists and contains zip files, rewrites them in-place to full
payloads (no delta snapshot recursion at load time).

Usage:
    winml expand_rules
    winml expand_rules --rules-dir-entry C:\\path\\to\\rules_zip
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import click

from ..analyze.utils.rule_expander import expand_rules_zip_dir
from ..analyze.utils.rule_loader import MODELKIT_RULES_DIR_ENV, _resolve_env_rules_dir_entry


if TYPE_CHECKING:
    from pathlib import Path


def _entries_from_env() -> list[str]:
    """Read all non-empty MODELKIT_RULES_DIR entries from environment."""
    env_val = os.environ.get(MODELKIT_RULES_DIR_ENV, "").strip()
    if not env_val:
        return []
    return [entry.strip() for entry in env_val.split(os.pathsep) if entry.strip()]


def _resolve_entries(entries: list[str]) -> list[Path]:
    """Resolve entries to unique paths while preserving order."""
    resolved_dirs: list[Path] = []
    seen: set[str] = set()

    for entry in entries:
        resolved = _resolve_env_rules_dir_entry(entry)
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        resolved_dirs.append(resolved)

    return resolved_dirs


@click.command("expand_rules")
@click.option(
    "--rules-dir-entry",
    "rules_dir_entries",
    type=str,
    multiple=True,
    help=(
        "Optional rule directory entry. May be repeated. "
        "If omitted, uses all entries from MODELKIT_RULES_DIR. "
        "Each entry is resolved by rule_loader._resolve_env_rules_dir_entry."
    ),
)
@click.option(
    "--glob",
    "glob_pattern",
    type=str,
    default="*.zip",
    show_default=True,
    help="Zip filename glob to process.",
)
def expand_rules(rules_dir_entries: tuple[str, ...], glob_pattern: str) -> None:
    """Expand runtime rules zip files in-place when directories and zips exist."""
    entries = list(rules_dir_entries) if rules_dir_entries else _entries_from_env()

    if not entries:
        click.echo(
            f"{MODELKIT_RULES_DIR_ENV} is not set (or empty) "
            "and no --rules-dir-entry provided, skip."
        )
        return

    rules_dirs = _resolve_entries(entries)
    if not rules_dirs:
        click.echo("No resolvable rules directories found, skip.")
        return

    grand_zip_count = 0
    grand_delta_zip_count = 0
    grand_json_count = 0
    grand_delta_count = 0

    for rules_dir in rules_dirs:
        if not rules_dir.exists() or not rules_dir.is_dir():
            click.echo(f"Rules directory does not exist, skip: {rules_dir}")
            continue

        matched = [
            path
            for path in sorted(rules_dir.glob(glob_pattern), key=lambda p: p.name)
            if ".materialized." not in path.name
        ]
        if not matched:
            click.echo(f"No zip files matched '{glob_pattern}' in {rules_dir}, skip.")
            continue

        click.echo(f"Expanding {len(matched)} zip(s) in: {rules_dir}")

        summary = expand_rules_zip_dir(
            rules_dir,
            output_dir=None,
            glob_pattern=glob_pattern,
        )

        for zip_name, json_count, materialized_count in summary.per_zip:
            click.echo(
                f"[{zip_name}] json_entries={json_count}, "
                f"materialized_delta_entries={materialized_count}"
            )

        click.echo("\nDone.")
        click.echo(f"  zip_files_processed: {summary.zip_files_processed}")
        click.echo(f"  zip_files_with_delta: {summary.zip_files_with_delta}")
        click.echo(f"  json_entries_processed: {summary.json_entries_processed}")
        click.echo(f"  delta_entries_materialized: {summary.delta_entries_materialized}")
        click.echo(f"  output_mode: {summary.output_mode}")

        grand_zip_count += summary.zip_files_processed
        grand_delta_zip_count += summary.zip_files_with_delta
        grand_json_count += summary.json_entries_processed
        grand_delta_count += summary.delta_entries_materialized

    if grand_zip_count > 0 and len(rules_dirs) > 1:
        click.echo("\nAggregate:")
        click.echo(f"  zip_files_processed: {grand_zip_count}")
        click.echo(f"  zip_files_with_delta: {grand_delta_zip_count}")
        click.echo(f"  json_entries_processed: {grand_json_count}")
        click.echo(f"  delta_entries_materialized: {grand_delta_count}")
