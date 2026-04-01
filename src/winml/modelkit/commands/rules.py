# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rules command for ModelKit CLI.

Download and manage runtime check rule files hosted on GitHub Releases.

Usage:
    winml rules download          # download missing/changed rule zips
    winml rules download --force  # re-download all rule zips
    winml rules status            # check which files are missing or outdated
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import click


logger = logging.getLogger(__name__)

_RULES_DIR = Path(__file__).resolve().parent.parent / "analyze" / "rules" / "runtime_check_rules"
_MANIFEST_PATH = _RULES_DIR / "rules_manifest.json"


def _load_manifest() -> dict:
    """Load rules_manifest.json."""
    with open(_MANIFEST_PATH, encoding="utf-8") as f:  # noqa: PTH123
        return json.load(f)


def _sha256_file(path: Path) -> str:
    """Compute sha256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:  # noqa: PTH123
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(repo: str, tag: str, filename: str, dest: Path) -> None:
    """Download a release asset, using gh CLI for private repos with fallback to direct URL.

    Args:
        repo: GitHub repo in owner/repo format
        tag: Release tag
        filename: Asset filename
        dest: Local destination path
    """
    # Try gh CLI first (handles authentication for private repos)
    gh_path = shutil.which("gh")
    if gh_path:
        result = subprocess.run(  # noqa: S603
            [
                gh_path,
                "release",
                "download",
                tag,
                "--repo",
                repo,
                "--pattern",
                filename,
                "--dir",
                str(dest.parent),
                "--clobber",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        logger.debug("gh CLI download failed: %s", result.stderr.strip())

    # Fallback to direct URL (works for public repos)
    url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
    urllib.request.urlretrieve(url, dest)  # noqa: S310


@click.group()
def rules() -> None:
    """Download and manage runtime check rule files."""


@rules.command()
@click.option("--force", is_flag=True, help="Re-download all files regardless of local state")
def download(force: bool) -> None:
    """Download runtime check rule zips from GitHub Releases."""
    manifest = _load_manifest()
    repo = manifest["github_repo"]
    tag = manifest["release_tag"]
    files = manifest["files"]

    _RULES_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    errors = 0
    for filename, info in files.items():
        local_path = _RULES_DIR / filename
        expected_hash = info["sha256"]

        if not force and local_path.exists() and _sha256_file(local_path) == expected_hash:
            skipped += 1
            continue

        size_mb = info["size"] / (1024 * 1024)
        click.echo(f"Downloading {filename} ({size_mb:.1f} MB)...")

        try:
            _download_file(repo, tag, filename, local_path)
        except Exception as e:
            click.echo(f"  ERROR: {e}", err=True)
            errors += 1
            continue

        actual_hash = _sha256_file(local_path)
        if actual_hash != expected_hash:
            click.echo(
                f"  WARNING: Hash mismatch for {filename} "
                f"(expected {expected_hash[:12]}..., got {actual_hash[:12]}...)",
                err=True,
            )
            errors += 1
        else:
            downloaded += 1

    click.echo(f"Done. Downloaded: {downloaded}, skipped: {skipped}, errors: {errors}")
    if errors:
        sys.exit(1)


@rules.command()
def status() -> None:
    """Check which rule files are missing or outdated."""
    manifest = _load_manifest()
    files = manifest["files"]

    missing = []
    outdated = []
    ok = []
    for filename, info in files.items():
        local_path = _RULES_DIR / filename
        if not local_path.exists():
            missing.append(filename)
        elif _sha256_file(local_path) != info["sha256"]:
            outdated.append(filename)
        else:
            ok.append(filename)

    if missing:
        click.echo(f"Missing ({len(missing)}):")
        for f in missing:
            click.echo(f"  {f}")
    if outdated:
        click.echo(f"Outdated ({len(outdated)}):")
        for f in outdated:
            click.echo(f"  {f}")
    if ok:
        click.echo(f"Up-to-date: {len(ok)}")

    if missing or outdated:
        click.echo("\nRun 'winml rules download' to fetch missing/outdated files.")
        sys.exit(1)
    else:
        click.echo("All rule files are up-to-date.")


@rules.command("cache-key")
def cache_key() -> None:
    """Print a short hash of the manifest for use as a CI cache key."""
    from ..analyze.utils.rule_loader import manifest_cache_key

    key = manifest_cache_key()
    if key:
        click.echo(key)
    else:
        click.echo("no-manifest", err=True)
        sys.exit(1)
