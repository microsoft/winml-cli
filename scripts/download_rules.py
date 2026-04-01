# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Download runtime check rule zip files from GitHub Releases.

Reads rules_manifest.json, compares sha256 hashes with local files,
and downloads only missing or changed zips from the configured GitHub Release.

Usage:
    python scripts/download_rules.py [--force]
    python scripts/download_rules.py --check  # verify only, no download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


logger = logging.getLogger(__name__)

RULES_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "winml"
    / "modelkit"
    / "analyze"
    / "rules"
    / "runtime_check_rules"
)
MANIFEST_PATH = RULES_DIR / "rules_manifest.json"


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    """Load and return the rules manifest."""
    with open(manifest_path, encoding="utf-8") as f:  # noqa: PTH123
        return json.load(f)


def sha256_file(path: Path) -> str:
    """Compute sha256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:  # noqa: PTH123
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(repo: str, tag: str, filename: str, dest: Path) -> None:
    """Download a release asset, using gh CLI for private repos with fallback to direct URL."""
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
        raise RuntimeError(f"gh CLI failed to download {filename}: {result.stderr.strip()}")

    # Fallback to direct URL (works for public repos only)
    url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    except urllib.error.HTTPError as e:
        if e.code in (404, 403):
            raise RuntimeError(
                f"Failed to download {filename} ({e.code}). "
                f"For private repos, install GitHub CLI: https://cli.github.com"
            ) from e
        raise


def check_rules(rules_dir: Path = RULES_DIR, manifest_path: Path = MANIFEST_PATH) -> list[str]:
    """Check which rule files are missing or have mismatched hashes.

    Returns:
        List of filenames that need downloading.
    """
    manifest = load_manifest(manifest_path)
    needs_download = []
    for filename, info in manifest["files"].items():
        local_path = rules_dir / filename
        if not local_path.exists() or sha256_file(local_path) != info["sha256"]:
            needs_download.append(filename)
    return needs_download


def download_rules(
    rules_dir: Path = RULES_DIR,
    manifest_path: Path = MANIFEST_PATH,
    force: bool = False,
) -> None:
    """Download rule zip files from GitHub Release.

    Args:
        rules_dir: Target directory for zip files
        manifest_path: Path to rules_manifest.json
        force: Re-download all files regardless of hash match
    """
    manifest = load_manifest(manifest_path)
    repo = manifest["github_repo"]
    tag = manifest["release_tag"]
    files = manifest["files"]

    rules_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    for filename, info in files.items():
        local_path = rules_dir / filename
        expected_hash = info["sha256"]

        if not force and local_path.exists() and sha256_file(local_path) == expected_hash:
            skipped += 1
            continue

        size_mb = info["size"] / (1024 * 1024)
        print(f"Downloading {filename} ({size_mb:.1f} MB)...")

        try:
            download_file(repo, tag, filename, local_path)
        except Exception as e:
            print(f"  ERROR: Failed to download {filename}: {e}", file=sys.stderr)
            continue

        actual_hash = sha256_file(local_path)
        if actual_hash != expected_hash:
            print(
                f"  WARNING: Hash mismatch for {filename} "
                f"(expected {expected_hash[:12]}..., got {actual_hash[:12]}...)",
                file=sys.stderr,
            )
        else:
            downloaded += 1

    print(f"Done. Downloaded: {downloaded}, skipped (up-to-date): {skipped}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Download runtime check rule files")
    parser.add_argument(
        "--force", action="store_true", help="Re-download all files regardless of local state"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which files need downloading (no download)",
    )
    args = parser.parse_args()

    if args.check:
        missing = check_rules()
        if missing:
            print(f"{len(missing)} file(s) need downloading:")
            for f in missing:
                print(f"  {f}")
            sys.exit(1)
        else:
            print("All rule files are up-to-date.")
            sys.exit(0)

    download_rules(force=args.force)


if __name__ == "__main__":
    main()
