# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Download runtime check rule zips from gim-home/ModelKitArtifacts.

Usage:
    uv run python scripts/download_rules.py          # download missing zips
    uv run python scripts/download_rules.py --force   # re-download all zips
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SOURCE_REPO = "gim-home/ModelKitArtifacts"
SOURCE_URL = f"https://github.com/{SOURCE_REPO}.git"
SOURCE_PATH = "op_check_results/rules"
RULES_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "winml"
    / "modelkit"
    / "analyze"
    / "rules"
    / "runtime_check_rules"
)


def _sparse_clone(clone_url: str, dest: Path) -> bool:
    """Sparse-clone only the rules folder. Returns True on success."""
    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            "main",
            clone_url,
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    result = subprocess.run(
        ["git", "sparse-checkout", "set", SOURCE_PATH],
        cwd=dest,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Download runtime check rule zips")
    parser.add_argument("--force", action="store_true", help="Re-download all zips")
    args = parser.parse_args()

    existing = set() if args.force else {f.name for f in RULES_DIR.glob("*.zip")}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "repo"
        print(f"Cloning {SOURCE_REPO} (sparse: {SOURCE_PATH})...")

        if not _sparse_clone(SOURCE_URL, tmp_path):
            msg = f"ERROR: Failed to clone {SOURCE_REPO}.\n"
            if not shutil.which("gh"):
                msg += (
                    "GitHub CLI (gh) is not installed.\n"
                    "Install from https://cli.github.com, then run:\n"
                    "  gh auth login\n"
                    "  gh auth setup-git\n"
                )
            else:
                msg += (
                    "Make sure git credentials are configured for the gim-home org.\n"
                    "Try running: gh auth setup-git\n"
                )
            print(msg, file=sys.stderr)
            sys.exit(1)

        src_dir = tmp_path / SOURCE_PATH
        zips = list(src_dir.glob("*.zip"))

        if not zips:
            print(f"No zip files found in {SOURCE_REPO}/{SOURCE_PATH}")
            sys.exit(1)

        RULES_DIR.mkdir(parents=True, exist_ok=True)
        copied = 0
        for zip_file in zips:
            if zip_file.name in existing:
                continue
            shutil.copy2(zip_file, RULES_DIR / zip_file.name)
            copied += 1

        total = len(zips)
        skipped = total - copied
        size_mb = sum((RULES_DIR / z.name).stat().st_size for z in zips) / 1024 / 1024
        print(f"Done. Copied: {copied}, skipped: {skipped}, total: {total} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
