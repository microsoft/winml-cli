# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Download runtime check rule zips from gim-home/ModelKitArtifacts.

Requires gh CLI with an account that has access to gim-home org.

Usage:
    uv run python scripts/download_rules.py --account <account>
    uv run python scripts/download_rules.py --account <account> --force
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SOURCE_REPO = "gim-home/ModelKitArtifacts"
SOURCE_PATH = "rules_zip"
RULES_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "winml"
    / "modelkit"
    / "analyze"
    / "rules"
    / "runtime_check_rules"
)


def _get_clone_url(account: str | None = None) -> str:
    """Build clone URL using gh account token."""
    gh_account = account or os.environ.get("GH_ACCOUNT")
    if not gh_account:
        print(
            "ERROR: gh account is required.\n"
            "Specify via --account or GH_ACCOUNT env var:\n"
            "  uv run python scripts/download_rules.py --account <account>\n"
            "  GH_ACCOUNT=<account> uv run python scripts/download_rules.py",
            file=sys.stderr,
        )
        sys.exit(1)

    if not shutil.which("gh"):
        print(
            "ERROR: gh CLI is not installed.\nInstall from https://cli.github.com",
            file=sys.stderr,
        )
        sys.exit(1)

    result = subprocess.run(
        ["gh", "auth", "token", "--user", gh_account],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"ERROR: Could not get token for account '{gh_account}'.\nRun 'gh auth login' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    token = result.stdout.strip()
    print(f"Using gh account: {gh_account}", flush=True)
    return f"https://x-access-token:{token}@github.com/{SOURCE_REPO}.git"


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
    parser.add_argument("--account", type=str, help="gh CLI account with access to gim-home org")
    args = parser.parse_args()

    clone_url = _get_clone_url(args.account)
    existing = set() if args.force else {f.name for f in RULES_DIR.glob("*.zip")}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "repo"
        print(f"Downloading rules from {SOURCE_REPO}...", flush=True)

        if not _sparse_clone(clone_url, tmp_path):
            print(
                f"ERROR: Failed to clone {SOURCE_REPO}.\n"
                "Make sure the GH_ACCOUNT has access to the gim-home org.",
                file=sys.stderr,
            )
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
