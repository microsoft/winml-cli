# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Fetch the model accuracy report from gim-home/ModelKitArtifacts.

For Microsoft internal use only. Requires gh CLI authenticated with an account
that has access to the gim-home org.

This script downloads the single, self-contained coverage report HTML page so
it can be published on the winml-cli documentation site. It pulls the file
directly from the `main` branch of gim-home/ModelKitArtifacts (under `site/`);
it does not download release assets. The upstream report carries a placeholder
version string, which this script stamps with the winml-cli version read from
pyproject.toml.

The report, this script, and the README live together in `docs/reports/` on
`main`. mkdocs publishes the HTML as a static asset under each docs version;
the script and README are excluded from the built site via `exclude_docs` in
`mkdocs.yml`. Refreshing the report is a normal docs change to `main`.

This script uses only the Python standard library, so it runs with a plain
`python` (no `uv` / project dependencies required).

Usage:
    python download_report.py --account <account>
    python download_report.py --account <account> --out <path>

By default the report is written next to this script, overwriting the
published copy (docs/reports/model_accuracy_report.html on `main`).

PUBLISHING (done by a maintainer via pull request):
    See README.md (co-located in this folder) for full instructions. After
    fetching, commit the refreshed report on a branch off `main` and open a PR:

        git add docs/reports/model_accuracy_report.html
        git commit -m "Update model accuracy report"

    Once merged, the docs workflow rebuilds and republishes the site.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib


SOURCE_REPO = "gim-home/ModelKitArtifacts"
SOURCE_BRANCH = "main"
SOURCE_FILE = "site/e2e_model_coverage_result/examples_compatibility_report.html"
# The upstream file name differs from the published name; publish under a
# stable name so the docs URL (.../reports/model_accuracy_report.html) never
# changes.
REPORT_FILENAME = "model_accuracy_report.html"
DEFAULT_OUT = Path(__file__).resolve().parent / REPORT_FILENAME
# winml-cli version source of truth. The upstream report ships a placeholder
# version string that is stamped with this value on fetch.
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _winml_cli_version() -> str:
    """Read the winml-cli version from pyproject.toml (stdlib only)."""
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def _get_clone_url(account: str | None = None) -> str:
    """Build clone URL using gh account token."""
    gh_account = account or os.environ.get("GH_ACCOUNT")
    if not gh_account:
        print(
            "ERROR: gh account is required.\n"
            "Specify via --account or GH_ACCOUNT env var:\n"
            "  uv run python docs/reports/download_report.py --account <account>\n"
            "  GH_ACCOUNT=<account> uv run python docs/reports/download_report.py\n"
            "\n"
            "This script is for Microsoft internal use (gim-home org access required).",
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
    """Sparse-clone only the single report file. Returns True on success."""
    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            SOURCE_BRANCH,
            clone_url,
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    result = subprocess.run(
        ["git", "sparse-checkout", "set", "--no-cone", SOURCE_FILE],
        cwd=dest,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the model accuracy report from gim-home/ModelKitArtifacts"
    )
    parser.add_argument("--account", type=str, help="gh CLI account with access to gim-home org")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output path for the fetched report (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    clone_url = _get_clone_url(args.account)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "repo"
        print(f"Downloading {SOURCE_FILE} from {SOURCE_REPO}@{SOURCE_BRANCH}...", flush=True)

        if not _sparse_clone(clone_url, tmp_path):
            print(
                f"ERROR: Failed to clone {SOURCE_REPO}@{SOURCE_BRANCH}.\n"
                "Make sure the GH_ACCOUNT has access to the gim-home org.",
                file=sys.stderr,
            )
            sys.exit(1)

        src_file = tmp_path / SOURCE_FILE
        if not src_file.is_file():
            print(
                f"ERROR: {SOURCE_FILE} not found in {SOURCE_REPO}@{SOURCE_BRANCH}.",
                file=sys.stderr,
            )
            sys.exit(1)

        version = _winml_cli_version()
        html, replaced = re.subn(
            r"(WinML CLI )\d+\.\d+\.\d+",
            rf"\g<1>{version}",
            src_file.read_text(encoding="utf-8"),
        )
        if replaced:
            print(f"Stamped report with WinML CLI {version}.")
        else:
            print(
                "WARNING: version placeholder 'WinML CLI x.y.z' not found in "
                "the report; published as-is.",
                file=sys.stderr,
            )

        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

        size_kb = out_path.stat().st_size / 1024
        print(f"Done. Wrote {out_path} ({size_kb:.0f} KB).")
        print(
            "\nNext: commit the refreshed report on a branch off main and open a PR "
            "(see the co-located README.md). The docs workflow republishes the site."
        )


if __name__ == "__main__":
    main()
