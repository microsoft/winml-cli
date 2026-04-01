# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Download runtime check rule zip files from GitHub Releases.

Thin wrapper around winml.modelkit.analyze.utils.rule_loader.

Usage:
    python scripts/download_rules.py [--force]
    python scripts/download_rules.py --check
"""

import argparse
import sys
from pathlib import Path


# Ensure the src directory is importable when running as a standalone script
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from winml.modelkit.analyze.utils.rule_loader import (  # noqa: E402
    check_rules_status,
    download_rules,
)


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
        missing, outdated, _ok = check_rules_status()
        needs_update = missing + outdated
        if needs_update:
            print(f"{len(needs_update)} file(s) need downloading:")
            for f in needs_update:
                print(f"  {f}")
            sys.exit(1)
        else:
            print("All rule files are up-to-date.")
            sys.exit(0)

    downloaded, skipped, errors = download_rules(force=args.force)
    print(f"Done. Downloaded: {downloaded}, skipped: {skipped}, errors: {errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
