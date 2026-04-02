# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Build a local HF-compatible dataset for indonlp/indonlu (posp subset).

The upstream ``indonlp/indonlu`` dataset uses a legacy loading script
that is no longer supported by ``datasets >= 4.x``.  This script downloads
the auto-converted parquet files and saves them locally.

Usage:
    python scripts/e2e_eval/datasets/build_indonlu_posp.py --output <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path


_PARQUET_REVISION = "refs/convert/parquet"
_PARQUET_PATH = "posp/validation/0000.parquet"


def build_dataset(output_dir: Path) -> None:
    """Download parquet and save as a local HF dataset."""
    if (output_dir / "dataset_info.json").exists():
        print(f"Dataset already exists at {output_dir}, skipping build.")
        return

    from datasets import load_dataset

    print("Loading indonlp/indonlu posp validation from parquet ...")
    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/indonlp/indonlu@{_PARQUET_REVISION}/{_PARQUET_PATH}",
        split="train",
    )
    print(f"Loaded {len(ds)} samples")

    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build indonlu posp dataset")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    build_dataset(args.output)


if __name__ == "__main__":
    main()
