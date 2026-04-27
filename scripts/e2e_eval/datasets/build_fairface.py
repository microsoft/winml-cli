# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Build a local HF-compatible dataset from nateraw/fairface.

The upstream ``nateraw/fairface`` dataset uses a legacy loading script
(``fairface.py``) that is no longer supported by ``datasets >= 4.x``.
This script downloads the raw ``val.pt`` pickle, converts ``img_bytes``
to PIL images, and saves a dataset with ``image``, ``age``, and ``gender``
columns.  Different models can then use ``--column label_column=age`` or
``--column label_column=gender`` to select the appropriate label.

Usage:
    python scripts/e2e_eval/datasets/build_fairface.py --output <dir>
"""

from __future__ import annotations

import argparse
import pickle
from io import BytesIO
from pathlib import Path


AGE_LABELS = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
]

GENDER_LABELS = ["female", "male"]


def build_dataset(output_dir: Path) -> None:
    """Download, convert, and save the fairface validation dataset."""
    if (output_dir / "dataset_info.json").exists():
        print(f"Dataset already exists at {output_dir}, skipping build.")
        return

    from collections import Counter

    from datasets import ClassLabel, Dataset, Features, Image
    from huggingface_hub import hf_hub_download
    from PIL import Image as PILImage

    print("Downloading nateraw/fairface val.pt ...")
    pt_path = hf_hub_download("nateraw/fairface", "val.pt", repo_type="dataset")

    print("Loading pickle ...")
    with Path(pt_path).open("rb") as f:
        data = pickle.load(f)  # noqa: S301

    print(f"Converting {len(data)} records ...")
    images = []
    ages = []
    genders = []
    skipped = 0
    for item in data:
        age_str = item["age"]
        gender_str = item["gender"].lower()
        if age_str not in AGE_LABELS or gender_str not in GENDER_LABELS:
            skipped += 1
            continue
        img = PILImage.open(BytesIO(item["img_bytes"])).convert("RGB")
        images.append(img)
        ages.append(AGE_LABELS.index(age_str))
        genders.append(GENDER_LABELS.index(gender_str))

    print(f"Converted {len(images)} samples (skipped {skipped})")
    print(f"Age distribution:    {dict(Counter(ages))}")
    print(f"Gender distribution: {dict(Counter(genders))}")

    features = Features(
        {
            "image": Image(),
            "age": ClassLabel(names=AGE_LABELS),
            "gender": ClassLabel(names=GENDER_LABELS),
        }
    )
    ds = Dataset.from_dict(
        {"image": images, "age": ages, "gender": genders},
        features=features,
    )
    print(f"Saving {len(ds)} samples to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print("Done.")


_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "winml" / "eval_datasets" / "build_fairface"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fairface validation dataset")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: ~/.cache/winml/eval_datasets/build_fairface)")
    args = parser.parse_args()
    output_dir = args.output or _DEFAULT_CACHE_DIR
    build_dataset(output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
