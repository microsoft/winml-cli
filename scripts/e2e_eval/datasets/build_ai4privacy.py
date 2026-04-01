# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Build a local HF-compatible dataset for Isotonic/distilbert_finetuned_ai4privacy_v2.

The ``ai4privacy/pii-masking-200k`` dataset uses string BIO labels
instead of ClassLabel integers.  This script converts them and renames
columns to match the token-classification eval pipeline.

Usage:
    python scripts/e2e_eval/datasets/build_ai4privacy.py --output <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path


_NUM_SAMPLES = 10000


def build_dataset(output_dir: Path) -> None:
    """Download, convert, and save the ai4privacy PII dataset."""
    if (output_dir / "dataset_info.json").exists():
        print(f"Dataset already exists at {output_dir}, skipping build.")
        return

    from datasets import ClassLabel, Dataset, Features, Sequence, Value, load_dataset

    print(f"Loading ai4privacy/pii-masking-200k ({_NUM_SAMPLES} samples) ...")
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=True)
    samples = list(ds.take(_NUM_SAMPLES))

    # Collect all unique labels
    all_labels = sorted({lbl for s in samples for lbl in s["mbert_bio_labels"]})
    print(f"Found {len(all_labels)} unique labels")
    label2id = {lbl: i for i, lbl in enumerate(all_labels)}

    # Convert to integer labels
    tokens_list = [s["mbert_text_tokens"] for s in samples]
    tags_list = [[label2id[lbl] for lbl in s["mbert_bio_labels"]] for s in samples]

    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=all_labels)),
        }
    )
    dataset = Dataset.from_dict(
        {"tokens": tokens_list, "ner_tags": tags_list},
        features=features,
    )
    print(f"Saving {len(dataset)} samples to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print("Done.")


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(description="Build ai4privacy PII dataset")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    build_dataset(args.output)


if __name__ == "__main__":
    main()
