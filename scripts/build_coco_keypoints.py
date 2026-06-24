# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Build a local COCO keypoints dataset for ``winml eval`` keypoint-detection.

Downloads the COCO person-keypoints annotations (cached once) and a chosen
number of validation images individually (so a small subset does not require
the full ~780 MB image zip), then writes an Arrow dataset to disk via
``datasets.Dataset.save_to_disk``. Point ``winml eval --dataset-path`` at the
output directory.

Each record has:
    - ``image``: the RGB image (datasets ``Image`` feature)
    - ``objects``: dict with parallel per-person lists ``keypoints`` (flat
      ``[x, y, v]`` triplets), ``bbox`` (COCO ``[x, y, w, h]``) and ``area``.

Only images containing at least one labeled-keypoint person are included.

Usage:
    uv run python scripts/build_coco_keypoints.py --output-dir ~/.cache/winml/datasets/coco_keypoints_val2017
    uv run python scripts/build_coco_keypoints.py --output-dir <dir> --num-images 100
    uv run python scripts/build_coco_keypoints.py --output-dir <dir> --num-images 0  # all images
"""

import argparse
import io
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATION_MEMBER = "annotations/person_keypoints_val2017.json"
IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{file_name}"

DEFAULT_CACHE = Path.home() / ".cache" / "winml" / "coco_build"


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` (skips if it already exists)."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh)


def _load_annotations(cache_dir: Path) -> dict:
    """Return the parsed person-keypoints annotation JSON, downloading once."""
    ann_zip = cache_dir / "annotations_trainval2017.zip"
    _download(ANNOTATIONS_URL, ann_zip)
    print("Reading keypoint annotations...")
    with zipfile.ZipFile(ann_zip) as zf, zf.open(ANNOTATION_MEMBER) as fh:
        return json.load(fh)


def _group_annotations_by_image(annotations: list[dict]) -> dict[int, list[dict]]:
    """Group person annotations by image id, keeping only labeled-keypoint people."""
    by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        if ann.get("num_keypoints", 0) <= 0 or ann.get("iscrowd", 0):
            continue
        by_image.setdefault(ann["image_id"], []).append(ann)
    return by_image


def _fetch_image(file_name: str) -> bytes:
    """Download one validation image and return its raw bytes."""
    url = IMAGE_URL_TEMPLATE.format(file_name=file_name)
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        return resp.read()


def build(output_dir: Path, num_images: int, cache_dir: Path, seed: int = 42) -> None:
    """Build and save the COCO keypoints dataset to ``output_dir``."""
    from datasets import Dataset, Features, Image, Sequence, Value
    from PIL import Image as PILImage

    coco = _load_annotations(cache_dir)
    images_by_id = {img["id"]: img for img in coco["images"]}
    by_image = _group_annotations_by_image(coco["annotations"])

    image_ids = sorted(by_image)
    if num_images > 0:
        # Shuffle before truncating so a small subset is a representative random
        # sample of the validation set rather than the lowest image ids. Seeded
        # so repeated builds produce the same subset.
        random.Random(seed).shuffle(image_ids)
        image_ids = image_ids[:num_images]
    print(f"Building {len(image_ids)} images with keypoint annotations...")

    records = []
    for idx, image_id in enumerate(image_ids, start=1):
        info = images_by_id[image_id]
        try:
            raw = _fetch_image(info["file_name"])
            image = PILImage.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            print(f"  skip {info['file_name']}: {exc}")
            continue

        persons = by_image[image_id]
        records.append(
            {
                "image": image,
                "objects": {
                    "keypoints": [[float(v) for v in p["keypoints"]] for p in persons],
                    "bbox": [[float(v) for v in p["bbox"]] for p in persons],
                    "area": [float(p["area"]) for p in persons],
                },
            }
        )
        if idx % 50 == 0:
            print(f"  {idx}/{len(image_ids)}")

    features = Features(
        {
            "image": Image(),
            "objects": {
                "keypoints": Sequence(Sequence(Value("float32"))),
                "bbox": Sequence(Sequence(Value("float32"))),
                "area": Sequence(Value("float32")),
            },
        }
    )
    dataset = Dataset.from_list(records, features=features)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print(f"Saved {len(dataset)} samples to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local COCO keypoints dataset.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Dataset output directory.")
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to include (0 = all images with keypoints).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Where to cache the downloaded annotations zip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting the image subset (used when --num-images > 0).",
    )
    args = parser.parse_args()
    build(args.output_dir, args.num_images, args.cache_dir, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
