"""Build a local HF-compatible object-detection dataset from PubTables-1M.

Downloads the validation split annotations (5 MB) and images (7 GB) from
``bsmock/pubtables-1m`` on HuggingFace, samples a small subset, and converts
PASCAL VOC XML annotations into the same schema used by
``detection-datasets/coco``.

Shared by:
- ``microsoft/table-transformer-detection``
- ``TahaDouaji/detr-doc-table-detection``

Usage:
    python scripts/e2e_eval/datasets/build_pubtables1m_detection.py --output <dir>
"""

from __future__ import annotations

import argparse
import random
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Labels matching microsoft/table-transformer-detection config
LABEL_NAMES = ["table", "table rotated"]

_NUM_SAMPLES = 1000
_SEED = 42


def _parse_voc_xml(xml_bytes: bytes) -> dict | None:
    """Parse a PASCAL VOC XML annotation and return structured data."""
    root = ET.fromstring(xml_bytes)
    filename = root.findtext("filename")
    size_el = root.find("size")
    if size_el is None or filename is None:
        return None
    width = int(float(size_el.findtext("width", "0")))
    height = int(float(size_el.findtext("height", "0")))
    if width == 0 or height == 0:
        return None

    bbox_ids: list[int] = []
    categories: list[int] = []
    bboxes: list[list[float]] = []
    areas: list[float] = []

    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}

    for i, obj in enumerate(root.findall("object")):
        name = obj.findtext("name", "").strip()
        cat_id = label2id.get(name)
        if cat_id is None:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        bbox_ids.append(i)
        categories.append(cat_id)
        bboxes.append([xmin, ymin, xmax, ymax])
        areas.append((xmax - xmin) * (ymax - ymin))

    if not bboxes:
        return None

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "bbox_id": bbox_ids,
        "category": categories,
        "bbox": bboxes,
        "area": areas,
    }


def build_dataset(output_dir: Path) -> None:
    """Download PubTables-1M val split, sample, and save as HF dataset."""
    if (output_dir / "dataset_info.json").exists():
        print(f"Dataset already exists at {output_dir}, skipping build.")
        return

    from huggingface_hub import hf_hub_download
    from PIL import Image

    repo_id = "bsmock/pubtables-1m"

    # Step 1: Download annotations tar (5 MB, cached by hf_hub)
    print("Downloading PubTables-1M detection annotations (val) ...")
    ann_tar_path = hf_hub_download(
        repo_id,
        "PubTables-1M-Detection_Annotations_Val.tar.gz",
        repo_type="dataset",
    )

    # Step 2: Parse all annotation XMLs
    print("Parsing PASCAL VOC annotations ...")
    all_annotations: list[dict] = []
    with tarfile.open(ann_tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".xml"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            parsed = _parse_voc_xml(f.read())
            if parsed is not None:
                all_annotations.append(parsed)

    print(f"  Parsed {len(all_annotations)} valid annotations")

    # Step 3: Sample a subset
    rng = random.Random(_SEED)
    sampled = rng.sample(all_annotations, min(_NUM_SAMPLES, len(all_annotations)))
    needed_filenames = {ann["filename"] for ann in sampled}
    print(f"  Sampled {len(sampled)} annotations, need {len(needed_filenames)} images")

    # Step 4: Download images tar (7 GB, cached by hf_hub)
    print("Downloading PubTables-1M detection images (val) — ~7 GB, one-time ...")
    img_tar_path = hf_hub_download(
        repo_id,
        "PubTables-1M-Detection_Images_Val.tar.gz",
        repo_type="dataset",
    )

    # Step 5: Extract only needed images from tar
    print("Extracting sampled images from tar ...")
    images_by_name: dict[str, Image.Image] = {}
    with tarfile.open(img_tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            basename = Path(member.name).name
            if basename in needed_filenames:
                f = tar.extractfile(member)
                if f is not None:
                    img = Image.open(f).convert("RGB")
                    # Load into memory so the tar file handle can be released
                    img.load()
                    images_by_name[basename] = img
                    if len(images_by_name) >= len(needed_filenames):
                        break

    print(f"  Extracted {len(images_by_name)} images")

    # Step 6: Build dataset rows (skip any missing images)
    from datasets import ClassLabel, Dataset, Features, Image as HFImage, Sequence, Value

    rows: list[dict] = []
    for idx, ann in enumerate(sampled):
        img = images_by_name.get(ann["filename"])
        if img is None:
            continue
        rows.append({
            "image_id": idx,
            "image": img,
            "width": ann["width"],
            "height": ann["height"],
            "objects": {
                "bbox_id": ann["bbox_id"],
                "category": ann["category"],
                "bbox": ann["bbox"],
                "area": ann["area"],
            },
        })

    features = Features({
        "image_id": Value("int64"),
        "image": HFImage(),
        "width": Value("int64"),
        "height": Value("int64"),
        "objects": {
            "bbox_id": Sequence(Value("int64")),
            "category": Sequence(ClassLabel(names=LABEL_NAMES)),
            "bbox": Sequence(Sequence(Value("float64"), length=4)),
            "area": Sequence(Value("float64")),
        },
    })

    dataset = Dataset.from_list(rows, features=features)
    print(f"Saving {len(dataset)} samples to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PubTables-1M detection dataset"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    build_dataset(args.output)


if __name__ == "__main__":
    main()
