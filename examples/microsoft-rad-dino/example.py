# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Verify a RAD-DINO ONNX build via 1-NN classification on chest X-rays.

Mirrors the HuggingFace RAD-DINO usage
(https://huggingface.co/microsoft/rad-dino) but loads the quantized
ONNX produced by ``winml build`` (step 1 of the README) via
:class:`WinMLAutoModel` instead of the original PyTorch checkpoint.

How it tells you the model is working
-------------------------------------
RAD-DINO has no classification head — by itself it just emits image
embeddings. To prove the embeddings are meaningful, this script:

1. Pulls one query X-ray and a small labelled reference bank from the
   same chest X-ray dataset the README evaluates against.
2. Embeds each image with the ONNX model.
3. Ranks the bank by cosine similarity to the query and predicts the
   query's class via 1-nearest-neighbour.
4. Compares the prediction to the dataset's ground-truth label and
   prints **PASS** or **FAIL**.

PASS = the ONNX-built model still produces clinically meaningful
embeddings (PNEUMONIA X-rays cluster together, NORMAL X-rays cluster
together). FAIL on a single sample is not catastrophic — quantization
can fool any single example — but repeated FAILs across different
``--seed`` values mean the build is broken.

Usage::

    uv run python examples/microsoft-rad-dino/example.py `
      --onnx $HOME/.cache/winml/artifacts/microsoft_rad-dino/`
            `imgfeat_<hash>_quantized.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

from winml.modelkit import WinMLAutoModel


HF_MODEL_ID = "microsoft/rad-dino"
DEFAULT_DATASET = "Ewakaa/pneumonia_classification_chest_xray"
DEFAULT_DATASET_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        required=True,
        type=Path,
        help="Path to the quantized ONNX produced by step 1 of the README "
        "(e.g. imgfeat_<hash>_quantized.onnx).",
    )
    parser.add_argument(
        "--device",
        default="npu",
        choices=["auto", "npu", "gpu", "cpu"],
        help="Target device (default: npu).",
    )
    parser.add_argument(
        "--ep",
        default="openvino",
        help="Execution provider alias (default: openvino).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Local chest X-ray image to use as the query. If omitted, the "
        "first shuffled image from the eval dataset is used (and its ground-"
        "truth label is known, so the script can print PASS/FAIL).",
    )
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=6,
        help="Number of labelled reference images to embed for 1-NN "
        "classification of the query (default: 6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for selecting query + reference samples "
        "(default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("query_embedding.npy"),
        help="Where to save the 1-D query embedding "
        "(default: query_embedding.npy).",
    )
    return parser.parse_args()


def to_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    """Coerce a dataset sample's image field to a PIL RGB image."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return image.convert("RGB")


def load_query_and_bank(
    image_arg: Path | None,
    bank_size: int,
    seed: int,
) -> tuple[Image.Image, int | None, list[Image.Image], list[int], dict[int, str]]:
    """Pick the query image + a small labelled reference bank.

    Returns (query_image, query_label_id_or_None, bank_images,
    bank_label_ids, id_to_label_name).
    """
    from datasets import load_dataset

    ds = load_dataset(DEFAULT_DATASET, split=DEFAULT_DATASET_SPLIT)
    ds = ds.shuffle(seed=seed)

    # Resolve human-readable label names from the dataset features if they
    # are exposed as a ClassLabel; otherwise fall back to stringified IDs.
    label_feature = ds.features.get("label")
    if label_feature is not None and hasattr(label_feature, "names"):
        id_to_label = dict(enumerate(label_feature.names))
    else:
        id_to_label = {}

    if image_arg is not None:
        query_image = Image.open(image_arg.expanduser()).convert("RGB")
        query_label_id = None
        bank_slice = ds.select(range(bank_size))
    else:
        slice_ = ds.select(range(bank_size + 1))
        query_image = to_rgb(slice_[0]["image"])
        query_label_id = int(slice_[0]["label"])
        bank_slice = slice_.select(range(1, len(slice_)))

    bank_images = [to_rgb(s["image"]) for s in bank_slice]
    bank_label_ids = [int(s["label"]) for s in bank_slice]
    return query_image, query_label_id, bank_images, bank_label_ids, id_to_label


def main() -> None:
    """Load the ONNX, run query + bank inference, print 1-NN verdict."""
    args = parse_args()

    image_processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)

    # skip_build=True uses the ONNX as-is; it has already been optimized
    # and quantized by `winml build`. use_cache=False avoids touching the
    # winml artifact cache for this read-only example.
    model = WinMLAutoModel.from_pretrained(
        args.onnx.expanduser(),
        task="image-feature-extraction",
        device=args.device,
        ep=args.ep,
        skip_build=True,
        use_cache=False,
    )

    # Match the processor's output size to the ONNX's static input shape so
    # pixel_values matches (B, C, H, W) exactly. Mirrors the same handling
    # in the WinML image-feature-extraction evaluator.
    input_shapes = (model.io_config.get("input_shapes") or [[]])[0]
    if len(input_shapes) == 4:
        _, _, h, w = input_shapes
        image_processor.size = {"height": h, "width": w}

    def embed(img: Image.Image) -> np.ndarray:
        inputs = image_processor(images=img, return_tensors="pt")
        outputs = model(pixel_values=inputs["pixel_values"])
        hidden = outputs.last_hidden_state
        # CLS token for 3-D [1, num_tokens, hidden]; raw vector if already pooled.
        if hidden.dim() == 3:
            vec = hidden[0, 0, :]
        elif hidden.dim() == 2:
            vec = hidden[0, :]
        else:
            raise RuntimeError(f"Unexpected last_hidden_state shape: {tuple(hidden.shape)}")
        return vec.cpu().numpy()

    print("Loading query + reference bank from "
          f"{DEFAULT_DATASET} [{DEFAULT_DATASET_SPLIT}]...")
    query_image, query_label_id, bank_images, bank_label_ids, id_to_label = (
        load_query_and_bank(args.image, args.reference_samples, args.seed)
    )

    def name(label_id: int) -> str:
        return id_to_label.get(label_id, str(label_id))

    print(f"Embedding {1 + len(bank_images)} images...")
    query_emb = embed(query_image)
    bank_embs = np.stack([embed(img) for img in bank_images])

    # Cosine similarity between query and each reference.
    query_unit = query_emb / np.linalg.norm(query_emb)
    bank_unit = bank_embs / np.linalg.norm(bank_embs, axis=1, keepdims=True)
    sims = bank_unit @ query_unit
    order = np.argsort(-sims)

    print("\nReference bank, sorted by cosine similarity to the query:")
    print(f"  {'rank':<5} {'label':<14} {'cosine sim':>10}")
    for rank, i in enumerate(order, start=1):
        print(f"  {rank:<5} {name(bank_label_ids[i]):<14} {sims[i]:>10.4f}")

    predicted_label_id = bank_label_ids[order[0]]
    print(f"\nPredicted class (1-NN): {name(predicted_label_id)}")

    if query_label_id is not None:
        verdict = "PASS" if predicted_label_id == query_label_id else "FAIL"
        print(f"True class:             {name(query_label_id)}")
        print(f"Verdict:                {verdict}")
    else:
        print("True class:             unknown (custom --image)")

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, query_emb)
    print(f"\nQuery embedding saved to: {output_path}")


if __name__ == "__main__":
    main()
