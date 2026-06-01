# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Run one image-classification inference with the WinML-built ONNX.

Mirrors the HuggingFace Swin Transformer usage example
(https://huggingface.co/docs/transformers/main/en/model_doc/swin) but
loads the quantized ONNX produced by ``winml build`` (step 1 of the
README) via :class:`WinMLAutoModel` instead of the original PyTorch
checkpoint.

The script preprocesses one image, runs inference, prints the top-5
predicted classes (HF-docs format), and writes an annotated image with
the top-1 label drawn in the corner so the result is visually
verifiable.

Usage::

    uv run python examples/microsoft_swin-large-patch4-window7-224/example.py `
      --onnx $HOME/.cache/winml/artifacts/microsoft_swin-large-patch4-window7-224/`
            `imgcls_<hash>_quantized.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoImageProcessor

from winml.modelkit import WinMLAutoModel


HF_MODEL_ID = "microsoft/swin-large-patch4-window7-224"
DEFAULT_DATASET = "timm/mini-imagenet"
DEFAULT_DATASET_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        required=True,
        type=Path,
        help="Path to the quantized ONNX produced by step 1 of the README "
        "(e.g. imgcls_<hash>_quantized.onnx).",
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
        help="Local image path. If omitted, streams the first image from "
        f"the {DEFAULT_DATASET} {DEFAULT_DATASET_SPLIT} split.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to print (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prediction.png"),
        help="Where to write the annotated image (default: prediction.png).",
    )
    return parser.parse_args()


def load_image(image_arg: Path | None) -> tuple[Image.Image, str | None]:
    """Load an image and (when streamed from the eval dataset) its WordNet synset.

    Returns ``(image, true_synset)``. ``true_synset`` is the WordNet ID
    (e.g. ``"n01532829"``) for the dataset's labelled class, used as the
    universal bridge between the dataset's class indexing and the model's.
    ``None`` when the user supplied a custom ``--image``.
    """
    if image_arg is not None:
        return Image.open(image_arg.expanduser()).convert("RGB"), None

    from datasets import load_dataset

    # streaming=True so we only fetch the first sample instead of downloading
    # the whole split. The ClassLabel feature (and its .names list) is still
    # available on the streamed dataset, so we can recover the WordNet synset
    # for the sample's integer label. trust_remote_code=False refuses to run
    # any dataset-bundled loading script.
    dataset = load_dataset(
        DEFAULT_DATASET,
        split=DEFAULT_DATASET_SPLIT,
        streaming=True,
        trust_remote_code=False,
    )
    sample = next(iter(dataset))

    image = sample["image"]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")

    label_value = sample.get("label")
    label_feature = dataset.features.get("label")
    if label_value is None or label_feature is None or not hasattr(label_feature, "names"):
        return image, None
    return image, label_feature.names[int(label_value)]


def imagenet_synset_to_id() -> dict[str, int]:
    """Map WordNet synset ID -> ImageNet-1k class id (0-999).

    Uses ``timm.data.ImageNetInfo`` so we don't have to ship the 1000-entry
    list inline. The mapping is the canonical ImageNet-1k ordering that
    the model was trained against.
    """
    from timm.data import ImageNetInfo

    info = ImageNetInfo()
    return {synset: idx for idx, synset in enumerate(info.label_names())}


def draw_top_prediction(
    image: Image.Image,
    label: str,
    score: float,
) -> Image.Image:
    """Draw the top-1 label + confidence on a copy of ``image``."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, annotated.height // 30))
    except OSError:
        font = ImageFont.load_default()

    caption = f"{label} ({score:.2f})"
    tx0, ty0, tx1, ty1 = draw.textbbox((10, 10), caption, font=font)
    pad = 6
    draw.rectangle(
        [(tx0 - pad, ty0 - pad), (tx1 + pad, ty1 + pad)],
        fill=(0, 0, 0),
    )
    draw.text((10, 10), caption, fill=(255, 255, 255), font=font)
    return annotated


def main() -> None:
    """Load the quantized ONNX, run one inference, print + save the result."""
    args = parse_args()

    image, true_synset = load_image(args.image)
    image_processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)

    # skip_build=True uses the ONNX as-is; it has already been optimized
    # and quantized by `winml build`. use_cache=False avoids touching the
    # winml artifact cache for this read-only example.
    model = WinMLAutoModel.from_pretrained(
        args.onnx.expanduser(),
        task="image-classification",
        device=args.device,
        ep=args.ep,
        skip_build=True,
        use_cache=False,
    )

    # Match the processor's output size to the ONNX's static input shape so
    # pixel_values matches (B, C, H, W) exactly.
    input_shapes = (model.io_config.get("input_shapes") or [[]])[0]
    # Only applies to 4D image inputs (B, C, H, W); skip for other shapes.
    if len(input_shapes) == 4:
        _, _, h, w = input_shapes
        image_processor.size = {"height": h, "width": w}

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(pixel_values=inputs["pixel_values"])

    # logits: (1, num_classes). softmax → probabilities, then top-k.
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[0]
    top_k = min(args.top_k, probs.numel())
    top_scores, top_ids = torch.topk(probs, k=top_k)

    # WinML's bare-ONNX path doesn't attach an HF config to the model, so
    # pull id2label from the HF hub for human-readable label names.
    id2label = AutoConfig.from_pretrained(HF_MODEL_ID).id2label

    top_ids_list = top_ids.tolist()
    top_label_names = [
        id2label.get(label_id, str(label_id)) for label_id in top_ids_list
    ]

    # Resolve the dataset's WordNet synset to an ImageNet-1k class id so we
    # can compare against the model's prediction. The dataset (e.g.
    # timm/mini-imagenet) often uses its own 0..N indexing over a subset of
    # ImageNet-1k, so the raw integer label from the dataset does NOT match
    # the model's class id — the synset is the universal bridge.
    true_label_id: int | None = None
    if true_synset is not None:
        synset_to_id = imagenet_synset_to_id()
        true_label_id = synset_to_id.get(true_synset)

    if true_synset is not None:
        if true_label_id is not None:
            true_label_name = id2label.get(true_label_id, str(true_label_id))
            print(f"True label:  {true_label_name} (synset={true_synset}, id={true_label_id})")
        else:
            print(f"True label:  synset={true_synset} (not in ImageNet-1k vocabulary)")
    else:
        print("True label:  unknown (custom --image)")
    print(f"\nTop {top_k} predictions:")
    for rank, (label, score) in enumerate(
        zip(top_label_names, top_scores.tolist(), strict=True), start=1,
    ):
        print(f"  {rank}. {label} ({score:.4f})")

    if true_label_id is not None:
        verdict = "PASS" if top_ids_list[0] == true_label_id else "FAIL"
        print(f"\nVerdict (top-1): {verdict}")

    annotated = draw_top_prediction(image, top_label_names[0], float(top_scores[0].item()))
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    print(f"\nAnnotated image written to {output_path}")


if __name__ == "__main__":
    main()
