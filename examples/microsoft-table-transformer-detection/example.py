# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Run one inference with the WinML-built ONNX and print detections.

Mirrors the HuggingFace ``TableTransformerForObjectDetection`` example
(https://huggingface.co/docs/transformers/main/en/model_doc/table-transformer)
but loads the quantized ONNX produced by ``winml build`` (step 1 of the
README) via :class:`WinMLAutoModel` instead of the original PyTorch
checkpoint.

Usage::

    uv run python examples/microsoft-table-transformer-detection/example.py `
      --onnx $HOME/.cache/winml/artifacts/microsoft_table-transformer-detection/`
            `objdet_<hash>_quantized.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoImageProcessor

from winml.modelkit import WinMLAutoModel


HF_MODEL_ID = "microsoft/table-transformer-detection"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        required=True,
        type=Path,
        help="Path to the quantized ONNX produced by step 1 of the README "
        "(e.g. objdet_<hash>_quantized.onnx).",
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
        "--threshold",
        type=float,
        default=0.9,
        help="Detection confidence threshold (default: 0.9).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Local image path. If omitted, downloads the example PDF page "
        "from the nielsr/example-pdf HuggingFace dataset (same image as "
        "the HF docs example).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("detections.png"),
        help="Where to write the annotated image (default: detections.png "
        "in the current directory).",
    )
    return parser.parse_args()


def draw_detections(
    image: Image.Image,
    results: dict,
    id2label: dict[int, str],
) -> Image.Image:
    """Draw bounding boxes and labels on a copy of ``image``."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, annotated.height // 60))
    except OSError:
        font = ImageFont.load_default()

    palette = [
        (220, 38, 38), (34, 197, 94), (59, 130, 246), (234, 179, 8),
        (168, 85, 247), (236, 72, 153), (20, 184, 166), (249, 115, 22),
    ]

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"], strict=True,
    ):
        x0, y0, x1, y1 = (round(v, 2) for v in box.tolist())
        label_id = label.item()
        color = palette[label_id % len(palette)]

        draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=3)

        caption = f"{id2label[label_id]} {score.item():.2f}"
        text_bbox = draw.textbbox((x0, y0), caption, font=font)
        tx0, ty0, tx1, ty1 = text_bbox
        # Anchor the caption above the box; flip below if it would clip.
        height = ty1 - ty0
        if ty0 - height < 0:
            ty0, ty1 = y0, y0 + height
        else:
            ty0, ty1 = y0 - height, y0
        draw.rectangle([(tx0, ty0), (tx1, ty1)], fill=color)
        draw.text((tx0, ty0), caption, fill="white", font=font)

    return annotated


def load_image(image_arg: Path | None) -> Image.Image:
    """Load the input image from disk or download the HF docs sample."""
    if image_arg is not None:
        return Image.open(image_arg.expanduser()).convert("RGB")
    sample_path = hf_hub_download(
        repo_id="nielsr/example-pdf",
        repo_type="dataset",
        filename="example_pdf.png",
    )
    return Image.open(sample_path).convert("RGB")


def main() -> None:
    """Load the quantized ONNX, run one inference, print detections."""
    args = parse_args()

    image = load_image(args.image)

    # HF processor handles resize/normalize and supplies post-processing.
    image_processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)

    # skip_build=True uses the ONNX as-is; it has already been optimized
    # and quantized by `winml build`. use_cache=False avoids touching the
    # winml artifact cache for this read-only example.
    model = WinMLAutoModel.from_pretrained(
        args.onnx.expanduser(),
        task="object-detection",
        device=args.device,
        ep=args.ep,
        skip_build=True,
        use_cache=False,
    )

    # Match the processor's output size to the ONNX's static input shape so
    # pixel_values matches (B, C, H, W) exactly. Mirrors the same handling
    # in the WinML object-detection evaluator.
    input_shapes = (model.io_config.get("input_shapes") or [[]])[0]
    input_names = model.io_config.get("input_names", [])
    if len(input_shapes) == 4:
        _, _, h, w = input_shapes
        if "pixel_mask" in input_names:
            image_processor.size = {
                "shortest_edge": min(h, w),
                "longest_edge": max(h, w),
            }
            if hasattr(image_processor, "pad_size"):
                image_processor.pad_size = {"height": h, "width": w}
            if hasattr(image_processor, "do_pad"):
                image_processor.do_pad = True
        else:
            image_processor.size = {"height": h, "width": w}
            if hasattr(image_processor, "do_pad"):
                image_processor.do_pad = False

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(
        pixel_values=inputs["pixel_values"],
        pixel_mask=inputs.get("pixel_mask"),
    )

    # post_process_object_detection expects outputs.logits and
    # outputs.pred_boxes (both torch tensors), which ObjectDetectionOutput
    # provides. target_sizes is (H, W) per image.
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs,
        threshold=args.threshold,
        target_sizes=target_sizes,
    )[0]

    # WinML's bare-ONNX path doesn't attach an HF config to the model, so
    # pull id2label from the HF hub for human-readable label names.
    id2label = AutoConfig.from_pretrained(HF_MODEL_ID).id2label

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"], strict=True,
    ):
        box = [round(v, 2) for v in box.tolist()]
        print(
            f"Detected {id2label[label.item()]} "
            f"with confidence {round(score.item(), 3)} at location {box}",
        )

    annotated = draw_detections(image, results, id2label)
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    print(f"Annotated image written to {output_path}")


if __name__ == "__main__":
    main()
