# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Spot-check SAM 3 ONNX against the reference outputs published by the
model authors in the onnx-community/sam3-tracker-ONNX model card.

Reference (from README.md of onnx-community/sam3-tracker-ONNX):
  Input: truck.jpg + point [[500, 375]] + label [[1]] (no boxes)
  Expected iou_scores: [0.9313147068023682, 0.037515610456466675, 0.5128555297851562]

Uses the same preprocessing as WinMLMaskGenerationEvaluator so this also
sanity-checks our pipeline end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from huggingface_hub import snapshot_download
from PIL import Image

# Import preprocessing helpers from our evaluator to keep the comparison
# consistent with the production code path.
from winml.modelkit.eval.mask_generation_evaluator import (
    _preprocess_image,
)


REFERENCE_IOU_SCORES = [0.9313147068023682, 0.037515610456466675, 0.5128555297851562]
TRUCK_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/"
    "resolve/main/truck.jpg"
)
POINT = (500.0, 375.0)


def main() -> None:
    snap = Path(
        snapshot_download(
            "onnx-community/sam3-tracker-ONNX",
            allow_patterns=["onnx/*_int8.onnx"],
        )
    )
    encoder_path = snap / "onnx" / "vision_encoder_int8.onnx"
    decoder_path = snap / "onnx" / "prompt_encoder_mask_decoder_int8.onnx"

    print(f"Encoder: {encoder_path}")
    print(f"Decoder: {decoder_path}")

    image_path = Path.home() / ".cache" / "winml" / "truck.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        print(f"Downloading {TRUCK_URL}")
        image_path.write_bytes(requests.get(TRUCK_URL, timeout=60).content)
    img = Image.open(image_path).convert("RGB")
    print(f"Image size: {img.size}")

    # Preprocess via the evaluator's helper (longest-side 1008 + pad + ImageNet norm)
    pixel_values, scale_x, scale_y = _preprocess_image(img)
    print(
        f"Preprocessed: pixel_values={pixel_values.shape}, "
        f"scale_x={scale_x:.4f}, scale_y={scale_y:.4f}"
    )

    # Encoder forward
    enc = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    enc_inputs = {"pixel_values": pixel_values}
    enc_names = [o.name for o in enc.get_outputs()]
    enc_out = enc.run(None, enc_inputs)
    emb = dict(zip(enc_names, enc_out))
    print(f"Encoder outputs: {[(k, v.shape) for k, v in emb.items()]}")

    # Build decoder inputs in point-prompt mode
    points = np.array(
        [[[[POINT[0] * scale_x, POINT[1] * scale_y]]]], dtype=np.float32
    )
    labels = np.array([[[1]]], dtype=np.int64)
    boxes = np.zeros((1, 0, 4), dtype=np.float32)

    dec = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    dec_inputs = {
        "input_points": points,
        "input_labels": labels,
        "input_boxes": boxes,
        **emb,
    }
    dec_out = dec.run(None, dec_inputs)
    dec_names = [o.name for o in dec.get_outputs()]
    out = dict(zip(dec_names, dec_out))

    iou = np.asarray(out["iou_scores"]).reshape(-1).tolist()
    print()
    print("Reference iou_scores : ", [f"{v:.6f}" for v in REFERENCE_IOU_SCORES])
    print("Our iou_scores       : ", [f"{v:.6f}" for v in iou])
    diffs = [a - b for a, b in zip(iou, REFERENCE_IOU_SCORES)]
    print("Absolute diff        : ", [f"{d:+.6f}" for d in diffs])
    print(f"Max abs diff         :  {max(abs(d) for d in diffs):.6f}")


if __name__ == "__main__":
    main()
