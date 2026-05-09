# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Validate ESRGAN port: ONNX export + WinMLAutoModel + HF pipeline inference.

Success criteria:
  1. Export ESRGAN to ONNX via WinMLAutoModel.from_pretrained.
  2. WinMLAutoModel returns a WinMLModelForImageToImage instance.
  3. HF pipeline("image-to-image", model=...) produces an upsampled image.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from transformers import pipeline

from winml.modelkit import WinMLAutoModel
from winml.modelkit.models.winml import WinMLModelForImageToImage


MODEL_ID = "temp/esrgan-x4"
HF_PROCESSOR_ID = "caidas/swin2SR-classical-sr-x4-64"
INPUT_IMAGE = "temp/esrgan_validate/lr.png"
OUTPUT_DIR = Path("temp/esrgan_validate")


def _ensure_input_image() -> Path:
    """Create a small RGB test image if none exists."""
    p = Path(INPUT_IMAGE)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(127, 64, 200)).save(p)
    return p


def main() -> None:
    """Run all three validation criteria."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lr_path = _ensure_input_image()

    # --- Criterion 1: export to ONNX ---
    # Pin shape to match Swin2SR processor output (pads 64x64 -> 72x72).
    model = WinMLAutoModel.from_pretrained(
        MODEL_ID,
        task="image-to-image",
        device="cpu",
        precision="fp32",
        shape_config={"height": 72, "width": 72},
    )
    onnx_path = getattr(model, "_onnx_path", None)
    print(f"[1] Exported ONNX: {onnx_path}")
    assert onnx_path, "model.onnx_path is unset"
    assert Path(onnx_path).exists(), f"ONNX not found: {onnx_path}"

    # --- Criterion 2: AutoModel returns image-to-image class ---
    assert isinstance(model, WinMLModelForImageToImage), type(model).__name__
    print(f"[2] AutoModel class: {type(model).__name__}")

    # --- Criterion 3: HF pipeline inference produces upsampled image ---
    lr = Image.open(lr_path).convert("RGB")
    pipe = pipeline("image-to-image", model=model, image_processor=HF_PROCESSOR_ID)
    sr = pipe(lr)
    sr_path = OUTPUT_DIR / "sr.png"
    sr.save(sr_path)
    assert sr.size[0] > lr.size[0], (lr.size, sr.size)
    assert sr.size[1] > lr.size[1], (lr.size, sr.size)
    print(f"[3] Upsampled {lr.size} -> {sr.size}, saved to {sr_path}")


if __name__ == "__main__":
    main()
