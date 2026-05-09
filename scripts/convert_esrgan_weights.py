# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Convert ai-forever/Real-ESRGAN .pth weights to PreTrainedModel format.

Usage:
    uv run python scripts/convert_esrgan_weights.py --scale 4 --output-dir temp/esrgan-x4
"""

import argparse

import torch
from huggingface_hub import hf_hub_download

from winml.modelkit.models.hf.esrgan import (
    ESRGANConfig,
    ESRGANForImageSuperResolution,
)


def convert(scale: int, output_dir: str) -> None:
    """Download Real-ESRGAN .pth and re-save as a PreTrainedModel directory."""
    filename = f"RealESRGAN_x{scale}.pth"
    pth_path = hf_hub_download("ai-forever/Real-ESRGAN", filename)

    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]

    config = ESRGANConfig(scale=scale)
    model = ESRGANForImageSuperResolution(config)
    model.load_state_dict(state, strict=True)
    model.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, choices=[2, 4, 8], default=4)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    convert(args.scale, args.output_dir)
