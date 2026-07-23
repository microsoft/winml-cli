"""Verify the ONNX tiled VAE decode matches PyTorch AutoencoderKLWan.decode."""
import numpy as np
import torch
from diffusers import AutoencoderKLWan

from run_vae_onnx import OrtVaeDecoder

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
LAT_FRAMES = (81 - 1) // 4 + 1  # 21
LAT_H, LAT_W = 60, 104


def main():
    torch.manual_seed(0)
    z = torch.randn(1, 16, LAT_FRAMES, LAT_H, LAT_W, dtype=torch.float32)

    print("PyTorch tiled decode (reference) ...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32).eval()
    vae.enable_tiling()
    with torch.no_grad():
        ref = vae.decode(z, return_dict=False)[0].float().cpu().numpy()

    print("ONNX Runtime tiled decode ...")
    dec = OrtVaeDecoder()
    print("providers:", dec.sess.get_providers())
    out = dec.decode(z).float().cpu().numpy()

    diff = np.abs(out - ref)
    denom = np.abs(ref).mean()
    print(f"shapes ref={ref.shape} onnx={out.shape}")
    print(f"max_abs_diff={diff.max():.5f}  mean_abs_diff={diff.mean():.6f}  "
          f"mean_abs_ref={denom:.4f}  rel={diff.mean() / denom:.4%}")


if __name__ == "__main__":
    main()
