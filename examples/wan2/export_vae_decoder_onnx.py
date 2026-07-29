"""Export the Wan2.1 VAE decoder (AutoencoderKLWan) to portable fp32 ONNX.

For text-to-video only the *decoder* is used (latent -> frames); the encoder is
skipped. The diffusers decode is a stateful, sequential process: it walks the
latent frames one chunk at a time, threading a causal temporal feature cache
(``feat_cache``) through 33 ``WanCausalConv3d`` layers, and -- when tiling is on
-- wraps that in a spatial tile loop with overlap blending.

We export the decode of a *single spatial tile* as one graph:

  * The 21-frame loop is **unrolled** inside the traced forward, so the temporal
    feature cache becomes ordinary intermediate tensors -- no 33-tensor cache to
    plumb in/out, and the ``first_chunk`` / cache-padding branches collapse to
    compile-time constants. This is numerically identical to PyTorch.
  * ``post_quant_conv`` is folded into the graph.
  * Spatial height/width are **dynamic** so the smaller edge tiles decode at
    their true size (exact parity with tiled_decode, no padding).

Because unrolling fixes the latent frame count, the graph has a frame
*capacity* rather than a dynamic frame axis. The decode is strictly **causal**
in time, though, so a shorter latent can simply be zero-padded up to the
capacity and the extra output frames dropped -- bit-exact, no seams. Use
``--lat-frames`` to build a graph for a different capacity.

The outer spatial tiling + blend loop stays in Python (see run_vae_onnx.py), so
all heavy compute (convs, group/RMS norms, upsamples) runs in ORT on any EP.
"""
import argparse
import os

import torch
from diffusers import AutoencoderKLWan
from torch.export import Dim

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
FP32_PATH = os.path.join(OUT_DIR, "wan_vae_decoder_fp32.onnx")
OPSET = 18

# 480P / 81 frames: 81 output frames -> 21 latent frames (temporal stride 4).
LAT_FRAMES = (81 - 1) // 4 + 1  # 21


class VaeDecoderTile(torch.nn.Module):
    """Decode one latent tile [1, 16, LAT_FRAMES, h, w] -> sample [1, 3, 81, 8h, 8w].

    Mirrors the inner loop of AutoencoderKLWan.tiled_decode for a single tile,
    with the frame loop unrolled and the causal cache kept in local lists so the
    dynamo exporter can trace it without module-attribute side effects.
    """

    def __init__(self, vae, num_latent_frames):
        super().__init__()
        self.vae = vae
        self.num_frames = num_latent_frames
        vae.clear_cache()
        self.conv_num = vae._conv_num

    def forward(self, z_tile):
        vae = self.vae
        feat_map = [None] * self.conv_num
        outs = []
        for k in range(self.num_frames):
            conv_idx = [0]
            tile = vae.post_quant_conv(z_tile[:, :, k : k + 1, :, :])
            decoded = vae.decoder(
                tile, feat_cache=feat_map, feat_idx=conv_idx, first_chunk=(k == 0))
            outs.append(decoded)
        return torch.cat(outs, dim=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat-frames", type=int, default=LAT_FRAMES,
                    help=f"latent frame capacity (default {LAT_FRAMES} -> "
                         f"{(LAT_FRAMES - 1) * 4 + 1} output frames)")
    ap.add_argument("--out", default=FP32_PATH, help="output .onnx path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print("Loading VAE (fp32) ...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32).eval()

    model = VaeDecoderTile(vae, args.lat_frames).eval()

    # A min-size latent tile (32x32) is enough to trace the graph; spatial dims
    # are exported dynamic so any tile size (incl. smaller edge tiles) works.
    z = torch.randn(1, vae.config.z_dim, args.lat_frames, 32, 32,
                    dtype=torch.float32)

    h = Dim("h", min=2, max=256)
    w = Dim("w", min=2, max=256)

    print(f"Exporting VAE decoder to ONNX fp32 "
          f"({args.lat_frames} latent frames -> "
          f"{(args.lat_frames - 1) * 4 + 1} output frames, dynamic H/W) ...")
    onnx_program = torch.onnx.export(
        model,
        (z,),
        input_names=["z_tile"],
        output_names=["sample_tile"],
        dynamic_shapes={"z_tile": {3: h, 4: w}},
        opset_version=OPSET,
        dynamo=True,
    )
    onnx_program.save(args.out, external_data=True)
    print(f"Saved fp32 VAE decoder ONNX to {args.out}")


if __name__ == "__main__":
    main()
