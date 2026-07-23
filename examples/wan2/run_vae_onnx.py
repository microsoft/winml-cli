"""Tiled VAE decode on ONNX Runtime, portable across execution providers.

Wraps the single-tile ONNX decoder (see export_vae_decoder_onnx.py) with the
spatial tiling + overlap-blend orchestration from AutoencoderKLWan.tiled_decode.
The heavy compute runs in ORT; only the cheap tiling/blend bookkeeping is Python.

The tiling constants below are the AutoencoderKLWan defaults for this model
(spatial compression 8, 256px min tile, 192px stride, patch_size=None).
"""
import os

import numpy as np
import onnxruntime as ort
import torch

FP32_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model", "wan_vae_decoder_fp32.onnx")

# AutoencoderKLWan defaults for Wan2.1-T2V-1.3B.
SPATIAL_COMP = 8
TILE_SAMPLE_MIN = 256
TILE_SAMPLE_STRIDE = 192
TILE_LATENT_MIN = TILE_SAMPLE_MIN // SPATIAL_COMP        # 32
TILE_LATENT_STRIDE = TILE_SAMPLE_STRIDE // SPATIAL_COMP  # 24
BLEND_EXTENT = TILE_SAMPLE_MIN - TILE_SAMPLE_STRIDE      # 64 (patch_size is None)


def _blend_v(a, b, blend_extent):
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    for y in range(blend_extent):
        b[:, :, :, y, :] = (
            a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent)
            + b[:, :, :, y, :] * (y / blend_extent))
    return b


def _blend_h(a, b, blend_extent):
    blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, :, x] = (
            a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent)
            + b[:, :, :, :, x] * (x / blend_extent))
    return b


class OrtVaeDecoder:
    """Portable tiled VAE decode backed by an ONNX Runtime session.

    Args:
        onnx_path: path to the single-tile decoder ONNX.
        providers: ORT execution providers, e.g. ``["DmlExecutionProvider"]`` or
            ``["OpenVINOExecutionProvider"]``. Defaults to CUDA then CPU.
    """

    def __init__(self, onnx_path=FP32_PATH, providers=None, gpu_mem_limit_gb=None):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # Optionally cap the CUDA arena so the decoder coexists with other GPU
        # sessions (e.g. the ONNX denoiser) without grabbing all the VRAM.
        if gpu_mem_limit_gb is not None:
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": int(gpu_mem_limit_gb * 1024 * 1024 * 1024),
                }) if p == "CUDAExecutionProvider" else p
                for p in providers
            ]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)

    def _decode_tile(self, z_tile):
        out = self.sess.run(["sample_tile"], {"z_tile": z_tile})[0]
        return torch.from_numpy(np.ascontiguousarray(out))

    def decode(self, z):
        """z: [1, 16, LAT_FRAMES, H, W] fp32 torch/np -> sample [1, 3, F, 8H, 8W]."""
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().float().numpy()
        z = np.ascontiguousarray(z, dtype=np.float32)
        _, _, _, height, width = z.shape
        sample_height, sample_width = height * SPATIAL_COMP, width * SPATIAL_COMP

        rows = []
        for i in range(0, height, TILE_LATENT_STRIDE):
            row = []
            for j in range(0, width, TILE_LATENT_STRIDE):
                tile = z[:, :, :, i : i + TILE_LATENT_MIN, j : j + TILE_LATENT_MIN]
                row.append(self._decode_tile(np.ascontiguousarray(tile)))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = _blend_v(rows[i - 1][j], tile, BLEND_EXTENT)
                if j > 0:
                    tile = _blend_h(row[j - 1], tile, BLEND_EXTENT)
                result_row.append(
                    tile[:, :, :, :TILE_SAMPLE_STRIDE, :TILE_SAMPLE_STRIDE])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
        return torch.clamp(dec, -1.0, 1.0)


def main():
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=21, help="latent frames (21 -> 81)")
    ap.add_argument("--lat-h", type=int, default=60)
    ap.add_argument("--lat-w", type=int, default=104)
    args = ap.parse_args()

    dec = OrtVaeDecoder()
    print("providers:", dec.sess.get_providers())
    z = torch.randn(1, 16, args.frames, args.lat_h, args.lat_w, dtype=torch.float32)
    t0 = time.time()
    out = dec.decode(z)
    print(f"decoded {tuple(out.shape)} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
