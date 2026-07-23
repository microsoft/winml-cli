"""Run the Wan2.1-T2V-1.3B pipeline end-to-end with the ONNX denoiser in ORT.

The T5 text encoder stays in PyTorch (it must not run in fp16). The diffusion
transformer -- the module executed on every denoising step -- is always replaced
by the fp16 ONNX graph running on ONNX Runtime (CUDA).

The VAE decode can run either in PyTorch (``--vae torch``, tiled) or, for a fully
portable path, on the ONNX decoder via ONNX Runtime (``--vae onnx``, the default;
see export_vae_decoder_onnx.py / run_vae_onnx.py).
"""
import argparse
import gc
import os
import time
from contextlib import contextmanager

import numpy as np
import onnxruntime as ort
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

from run_vae_onnx import OrtVaeDecoder

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FP16_PATH = os.path.join(os.path.dirname(__file__), "model", "wan_transformer_fp16.onnx")


class OrtTransformer:
    """Drop-in replacement for WanTransformer3DModel backed by an ORT session.

    Implements just the surface the WanPipeline touches: ``config``, ``dtype``,
    a no-op ``cache_context`` and a ``__call__`` returning ``(noise_pred,)``.
    """

    def __init__(self, onnx_path, config, dtype, device):
        self.config = config
        self.dtype = dtype
        self._device = device
        so = ort.SessionOptions()
        cuda_opts = {
            "device_id": 0,
            # Keep ORT's arena small so the fp32 VAE decode has GPU headroom.
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": 8 * 1024 * 1024 * 1024,
        }
        self.sess = ort.InferenceSession(
            onnx_path, so,
            providers=[("CUDAExecutionProvider", cuda_opts),
                       "CPUExecutionProvider"])
        self._itypes = {i.name: i.type for i in self.sess.get_inputs()}
        self.n_calls = 0
        self.total_s = 0.0

    @contextmanager
    def cache_context(self, name):
        yield

    def _np(self, name, t):
        a = t.detach().to("cpu")
        if "float16" in self._itypes[name]:
            return a.to(torch.float16).numpy()
        return a.to(torch.float32).numpy()

    def __call__(self, hidden_states, timestep, encoder_hidden_states,
                 attention_kwargs=None, return_dict=False, **kwargs):
        feeds = {
            "hidden_states": self._np("hidden_states", hidden_states),
            "timestep": self._np("timestep", timestep),
            "encoder_hidden_states": self._np("encoder_hidden_states",
                                              encoder_hidden_states),
        }
        t0 = time.time()
        out = self.sess.run(["noise_pred"], feeds)[0]
        self.total_s += time.time() - t0
        self.n_calls += 1
        noise_pred = torch.from_numpy(np.ascontiguousarray(out)).to(
            self._device, hidden_states.dtype)
        return (noise_pred,)


def onnx_vae_decode(pipe, latents, ort_vae):
    """Replicate WanPipeline's latent un-scaling, then decode with the ONNX VAE.

    Mirrors the ``output_type != "latent"`` branch of WanPipeline.__call__:
    un-normalize with the VAE's latents_mean/std, decode, then postprocess.
    """
    cfg = pipe.vae.config
    z_dim = cfg.z_dim
    latents = latents.to(torch.float32).cpu()
    mean = torch.tensor(cfg.latents_mean).view(1, z_dim, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(cfg.latents_std).view(1, z_dim, 1, 1, 1)
    latents = latents / inv_std + mean
    video = ort_vae.decode(latents)  # [1, 3, F, H, W] in [-1, 1]
    return pipe.video_processor.postprocess_video(video, output_type="np")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--out", default="output_onnx.mp4")
    ap.add_argument("--vae", choices=["onnx", "torch"], default="onnx",
                    help="VAE decode backend (onnx = portable ORT decoder).")
    args = ap.parse_args()

    device = "cuda"
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True,
        num_train_timesteps=1000, flow_shift=3.0)
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.to(device)

    ort_vae = None
    if args.vae == "onnx":
        # Decode via the ONNX VAE; the torch VAE weights are only needed for
        # config/un-scaling, so move them off the GPU to free VRAM.
        pipe.vae.to("cpu")
        ort_vae = OrtVaeDecoder(gpu_mem_limit_gb=8)
    else:
        # Tile the VAE decode so 81 fp32 frames fit alongside T5 + the ORT session.
        pipe.vae.enable_tiling()

    # Swap the torch transformer for the ONNX session; free the torch weights.
    cfg = pipe.transformer.config
    pipe.transformer = None
    gc.collect()
    torch.cuda.empty_cache()
    print("Creating ORT session (CUDA) ...")
    pipe.transformer = OrtTransformer(FP16_PATH, cfg, torch.float16, device)

    prompt = ("Two anthropomorphic cats in comfy boxing gear and bright gloves "
              "fight intensely on a spotlighted stage.")
    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, "
        "works, paintings, images, static, overall gray, worst quality, low "
        "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
        "limbs, fused fingers, still picture, messy background, three legs, many "
        "people in the background, walking backwards")

    # For the ONNX VAE path, stop the pipeline at the latent and decode with ORT.
    output_type = "latent" if args.vae == "onnx" else "np"

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=6.0,
        output_type=output_type,
    ).frames
    tr = pipe.transformer

    if args.vae == "onnx":
        # Free the denoiser ORT session before the VAE session decodes.
        pipe.transformer = None
        gc.collect()
        torch.cuda.empty_cache()
        output = onnx_vae_decode(pipe, result, ort_vae)
    else:
        output = result[0]
    dt = time.time() - t0

    export_to_video(output, args.out, fps=16)
    print(f"Saved {args.out} (vae={args.vae})")
    print(f"Total pipeline time: {dt:.1f}s | ORT denoiser calls: {tr.n_calls} | "
          f"avg ORT/call: {tr.total_s / max(tr.n_calls, 1):.3f}s")


if __name__ == "__main__":
    main()
