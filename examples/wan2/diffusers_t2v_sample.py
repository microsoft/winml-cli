"""Minimal Diffusers text-to-video sample for Wan-AI/Wan2.1-T2V-1.3B-Diffusers.

Adapted from the repo README. Uses 480P settings recommended for the 1.3B model.
"""
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# 1.3B is trained at 480P; flow_shift=3.0 recommended for 480P.
flow_shift = 3.0

vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32)
scheduler = UniPCMultistepScheduler(
    prediction_type="flow_prediction",
    use_flow_sigmas=True,
    num_train_timesteps=1000,
    flow_shift=flow_shift,
)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = scheduler
pipe.to("cuda")

prompt = ("Two anthropomorphic cats in comfy boxing gear and bright gloves "
          "fight intensely on a spotlighted stage.")
negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
    "limbs, fused fingers, still picture, messy background, three legs, many "
    "people in the background, walking backwards")

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=6.0,
).frames[0]

export_to_video(output, "output.mp4", fps=16)
print("Saved output.mp4")
