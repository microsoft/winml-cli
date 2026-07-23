"""Verify the fp16 ONNX denoiser matches the PyTorch transformer in ORT-CUDA."""
import os

import numpy as np
import onnxruntime as ort
import torch
from diffusers import WanTransformer3DModel

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FP16_PATH = os.path.join(os.path.dirname(__file__), "model", "wan_transformer_fp16.onnx")

LAT_FRAMES = (81 - 1) // 4 + 1
LAT_H, LAT_W = 480 // 8, 832 // 8


def main():
    torch.manual_seed(0)
    device = "cuda"

    model = WanTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.float16).eval().to(device)
    text_dim = model.config.text_dim

    hidden = torch.randn(1, model.config.in_channels, LAT_FRAMES, LAT_H, LAT_W,
                         dtype=torch.float16, device=device)
    timestep = torch.tensor([987.0], dtype=torch.float32, device=device)
    enc = torch.randn(1, 512, text_dim, dtype=torch.float16, device=device)

    with torch.no_grad():
        ref = model(hidden_states=hidden, timestep=timestep,
                    encoder_hidden_states=enc, return_dict=False)[0]
    ref = ref.float().cpu().numpy()

    sess = ort.InferenceSession(
        FP16_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("ORT providers:", sess.get_providers())
    itypes = {i.name: i.type for i in sess.get_inputs()}
    print("ONNX input types:", itypes)

    def cast(name, t):
        return t.astype(np.float16) if "float16" in itypes[name] else t.astype(np.float32)

    feeds = {
        "hidden_states": cast("hidden_states", hidden.float().cpu().numpy()),
        "timestep": cast("timestep", timestep.float().cpu().numpy()),
        "encoder_hidden_states": cast("encoder_hidden_states", enc.float().cpu().numpy()),
    }
    out = sess.run(["noise_pred"], feeds)[0].astype(np.float32)

    diff = np.abs(out - ref)
    denom = np.abs(ref).mean()
    print(f"shape={out.shape} dtype_onnx_out={out.dtype}")
    print(f"max_abs_diff={diff.max():.4f}  mean_abs_diff={diff.mean():.5f}  "
          f"mean_abs_ref={denom:.4f}  rel={diff.mean()/denom:.4%}")


if __name__ == "__main__":
    main()
