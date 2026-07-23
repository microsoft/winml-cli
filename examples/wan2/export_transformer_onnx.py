"""Export the Wan2.1-T2V-1.3B denoiser (WanTransformer3DModel) to fp16 ONNX.

Strategy:
  1. Replace each self/cross-attention with a single fused
     `com.microsoft.MultiHeadAttention` contrib node, emitted via
     torch.onnx.ops.symbolic. The naive softmax(QK^T)V export materializes a
     ~26GB score tensor per layer over the 32,760-token sequence and is unusable
     in ORT; the fused node lets ORT use a flash-attention kernel instead.
  2. Export the transformer in uniform fp32 with the dynamo (torch.export)
     exporter. Uniform dtype avoids the mixed fp16/fp32 adds around
     scale_shift_table that break the type-promotion pass, and the fake-tensor
     tracing avoids the multi-GB activation blow-up that OOMs the legacy tracer.
  3. Convert the fp32 ONNX graph to fp16 with onnxconverter-common.

Only the denoising transformer is exported -- it is the module that runs on
every diffusion step. The T5 text encoder and the VAE stay in PyTorch.

Static shapes for 480P / 81 frames, batch=1 (the diffusers WanPipeline calls the
transformer once for cond and once for uncond, each with batch=1).
"""
import os

import onnx
import torch
from diffusers import WanTransformer3DModel
from diffusers.models.transformers import transformer_wan as twan
from onnxconverter_common import float16

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OUT_DIR = os.path.join(os.path.dirname(__file__), "model")
FP32_PATH = os.path.join(OUT_DIR, "wan_transformer_fp32.onnx")
FP16_PATH = os.path.join(OUT_DIR, "wan_transformer_fp16.onnx")
OPSET = 18

# 480P / 81 frames latent geometry: vae temporal stride 4, spatial stride 8.
LAT_FRAMES = (81 - 1) // 4 + 1   # 21
LAT_H = 480 // 8                 # 60
LAT_W = 832 // 8                 # 104


# ---------------------------------------------------------------------------
# Fused attention -> com.microsoft.MultiHeadAttention
# ---------------------------------------------------------------------------


def _apply_rotary_emb_onnx(hs, freqs_cos, freqs_sin):
    """ScatterND-free rotary embedding (interleaved layout).

    Equivalent to the diffusers reference which uses out[..., 0::2]/out[..., 1::2]
    slice-assignment (exports to hundreds of ScatterND ops).
    """
    xshaped = hs.unflatten(-1, (-1, 2))
    x1 = xshaped[..., 0]
    x2 = xshaped[..., 1]
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return out.type_as(hs)


def wan_mha_call(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
    """Drop-in WanAttnProcessor.__call__ emitting the fused MHA op (T2V only)."""
    query, key, value = twan._get_qkv_projections(
        attn, hidden_states, encoder_hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    if rotary_emb is not None:
        query = _apply_rotary_emb_onnx(query, *rotary_emb)
        key = _apply_rotary_emb_onnx(key, *rotary_emb)

    B, Sq, N, H = query.shape
    Sk = key.shape[1]
    D = N * H
    query = query.reshape(B, Sq, D)
    key = key.reshape(B, Sk, D)
    value = value.reshape(B, Sk, D)

    out = torch.onnx.ops.symbolic(
        "com.microsoft::MultiHeadAttention",
        (query, key, value),
        attrs={"num_heads": int(attn.heads)},
        dtype=query.dtype,
        shape=(B, Sq, D),
        version=1,
    )
    out = out.type_as(query)
    out = attn.to_out[0](out)
    out = attn.to_out[1](out)
    return out


class TransformerWrapper(torch.nn.Module):
    """Fixed-signature wrapper that returns just the predicted-noise tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return self.model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class OnnxRMSNorm(torch.nn.Module):
    """Explicit RMSNorm equivalent to torch.nn.RMSNorm.

    torch.nn.RMSNorm dispatches to aten._fused_rms_norm, which the dynamo ONNX
    exporter cannot lower. This computes the same result with basic ops.
    """

    def __init__(self, weight, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().clone())
        self.eps = 1e-6 if eps is None else float(eps)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


def replace_rmsnorm(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.RMSNorm):
            setattr(module, name, OnnxRMSNorm(child.weight, child.eps))
        else:
            replace_rmsnorm(child)


def export_fp32():
    device = "cuda"
    print("Loading transformer (fp32) ...")
    model = WanTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.float32)
    model.eval().to(device)

    cfg = model.config
    text_dim = cfg.text_dim
    in_ch = cfg.in_channels
    print(f"config: in_channels={in_ch} text_dim={text_dim} "
          f"num_layers={cfg.num_layers}")

    # Rotary buffers are float64 by default -> force fp32 (no float64 in graph).
    model.rope.freqs_cos = model.rope.freqs_cos.to(torch.float32)
    model.rope.freqs_sin = model.rope.freqs_sin.to(torch.float32)

    # Swap torch.nn.RMSNorm (fused op) for an ONNX-exportable implementation.
    replace_rmsnorm(model)
    model.to(device)

    # Emit fused MultiHeadAttention contrib nodes instead of naive softmax.
    twan.WanAttnProcessor.__call__ = wan_mha_call

    wrapper = TransformerWrapper(model).eval()

    hidden_states = torch.randn(
        1, in_ch, LAT_FRAMES, LAT_H, LAT_W, dtype=torch.float32, device=device)
    timestep = torch.tensor([999.0], dtype=torch.float32, device=device)
    encoder_hidden_states = torch.randn(
        1, 512, text_dim, dtype=torch.float32, device=device)

    # Note: with the fused MHA op, eager output values are placeholders; only
    # the traced graph is meaningful, so we skip a numeric ref forward here.
    print("Exporting to ONNX fp32 (dynamo / torch.export) ...")
    onnx_program = torch.onnx.export(
        wrapper,
        (hidden_states, timestep, encoder_hidden_states),
        input_names=["hidden_states", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"],
        opset_version=OPSET,
        dynamo=True,
    )
    onnx_program.save(FP32_PATH, external_data=True)
    print(f"Saved fp32 ONNX to {FP32_PATH}")


def convert_fp16():
    print("Converting fp32 ONNX -> fp16 ...")
    model = onnx.load(FP32_PATH)
    model16 = float16.convert_float_to_float16(
        model, keep_io_types=False, disable_shape_infer=True)
    onnx.save(
        model16,
        FP16_PATH,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(FP16_PATH) + ".data",
    )
    print(f"Saved fp16 ONNX to {FP16_PATH}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    export_fp32()
    convert_fp16()


if __name__ == "__main__":
    main()
