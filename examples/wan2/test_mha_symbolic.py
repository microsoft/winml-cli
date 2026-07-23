"""Validate torch.onnx.ops.symbolic -> com.microsoft.MultiHeadAttention (dynamo)."""
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

TINY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tiny_sym.onnx")


def sdpa_ref(q, k, v, num_heads):
    B, Sq, D = q.shape
    H = D // num_heads
    Sk = k.shape[1]
    qh = q.view(B, Sq, num_heads, H).transpose(1, 2)
    kh = k.view(B, Sk, num_heads, H).transpose(1, 2)
    vh = v.view(B, Sk, num_heads, H).transpose(1, 2)
    o = F.scaled_dot_product_attention(qh, kh, vh)
    return o.transpose(1, 2).reshape(B, Sq, D)


class Tiny(torch.nn.Module):
    def forward(self, q, k, v):
        B, Sq, D = q.shape
        return torch.onnx.ops.symbolic(
            "com.microsoft::MultiHeadAttention",
            (q, k, v),
            attrs={"num_heads": 2},
            dtype=q.dtype,
            shape=(B, Sq, D),
            version=1,
        )


def main():
    torch.manual_seed(0)
    B, Sq, Sk, N, H = 1, 8, 8, 2, 4
    D = N * H
    q = torch.randn(B, Sq, D)
    k = torch.randn(B, Sk, D)
    v = torch.randn(B, Sk, D)

    ref = sdpa_ref(q, k, v, N).detach().numpy()

    prog = torch.onnx.export(
        Tiny().eval(), (q, k, v),
        input_names=["q", "k", "v"], output_names=["o"],
        opset_version=18, dynamo=True,
    )
    prog.save(TINY_PATH)

    model = onnx.load(TINY_PATH)
    print("nodes:", [n.op_type + "(" + n.domain + ")" for n in model.graph.node])

    sess = ort.InferenceSession(
        TINY_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out = sess.run(["o"], {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()})[0]
    print("max_abs_diff vs torch SDPA:", np.abs(out - ref).max())


if __name__ == "__main__":
    main()
