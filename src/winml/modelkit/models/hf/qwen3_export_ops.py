# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Custom ONNX export ops + the entry point that reshapes HF's Qwen3 modules
for the transformer-only export.

These reshape the standard HF Qwen3 modules so winml-cli can produce a
QNN-friendly, transformer-only graph:

- ``LpNormalization`` replaces the eager RMSNorm Mul/Pow/ReduceMean chain.
- ``com.microsoft::GroupQueryAttention`` replaces the eager QKV MatMul +
  Softmax + KV-update path (with built-in rotary).
- 1x1 ``Conv`` (NHWC<->NCHW) replaces ``nn.Linear`` for QNN-friendly
  projections.

Everything here operates only on the standard ``transformers.models.qwen3``
module attributes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.onnx import symbolic_helper


# =============================================================================
# Custom ONNX symbolic functions
# =============================================================================


class LpNormOnnxExport(torch.autograd.Function):
    """RMSNorm body → ONNX ``LpNormalization`` (p=2 along last dim)."""

    @staticmethod
    def symbolic(g, input, axis, p):  # noqa: D401
        output_type = input.type().with_sizes(symbolic_helper._get_tensor_sizes(input))
        output = g.op(
            "onnx::LpNormalization",
            input,
            axis_i=int(axis),
            p_i=int(p),
        )
        return output.setType(output_type)

    @staticmethod
    def forward(ctx, input, axis, p):  # noqa: ARG004
        return input  # placeholder — real compute happens in symbolic


class GroupQueryAttentionOnnxExport(torch.autograd.Function):
    """Fused Q/K/V + KV-cache + rotary → ``com.microsoft::GroupQueryAttention``."""

    @staticmethod
    def symbolic(
        g,
        query,
        key,
        value,
        past_key,
        past_value,
        seqlens_k,
        total_sequence_length,
        cos_cache,
        sin_cache,
        do_rotary,
        kv_num_heads,
        num_heads,
    ):
        args = [query, key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache, sin_cache]
        attention_output, present_keys, present_values = g.op(
            "com.microsoft::GroupQueryAttention",
            *args,
            do_rotary_i=int(do_rotary),
            kv_num_heads_i=int(kv_num_heads),
            num_heads_i=int(num_heads),
            outputs=3,
        )

        query_sizes = symbolic_helper._get_tensor_sizes(query)
        attention_output.setType(query.type().with_sizes(query_sizes))
        present_keys.setType(past_key.type().with_sizes(symbolic_helper._get_tensor_sizes(past_key)))
        present_values.setType(past_value.type().with_sizes(symbolic_helper._get_tensor_sizes(past_value)))
        return attention_output, present_keys, present_values

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        past_key,
        past_value,
        seqlens_k,
        total_sequence_length,
        cos_cache,
        sin_cache,
        do_rotary,
        kv_num_heads,
        num_heads,
    ):  # noqa: ARG004
        return query, past_key, past_value  # placeholder shapes


# =============================================================================
# 1x1 Conv replacement for nn.Linear
# =============================================================================


class TransposeConv2d1x1Transpose(nn.Module):
    """``nn.Linear`` → 1x1 ``Conv2d`` with NHWC<->NCHW permutes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        weight: torch.nn.Parameter,
        bias: torch.nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        # Linear weight is (out, in); Conv2d weight is (out, in, 1, 1).
        self.weight = nn.Parameter(weight.data.view(out_channels, in_channels, 1, 1))
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = torch.nn.functional.conv2d(x, self.weight)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        if self.bias is not None:
            x = x + self.bias
        return x

    @classmethod
    def from_linear_module(cls, linear: nn.Linear) -> TransposeConv2d1x1Transpose:
        return cls(linear.in_features, linear.out_features, linear.weight, linear.bias)


# =============================================================================
# Apply export prep: bind winml Qwen3 export methods onto a loaded model
# =============================================================================


def apply_transformer_only_export_prep(causal_lm: nn.Module, *, matmul_to_conv: bool = True) -> None:
    """Mutate ``Qwen3ForCausalLM`` in-place into the export topology.

    Binds the winml-owned export behaviour from :mod:`.qwen3_modeling` onto each
    Qwen3 submodule (runs ``prepare_for_onnx_export`` and rebinds ``forward``).
    After this call, ``causal_lm.model(inputs_embeds, past_key_values,
    past_seq_len, total_seq_len)`` runs the transformer-only forward.

    Args:
        causal_lm: A ``transformers.Qwen3ForCausalLM`` instance.
        matmul_to_conv: Swap ``nn.Linear`` projections to 1x1 ``Conv2d`` so
            QNN sees them as Conv.
    """
    from .qwen3_modeling import (
        WinMLQwen3Attention,
        WinMLQwen3DecoderLayer,
        WinMLQwen3MLP,
        WinMLQwen3Model,
        WinMLQwen3RMSNorm,
    )

    def _bind(module: nn.Module, owner: type) -> None:
        module.forward = owner.forward.__get__(module, type(module))

    # Identify Qwen3 submodules by their (stock HF) class name so we don't
    # depend on importing ``transformers.models.qwen3`` here.
    def _is(module: nn.Module, name: str) -> bool:
        return type(module).__name__ == name

    # Patch every RMSNorm first (Qwen3RMSNorm appears at top, in q_norm/k_norm,
    # in input/post_attention layernorms).
    for mod in causal_lm.modules():
        if _is(mod, "Qwen3RMSNorm"):
            WinMLQwen3RMSNorm.prepare_for_onnx_export(mod)
            _bind(mod, WinMLQwen3RMSNorm)

    for mod in causal_lm.modules():
        if _is(mod, "Qwen3Attention"):
            WinMLQwen3Attention.prepare_for_onnx_export(mod, matmul_to_conv=matmul_to_conv)
            _bind(mod, WinMLQwen3Attention)
        elif _is(mod, "Qwen3MLP"):
            # MLP forward is unchanged; only the projections are swapped to Conv.
            WinMLQwen3MLP.prepare_for_onnx_export(mod, matmul_to_conv=matmul_to_conv)

    # HF moved ``rotary_emb`` from ``Qwen3Attention`` up to ``Qwen3Model``;
    # the export forward invokes ``self.rotary_emb`` on the attention module,
    # so re-attach a reference from the parent model.
    for mod in causal_lm.modules():
        if _is(mod, "Qwen3Model") and hasattr(mod, "rotary_emb"):
            for layer in mod.layers:
                layer.self_attn.rotary_emb = mod.rotary_emb

    for mod in causal_lm.modules():
        if _is(mod, "Qwen3DecoderLayer"):
            _bind(mod, WinMLQwen3DecoderLayer)

    for mod in causal_lm.modules():
        if _is(mod, "Qwen3Model"):
            WinMLQwen3Model.prepare_for_onnx_export(mod, matmul_to_conv=matmul_to_conv)
            _bind(mod, WinMLQwen3Model)


__all__ = [
    "GroupQueryAttentionOnnxExport",
    "LpNormOnnxExport",
    "TransposeConv2d1x1Transpose",
    "apply_transformer_only_export_prep",
]
