# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""winml-owned Qwen3 model definitions for the transformer-only ONNX export.

Each class is a plain ``nn.Module`` that carries the export-time behaviour
directly (``prepare_for_onnx_export`` + ``forward``). The export entry point
binds these ``forward`` methods onto the corresponding live Qwen3 submodules,
so the stock eager model is left untouched.

What each class emits:

- ``WinMLQwen3RMSNorm``   -> ``onnx::LpNormalization`` body.
- ``WinMLQwen3Attention`` -> ``com.microsoft::GroupQueryAttention`` (built-in
  rotary) with optional 1x1 ``Conv`` projections.
- ``WinMLQwen3MLP``       -> 1x1 ``Conv`` projections (NHWC).
- ``WinMLQwen3DecoderLayer`` / ``WinMLQwen3Model`` -> transformer-only forward
  that threads the KV cache + seq-len tensors and omits embeddings / lm_head.

``apply_transformer_only_export_prep`` (in ``qwen3_export_ops``) walks a loaded
``Qwen3ForCausalLM``, calls ``prepare_for_onnx_export`` on each submodule, and
binds the matching ``forward`` from these classes onto it.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .qwen3_export_ops import (
    GroupQueryAttentionOnnxExport,
    LpNormOnnxExport,
    TransposeConv2d1x1Transpose,
)


class WinMLQwen3RMSNorm(nn.Module):
    """RMSNorm export variant — ``onnx::LpNormalization`` body."""

    def prepare_for_onnx_export(self) -> None:
        # Pre-multiply the gain into the weight (LpNorm has unit gain).
        n = self.weight.numel()
        scale = torch.sqrt(
            torch.tensor([n], device=self.weight.device, dtype=self.weight.dtype)
        )
        if torch.any(self.weight.data != torch.ones_like(self.weight)).item():
            new_w = scale * self.weight
        else:
            new_w = scale
        self.weight = nn.Parameter(new_w)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = LpNormOnnxExport.apply(hidden_states, -1, 2)
        return self.weight * out


class WinMLQwen3MLP(nn.Module):
    """MLP export variant — 1x1 Conv projections (forward unchanged)."""

    def prepare_for_onnx_export(self, *, matmul_to_conv: bool) -> None:
        if not matmul_to_conv:
            return
        self.gate_proj = TransposeConv2d1x1Transpose.from_linear_module(self.gate_proj)
        self.up_proj = TransposeConv2d1x1Transpose.from_linear_module(self.up_proj)
        self.down_proj = TransposeConv2d1x1Transpose.from_linear_module(self.down_proj)


class WinMLQwen3Attention(nn.Module):
    """Attention export variant — fused ``GroupQueryAttention`` op."""

    def prepare_for_onnx_export(self, *, matmul_to_conv: bool) -> None:
        if matmul_to_conv:
            self.q_proj = TransposeConv2d1x1Transpose.from_linear_module(self.q_proj)
            self.k_proj = TransposeConv2d1x1Transpose.from_linear_module(self.k_proj)
            self.v_proj = TransposeConv2d1x1Transpose.from_linear_module(self.v_proj)
            self.o_proj = TransposeConv2d1x1Transpose.from_linear_module(self.o_proj)
        self._matmul_to_conv = matmul_to_conv  # noqa: SLF001

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_seq_len: torch.Tensor | None = None,
        total_seq_len: torch.Tensor | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[torch.Tensor, None, tuple[torch.Tensor, torch.Tensor]]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        input_shape = hidden_states.shape[1:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(key_states.view(hidden_shape))

        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        query_dim = num_heads * self.head_dim
        key_dim = num_kv_heads * self.head_dim
        query_states = query_states.reshape(1, -1, query_dim)
        key_states = key_states.reshape(1, -1, key_dim)

        if self._matmul_to_conv:
            value_states = value_states.squeeze(0)

        past_keys, past_values = past_key_value

        # GroupQueryAttention requires Q/K/V/past_K/past_V to share dtype.
        # The KV cache is FP16, so cast Q/K/V to the same dtype; otherwise ORT
        # type inference rejects the node.
        kv_dtype = past_keys.dtype
        if query_states.dtype != kv_dtype:
            query_states = query_states.to(kv_dtype)
            key_states = key_states.to(kv_dtype)
            value_states = value_states.to(kv_dtype)

        cos, sin = self.rotary_emb(
            value_states,
            torch.arange(self.config.max_position_embeddings).unsqueeze(0),
        )
        cos = cos.squeeze(0)[:, : cos.shape[-1] // 2]
        sin = sin.squeeze(0)[:, : sin.shape[-1] // 2]
        if cos.dtype != kv_dtype:
            cos = cos.to(kv_dtype)
            sin = sin.to(kv_dtype)

        if isinstance(past_seq_len, int):
            past_seq_len = torch.tensor(past_seq_len)
        past_seq_len = torch.atleast_2d(past_seq_len)

        attention_output, present_keys, present_values = GroupQueryAttentionOnnxExport.apply(
            query_states,
            key_states,
            value_states,
            past_keys,
            past_values,
            past_seq_len,
            total_seq_len,
            cos,
            sin,
            1,  # do_rotary
            num_kv_heads,
            num_heads,
        )

        # Cast back to the residual-stream dtype so the downstream Conv
        # (o_proj) sees its expected weight dtype.
        if attention_output.dtype != hidden_states.dtype:
            attention_output = attention_output.to(hidden_states.dtype)

        if self._matmul_to_conv:
            attention_output = attention_output.unsqueeze(0)

        attention_output = self.o_proj(attention_output)
        return attention_output, None, (present_keys, present_values)


class WinMLQwen3DecoderLayer(nn.Module):
    """Decoder-layer export variant — threads KV cache + seq-len kwargs."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_seq_len: torch.Tensor | None = None,
        total_seq_len: torch.Tensor | None = None,
        use_cache: bool = True,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _, present_kv = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            past_seq_len=past_seq_len,
            total_seq_len=total_seq_len,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_kv,)
        return outputs


class WinMLQwen3Model(nn.Module):
    """Model export variant — transformer-only body (no embeddings / lm_head)."""

    def prepare_for_onnx_export(self, *, matmul_to_conv: bool) -> None:
        self._matmul_to_conv = matmul_to_conv  # noqa: SLF001

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        past_seq_len: torch.Tensor,
        total_seq_len: torch.Tensor,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        hidden_states = inputs_embeds
        if self._matmul_to_conv:
            hidden_states = hidden_states.unsqueeze(0)  # NHWC for Conv path

        present_kvs: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
        for idx, layer in enumerate(self.layers):
            out = layer(
                hidden_states,
                past_key_value=past_key_values[idx],
                past_seq_len=past_seq_len,
                total_seq_len=total_seq_len,
                use_cache=use_cache,
            )
            hidden_states = out[0]
            if use_cache:
                present_kvs += (out[1],)

        hidden_states = self.norm(hidden_states)
        if self._matmul_to_conv:
            hidden_states = hidden_states.squeeze(0)
        return hidden_states, present_kvs


__all__ = [
    "WinMLQwen3Attention",
    "WinMLQwen3DecoderLayer",
    "WinMLQwen3MLP",
    "WinMLQwen3Model",
    "WinMLQwen3RMSNorm",
]
