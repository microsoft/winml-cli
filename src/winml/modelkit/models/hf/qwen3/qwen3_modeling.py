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

``apply_transformer_only_export_prep`` (defined below) walks a loaded
``Qwen3ForCausalLM``, calls ``prepare_for_onnx_export`` on each submodule, and
binds the matching ``forward`` from these classes onto it.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from .qwen3_export_ops import (
    GroupQueryAttentionOnnxExport,
    LpNormOnnxExport,
    TransposeConv2d1x1Transpose,
)


class WinMLQwen3RMSNorm(nn.Module):
    """RMSNorm export variant — ``onnx::LpNormalization`` body."""

    # Bound at runtime onto a live ``Qwen3RMSNorm`` module; declared so the
    # type checker knows the attribute these methods rely on.
    weight: torch.Tensor

    def prepare_for_onnx_export(self) -> None:
        """Fold the RMSNorm gain into the weight (LpNorm has unit gain)."""
        # Pre-multiply the gain into the weight (LpNorm has unit gain).
        # ``scale`` is shape ``[1]`` and broadcasts over ``self.weight``
        # (shape ``[hidden_size]``), so the result keeps the per-channel
        # shape even when the original weights are all ones.
        n = self.weight.numel()
        scale = torch.sqrt(torch.tensor([n], device=self.weight.device, dtype=self.weight.dtype))
        self.weight = nn.Parameter(scale * self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the LpNormalization-based RMSNorm body."""
        out = cast("torch.Tensor", LpNormOnnxExport.apply(hidden_states, -1, 2))
        return self.weight * out


class WinMLQwen3MLP(nn.Module):
    """MLP export variant — 1x1 Conv projections (forward unchanged)."""

    # Bound at runtime onto a live ``Qwen3MLP`` module; declared so the type
    # checker has a non-circular type for the projections these methods swap.
    gate_proj: nn.Module
    up_proj: nn.Module
    down_proj: nn.Module

    def prepare_for_onnx_export(self, *, matmul_to_conv: bool) -> None:
        """Optionally swap the MLP's linear projections for 1x1 convs."""
        if not matmul_to_conv:
            return
        self.gate_proj = TransposeConv2d1x1Transpose.from_linear_module(
            cast("nn.Linear", self.gate_proj)
        )
        self.up_proj = TransposeConv2d1x1Transpose.from_linear_module(
            cast("nn.Linear", self.up_proj)
        )
        self.down_proj = TransposeConv2d1x1Transpose.from_linear_module(
            cast("nn.Linear", self.down_proj)
        )


class WinMLQwen3Attention(nn.Module):
    """Attention export variant — fused ``GroupQueryAttention`` op."""

    # Bound at runtime onto a live ``Qwen3Attention`` module; declared so the
    # type checker knows the attributes these methods rely on.
    config: Any
    head_dim: int
    q_proj: nn.Module
    k_proj: nn.Module
    v_proj: nn.Module
    o_proj: nn.Module

    def prepare_for_onnx_export(self, *, matmul_to_conv: bool) -> None:
        """Optionally swap the Q/K/V/O projections for 1x1 convs."""
        if matmul_to_conv:
            self.q_proj = TransposeConv2d1x1Transpose.from_linear_module(
                cast("nn.Linear", self.q_proj)
            )
            self.k_proj = TransposeConv2d1x1Transpose.from_linear_module(
                cast("nn.Linear", self.k_proj)
            )
            self.v_proj = TransposeConv2d1x1Transpose.from_linear_module(
                cast("nn.Linear", self.v_proj)
            )
            self.o_proj = TransposeConv2d1x1Transpose.from_linear_module(
                cast("nn.Linear", self.o_proj)
            )
        self._matmul_to_conv = matmul_to_conv

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_seq_len: torch.Tensor | int | None = None,
        total_seq_len: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, tuple[torch.Tensor, torch.Tensor]]:
        """Run fused GQA attention and return (output, None, present_kv)."""
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        input_shape = hidden_states.shape[1:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = cast("nn.Module", self.q_norm)(query_states.view(hidden_shape))
        key_states = cast("nn.Module", self.k_norm)(key_states.view(hidden_shape))

        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        query_dim = num_heads * self.head_dim
        key_dim = num_kv_heads * self.head_dim
        query_states = query_states.reshape(1, -1, query_dim)
        key_states = key_states.reshape(1, -1, key_dim)

        if self._matmul_to_conv:
            value_states = value_states.squeeze(0)

        assert past_key_value is not None
        past_keys, past_values = past_key_value

        # GroupQueryAttention requires Q/K/V/past_K/past_V to share dtype.
        # The KV cache is FP16, so cast Q/K/V to the same dtype; otherwise ORT
        # type inference rejects the node.
        kv_dtype = past_keys.dtype
        if query_states.dtype != kv_dtype:
            query_states = query_states.to(kv_dtype)
            key_states = key_states.to(kv_dtype)
            value_states = value_states.to(kv_dtype)

        cos, sin = cast("nn.Module", self.rotary_emb)(
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
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        """Run the decoder layer (attention + MLP) with residual adds."""
        residual = hidden_states
        hidden_states = cast("nn.Module", self.input_layernorm)(hidden_states)
        attn_out, _, present_kv = cast("nn.Module", self.self_attn)(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            past_seq_len=past_seq_len,
            total_seq_len=total_seq_len,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = cast("nn.Module", self.post_attention_layernorm)(hidden_states)
        hidden_states = cast("nn.Module", self.mlp)(hidden_states)
        hidden_states = residual + hidden_states

        outputs: tuple[Any, ...] = (hidden_states,)
        if use_cache:
            outputs += (present_kv,)
        return outputs


class WinMLQwen3Model(nn.Module):
    """Model export variant — transformer-only body (no embeddings / lm_head)."""

    def prepare_for_onnx_export(self, *, matmul_to_conv: bool) -> None:
        """Record whether projections use the 1x1-conv (NHWC) path."""
        self._matmul_to_conv = matmul_to_conv

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        past_seq_len: torch.Tensor,
        total_seq_len: torch.Tensor,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Run the transformer-only body, returning hidden states + KV."""
        hidden_states = inputs_embeds
        if self._matmul_to_conv:
            hidden_states = hidden_states.unsqueeze(0)  # NHWC for Conv path

        present_kvs: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
        for idx, layer in enumerate(cast("nn.ModuleList", self.layers)):
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

        hidden_states = cast("nn.Module", self.norm)(hidden_states)
        if self._matmul_to_conv:
            hidden_states = hidden_states.squeeze(0)
        return hidden_states, present_kvs


# =============================================================================
# Apply export prep: bind winml Qwen3 export methods onto a loaded model
# =============================================================================


def apply_transformer_only_export_prep(
    causal_lm: nn.Module, *, matmul_to_conv: bool = True
) -> None:
    """Mutate ``Qwen3ForCausalLM`` in-place into the export topology.

    Binds the winml-owned export behaviour (the ``WinMLQwen3*`` classes in this
    module) onto each Qwen3 submodule (runs ``prepare_for_onnx_export`` and
    rebinds ``forward``). After this call, ``causal_lm.model(inputs_embeds,
    past_key_values, past_seq_len, total_seq_len)`` runs the transformer-only
    forward.

    Args:
        causal_lm: A ``transformers.Qwen3ForCausalLM`` instance.
        matmul_to_conv: Swap ``nn.Linear`` projections to 1x1 ``Conv2d`` so
            QNN sees them as Conv.

    Raises:
        RuntimeError: If any expected Qwen3 submodule class is not found,
            meaning the loaded model does not match the expected topology
            (e.g. the stock HF class names changed).
    """

    def _bind(module: nn.Module, owner: type[nn.Module]) -> None:
        module.forward = owner.forward.__get__(module, type(module))

    # Identify Qwen3 submodules by their (stock HF) class name so we don't
    # depend on importing ``transformers.models.qwen3`` here.
    def _is(module: nn.Module, name: str) -> bool:
        return type(module).__name__ == name

    patched = {
        "Qwen3RMSNorm": 0,
        "Qwen3Attention": 0,
        "Qwen3MLP": 0,
        "Qwen3DecoderLayer": 0,
        "Qwen3Model": 0,
    }

    # Patch every RMSNorm first (Qwen3RMSNorm appears at top, in q_norm/k_norm,
    # in input/post_attention layernorms).
    for mod in causal_lm.modules():
        if _is(mod, "Qwen3RMSNorm"):
            WinMLQwen3RMSNorm.prepare_for_onnx_export(cast("WinMLQwen3RMSNorm", mod))
            _bind(mod, WinMLQwen3RMSNorm)
            patched["Qwen3RMSNorm"] += 1

    for mod in causal_lm.modules():
        if _is(mod, "Qwen3Attention"):
            WinMLQwen3Attention.prepare_for_onnx_export(
                cast("WinMLQwen3Attention", mod), matmul_to_conv=matmul_to_conv
            )
            _bind(mod, WinMLQwen3Attention)
            patched["Qwen3Attention"] += 1
        elif _is(mod, "Qwen3MLP"):
            # MLP forward is unchanged; only the projections are swapped to Conv.
            WinMLQwen3MLP.prepare_for_onnx_export(
                cast("WinMLQwen3MLP", mod), matmul_to_conv=matmul_to_conv
            )
            patched["Qwen3MLP"] += 1

    # HF moved ``rotary_emb`` from ``Qwen3Attention`` up to ``Qwen3Model``;
    # the export forward invokes ``self.rotary_emb`` on the attention module,
    # so re-attach a reference from the parent model.
    for mod in causal_lm.modules():
        if _is(mod, "Qwen3Model") and hasattr(mod, "rotary_emb"):
            for layer in cast("nn.ModuleList", mod.layers):
                cast("nn.Module", layer.self_attn).rotary_emb = mod.rotary_emb

    for mod in causal_lm.modules():
        if _is(mod, "Qwen3DecoderLayer"):
            _bind(mod, WinMLQwen3DecoderLayer)
            patched["Qwen3DecoderLayer"] += 1

    for mod in causal_lm.modules():
        if _is(mod, "Qwen3Model"):
            WinMLQwen3Model.prepare_for_onnx_export(
                cast("WinMLQwen3Model", mod), matmul_to_conv=matmul_to_conv
            )
            _bind(mod, WinMLQwen3Model)
            patched["Qwen3Model"] += 1

    missing = [name for name, count in patched.items() if count == 0]
    if missing:
        raise RuntimeError(
            "transformer-only export prep found no "
            f"{missing} submodule(s) to patch; the loaded model does not match "
            "the expected Qwen3 topology (stock HF class names may have changed)."
        )


__all__ = [
    "WinMLQwen3Attention",
    "WinMLQwen3DecoderLayer",
    "WinMLQwen3MLP",
    "WinMLQwen3Model",
    "WinMLQwen3RMSNorm",
    "apply_transformer_only_export_prep",
]
