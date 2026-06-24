# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Custom ONNX export ops that reshape HF's Qwen3 modules for export.

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

from typing import Any

import torch
import torch.nn as nn
from torch.onnx import symbolic_helper


# =============================================================================
# Custom ONNX symbolic functions
# =============================================================================


class LpNormOnnxExport(torch.autograd.Function):
    """RMSNorm body → ONNX ``LpNormalization`` (p=2 along last dim)."""

    @staticmethod
    def symbolic(g, input, axis, p) -> Any:
        """Emit the ONNX ``LpNormalization`` node during export."""
        output_type = input.type().with_sizes(symbolic_helper._get_tensor_sizes(input))
        output = g.op(
            "onnx::LpNormalization",
            input,
            axis_i=int(axis),
            p_i=int(p),
        )
        return output.setType(output_type)

    @staticmethod
    def forward(ctx, input, axis, p) -> Any:
        """Shape-only tracing placeholder; returns ``input`` unchanged.

        The real op is emitted by ``symbolic`` during ONNX export; ``forward``
        exists solely so the TorchScript exporter (and Optimum's pre-export dry
        run) can trace output shapes. It is NOT a correct eager RMSNorm — do
        not call this module for real inference.
        """
        return input


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
    ) -> Any:
        """Emit the fused ``com.microsoft::GroupQueryAttention`` node."""
        args = [
            query,
            key,
            value,
            past_key,
            past_value,
            seqlens_k,
            total_sequence_length,
            cos_cache,
            sin_cache,
        ]
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
        present_keys.setType(
            past_key.type().with_sizes(symbolic_helper._get_tensor_sizes(past_key))
        )
        present_values.setType(
            past_value.type().with_sizes(symbolic_helper._get_tensor_sizes(past_value))
        )
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
    ) -> Any:
        """Shape-only tracing placeholder; returns stand-in (output, KV).

        The real op is emitted by ``symbolic`` during ONNX export; ``forward``
        exists solely so the TorchScript exporter (and Optimum's pre-export dry
        run) can trace output shapes. It is NOT correct attention — do not call
        this module for real inference.
        """
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
        """Apply the 1x1 conv with NHWC<->NCHW permutes (+ optional bias)."""
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = torch.nn.functional.conv2d(x, self.weight)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        if self.bias is not None:
            x = x + self.bias
        return x

    @classmethod
    def from_linear_module(cls, linear: nn.Linear) -> TransposeConv2d1x1Transpose:
        """Build a 1x1-conv replacement from an existing ``nn.Linear``."""
        return cls(linear.in_features, linear.out_features, linear.weight, linear.bias)


__all__ = [
    "GroupQueryAttentionOnnxExport",
    "LpNormOnnxExport",
    "TransposeConv2d1x1Transpose",
]
