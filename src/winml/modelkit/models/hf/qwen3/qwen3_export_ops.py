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
    def symbolic(g: Any, input: Any, axis: Any, p: Any) -> Any:
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
    def forward(ctx: Any, input: Any, axis: Any, p: Any) -> Any:
        """Real ``LpNormalization`` (``input / ||input||_p`` along ``axis``).

        The exported node comes from ``symbolic``; this eager body computes the
        same value so any eager execution (unit tests, calibration debug runs,
        the exporter's own shape-tracing pass) gets correctly normalized output
        instead of a silent identity. It matches the ONNX op faithfully (no
        RMSNorm epsilon), since that is exactly what ``symbolic`` emits.
        """
        return input / torch.linalg.vector_norm(input, ord=p, dim=axis, keepdim=True)


class GroupQueryAttentionOnnxExport(torch.autograd.Function):
    """Fused Q/K/V + KV-cache + rotary → ``com.microsoft::GroupQueryAttention``."""

    @staticmethod
    def symbolic(
        g: Any,
        query: Any,
        key: Any,
        value: Any,
        past_key: Any,
        past_value: Any,
        seqlens_k: Any,
        total_sequence_length: Any,
        cos_cache: Any,
        sin_cache: Any,
        do_rotary: Any,
        kv_num_heads: Any,
        num_heads: Any,
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
        ctx: Any,
        query: Any,
        key: Any,
        value: Any,
        past_key: Any,
        past_value: Any,
        seqlens_k: Any,
        total_sequence_length: Any,
        cos_cache: Any,
        sin_cache: Any,
        do_rotary: Any,
        kv_num_heads: Any,
        num_heads: Any,
    ) -> Any:
        """Shape-only tracing placeholder; returns a stand-in ``(output, KV)``.

        The real op is emitted by ``symbolic`` during ONNX export; this body
        only needs to return tensors of the right shape/dtype. It deliberately
        does NOT raise on eager execution, even though that yields a stale
        (never-advanced) KV cache: the HTP export pipeline runs a real eager
        ``forward`` pass to capture the module hierarchy (see
        ``export/htp/hierarchy.py::trace_model_execution``), and that pass is
        indistinguishable from misuse — ``torch.jit.is_tracing()`` and
        ``torch.onnx.is_in_onnx_export()`` are both False there — so raising
        would break the actual build. There is also no cheap faithful eager
        equivalent (correct attention would grow the sequence axis that the
        static-shape export freezes). This module is export-only by design and
        is never run for real inference; calibration loads a fresh real model.
        """
        return query, past_key, past_value  # placeholder shapes (export-only)


# =============================================================================
# 1x1 Conv replacement for nn.Linear
# =============================================================================


class TransposeConv2d1x1Transpose(nn.Module):
    """``nn.Linear`` → 1x1 ``Conv2d`` with NHWC<->NCHW permutes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
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
