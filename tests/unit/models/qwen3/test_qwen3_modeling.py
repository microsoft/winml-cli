# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for WinMLQwen3Attention rope cache — static _max_rope_len approach.

The rope cache length is now pinned to a plain Python int attribute
``_max_rope_len`` set by ``prepare_for_onnx_export``, so the ONNX exporter
bakes a static-shape constant rather than a symbolic range derived from
``total_seq_len.item()``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from winml.modelkit.models.hf.qwen3.qwen3_modeling import WinMLQwen3Attention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_attention_module(
    max_position_embeddings: int = 40960,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 64,
) -> MagicMock:
    """Build a minimal mock bound to WinMLQwen3Attention methods."""
    hidden_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    mod = MagicMock()
    mod.head_dim = head_dim
    mod._matmul_to_conv = False
    mod.config = SimpleNamespace(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=max_position_embeddings,
    )

    # Identity projections: return float32 tensors of the right shape
    mod.q_proj.side_effect = lambda x: torch.zeros(1, x.shape[1], hidden_size)
    mod.k_proj.side_effect = lambda x: torch.zeros(1, x.shape[1], kv_size)
    mod.v_proj.side_effect = lambda x: torch.zeros(1, x.shape[1], kv_size)

    # q_norm / k_norm: identity
    mod.q_norm.side_effect = lambda x: x
    mod.k_norm.side_effect = lambda x: x

    return mod


def _run_forward(
    mod: WinMLQwen3Attention,
    seq_len: int,
    max_cache_len: int,
    kv_dtype: torch.dtype = torch.float16,
) -> list[torch.Tensor]:
    """Invoke WinMLQwen3Attention.forward and capture rotary_emb position_ids."""
    hidden = torch.zeros(1, seq_len, mod.config.num_attention_heads * mod.head_dim)
    past_keys = torch.zeros(
        1, mod.config.num_key_value_heads, max_cache_len, mod.head_dim, dtype=kv_dtype
    )
    past_vals = torch.zeros(
        1, mod.config.num_key_value_heads, max_cache_len, mod.head_dim, dtype=kv_dtype
    )
    past_seq_len = torch.zeros(1, 1, dtype=torch.int32)
    total_seq_len = torch.tensor([max_cache_len], dtype=torch.int32)

    captured_pos_ids: list[torch.Tensor] = []

    def _fake_rotary_emb(values, position_ids):
        captured_pos_ids.append(position_ids)
        seq_dim = position_ids.shape[-1]
        cos = torch.ones(1, seq_dim, mod.head_dim, dtype=values.dtype)
        sin = torch.zeros(1, seq_dim, mod.head_dim, dtype=values.dtype)
        return cos, sin

    mod.rotary_emb.side_effect = _fake_rotary_emb

    # GQA op: return (attn_out, present_keys, present_values)
    with patch(
        "winml.modelkit.models.hf.qwen3.qwen3_modeling.GroupQueryAttentionOnnxExport.apply"
    ) as mock_gqa:
        attn_out = torch.zeros(
            1, seq_len, mod.config.num_attention_heads * mod.head_dim, dtype=kv_dtype
        )
        mock_gqa.return_value = (attn_out, past_keys, past_vals)
        WinMLQwen3Attention.forward(
            mod,
            hidden,
            past_key_value=(past_keys, past_vals),
            past_seq_len=past_seq_len,
            total_seq_len=total_seq_len,
        )

    return captured_pos_ids


# ---------------------------------------------------------------------------
# Tests for prepare_for_onnx_export — _max_rope_len attribute
# ---------------------------------------------------------------------------


class TestPrepareForOnnxExport:
    def test_explicit_max_rope_len_is_stored(self):
        """prepare_for_onnx_export stores the supplied max_rope_len as a Python int."""
        mod = _make_attention_module(max_position_embeddings=40960)
        WinMLQwen3Attention.prepare_for_onnx_export(mod, matmul_to_conv=False, max_rope_len=4096)
        assert mod._max_rope_len == 4096
        assert isinstance(mod._max_rope_len, int)

    def test_fallback_to_max_position_embeddings(self):
        """Without max_rope_len, _max_rope_len falls back to max_position_embeddings."""
        mod = _make_attention_module(max_position_embeddings=512)
        WinMLQwen3Attention.prepare_for_onnx_export(mod, matmul_to_conv=False)
        assert mod._max_rope_len == 512

    def test_max_rope_len_is_plain_int(self):
        """_max_rope_len must be a plain int, not a tensor or other type."""
        mod = _make_attention_module()
        WinMLQwen3Attention.prepare_for_onnx_export(mod, matmul_to_conv=False, max_rope_len=4096)
        assert type(mod._max_rope_len) is int


# ---------------------------------------------------------------------------
# Tests for forward — uses _max_rope_len, ignores total_seq_len for rope
# ---------------------------------------------------------------------------


class TestRopeCacheSizing:
    def test_forward_uses_max_rope_len_attribute(self):
        """forward passes torch.arange(_max_rope_len) to rotary_emb regardless of total_seq_len."""
        mod = _make_attention_module(max_position_embeddings=40960)
        mod._max_rope_len = 256  # explicitly set; different from total_seq_len=4096 below

        pos_ids = _run_forward(mod, seq_len=1, max_cache_len=4096)
        assert len(pos_ids) == 1
        assert pos_ids[0].shape[-1] == 256, (
            f"Expected rope cache length 256 (from _max_rope_len) but got {pos_ids[0].shape[-1]}"
        )

    def test_rope_cache_does_not_depend_on_total_seq_len(self):
        """Changing total_seq_len does NOT change the rope cache length."""
        mod_a = _make_attention_module()
        mod_a._max_rope_len = 512
        pos_a = _run_forward(mod_a, seq_len=1, max_cache_len=128)

        mod_b = _make_attention_module()
        mod_b._max_rope_len = 512
        pos_b = _run_forward(mod_b, seq_len=1, max_cache_len=4096)

        # Both use _max_rope_len=512; total_seq_len differs but rope length must match
        assert pos_a[0].shape[-1] == pos_b[0].shape[-1] == 512

    def test_fallback_rope_len_is_max_position_embeddings(self):
        """Without max_rope_len arg, forward uses max_position_embeddings as rope len."""
        mod = _make_attention_module(max_position_embeddings=512)
        WinMLQwen3Attention.prepare_for_onnx_export(mod, matmul_to_conv=False)
        pos_ids = _run_forward(mod, seq_len=1, max_cache_len=256)
        assert pos_ids[0].shape[-1] == 512
