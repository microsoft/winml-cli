# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for WinMLQwen3Attention.forward — rope cache sizing."""

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
    """Build a minimal mock bound to WinMLQwen3Attention.forward."""
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
# Tests
# ---------------------------------------------------------------------------


class TestRopeCacheSizing:
    def test_rope_cache_uses_total_seq_len_not_max_position_embeddings(self):
        """rope cache length == max_cache_len, not max_position_embeddings."""
        mod = _make_attention_module(max_position_embeddings=40960)
        pos_ids = _run_forward(mod, seq_len=64, max_cache_len=256)
        assert len(pos_ids) == 1
        assert pos_ids[0].shape[-1] == 256, (
            f"Expected rope cache length 256 but got {pos_ids[0].shape[-1]}"
        )

    def test_rope_cache_matches_max_cache_len(self):
        """rope cache length equals the max_cache_len used for KV cache."""
        for max_cache_len in (128, 512, 4096):
            mod = _make_attention_module(max_position_embeddings=40960)
            pos_ids = _run_forward(mod, seq_len=1, max_cache_len=max_cache_len)
            assert pos_ids[0].shape[-1] == max_cache_len

    def test_rope_cache_much_smaller_than_max_position_embeddings(self):
        """With max_cache_len=256, cache is 160x smaller than full rope."""
        mod = _make_attention_module(max_position_embeddings=40960)
        pos_ids = _run_forward(mod, seq_len=1, max_cache_len=256)
        assert pos_ids[0].shape[-1] < mod.config.max_position_embeddings

    def test_fallback_when_total_seq_len_is_none(self):
        """When total_seq_len is None, falls back to max_position_embeddings."""
        mod = _make_attention_module(max_position_embeddings=512)
        hidden = torch.zeros(1, 1, mod.config.num_attention_heads * mod.head_dim)
        past_keys = torch.zeros(
            1, mod.config.num_key_value_heads, 256, mod.head_dim, dtype=torch.float16
        )
        past_vals = torch.zeros_like(past_keys)

        captured: list[torch.Tensor] = []

        def _fake_rotary_emb(values, position_ids):
            captured.append(position_ids)
            seq_dim = position_ids.shape[-1]
            cos = torch.ones(1, seq_dim, mod.head_dim, dtype=values.dtype)
            sin = torch.zeros(1, seq_dim, mod.head_dim, dtype=values.dtype)
            return cos, sin

        mod.rotary_emb.side_effect = _fake_rotary_emb

        with patch(
            "winml.modelkit.models.hf.qwen3.qwen3_modeling.GroupQueryAttentionOnnxExport.apply"
        ) as mock_gqa:
            attn_out = torch.zeros(
                1, 1, mod.config.num_attention_heads * mod.head_dim, dtype=torch.float16
            )
            mock_gqa.return_value = (attn_out, past_keys, past_vals)
            WinMLQwen3Attention.forward(
                mod,
                hidden,
                past_key_value=(past_keys, past_vals),
                past_seq_len=torch.zeros(1, 1, dtype=torch.int32),
                total_seq_len=None,
            )

        assert captured[0].shape[-1] == 512  # falls back to max_position_embeddings
