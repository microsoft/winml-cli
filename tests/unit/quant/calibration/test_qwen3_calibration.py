# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the transformer-only Qwen3 quant calibration readers.

These are fast, offline tests (no model download, no ONNX Runtime): they
exercise the graph-shape introspection, GroupQueryAttention node discovery,
and the exact feed contract (names / dtypes / shapes) the two calibration
readers must satisfy. All expectations are derived in-code from the inputs,
never hardcoded from a model run.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from onnx import TensorProto, helper
from onnx import save as onnx_save

from winml.modelkit.quant.calibration.qwen3_transformer_only import (
    Qwen3DecodeTrajectoryCalibReader,
    Qwen3TransformerOnlyCalibReader,
    _gqa_node_names,
    _graph_shapes,
)


NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 4
HIDDEN = NUM_KV_HEADS * HEAD_DIM
VOCAB = 16


def _fake_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=NUM_LAYERS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN,
        num_attention_heads=NUM_KV_HEADS,
    )


def _build_tiny_onnx(path, *, seq_len: int, max_cache_len: int) -> None:
    """Write a minimal graph carrying the inputs the readers introspect."""
    inputs = [
        helper.make_tensor_value_info(
            "input_hidden_states", TensorProto.FLOAT, [1, seq_len, HIDDEN]
        ),
        helper.make_tensor_value_info(
            "past_keys_0", TensorProto.FLOAT16, [1, NUM_KV_HEADS, max_cache_len, HEAD_DIM]
        ),
    ]
    out = helper.make_tensor_value_info(
        "output_hidden_states", TensorProto.FLOAT, [1, seq_len, HIDDEN]
    )
    gqa = helper.make_node(
        "GroupQueryAttention",
        ["input_hidden_states"],
        ["attn_out"],
        name="gqa_layer_0",
        domain="com.microsoft",
    )
    identity = helper.make_node("Identity", ["attn_out"], ["output_hidden_states"])
    graph = helper.make_graph([gqa, identity], "tiny", inputs, [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx_save(model, str(path))


def test_graph_shapes_and_gqa_nodes(tmp_path):
    p = tmp_path / "tiny.onnx"
    _build_tiny_onnx(p, seq_len=1, max_cache_len=16)

    assert _graph_shapes(p) == (1, 16)
    assert _gqa_node_names(p) == ["gqa_layer_0"]


def test_graph_shapes_prefill(tmp_path):
    p = tmp_path / "tiny_prefill.onnx"
    _build_tiny_onnx(p, seq_len=64, max_cache_len=256)

    assert _graph_shapes(p) == (64, 256)


def _drain(reader) -> list[dict[str, np.ndarray]]:
    feeds = []
    while (feed := reader.get_next()) is not None:
        feeds.append(feed)
    return feeds


def test_prefill_reader_feed_contract():
    seq_len, max_cache_len = 4, 16
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    token_ids = [torch.tensor([[1, 2, 3, 4, 5]])]

    reader = Qwen3TransformerOnlyCalibReader(
        embed,
        _fake_config(),
        token_ids,
        seq_len=seq_len,
        max_cache_len=max_cache_len,
    )
    feeds = _drain(reader)

    assert len(feeds) == len(token_ids)
    feed = feeds[0]

    # input_hidden_states: FP32, truncated to seq_len.
    assert feed["input_hidden_states"].dtype == np.float32
    assert feed["input_hidden_states"].shape == (1, seq_len, HIDDEN)

    # seqlens_k contract: past_seq_len = seq_len - 1 (INT32 [1,1]).
    assert feed["past_seq_len"].dtype == np.int32
    np.testing.assert_array_equal(feed["past_seq_len"], [[seq_len - 1]])

    # total_seq_len: full cache (INT32 [1]).
    assert feed["total_seq_len"].dtype == np.int32
    np.testing.assert_array_equal(feed["total_seq_len"], [max_cache_len])

    # KV buffers: FP16, full cache shape, present for every layer.
    for i in range(NUM_LAYERS):
        for prefix in ("past_keys_", "past_values_"):
            kv = feed[f"{prefix}{i}"]
            assert kv.dtype == np.float16
            assert kv.shape == (1, NUM_KV_HEADS, max_cache_len, HEAD_DIM)

    # rewind() replays the same samples.
    reader.rewind()
    assert len(_drain(reader)) == len(token_ids)


def test_prefill_reader_pads_short_prompts():
    seq_len = 6  # longer than the 3-token prompt -> must pad
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    token_ids = [torch.tensor([[1, 2, 3]])]

    reader = Qwen3TransformerOnlyCalibReader(
        embed, _fake_config(), token_ids, seq_len=seq_len, max_cache_len=16
    )
    feed = _drain(reader)[0]
    assert feed["input_hidden_states"].shape == (1, seq_len, HIDDEN)


class _StubCausalLM:
    """Minimal HF-like model: grows a tuple-of-tuples KV cache by 1 each call.

    Always predicts ``next_token`` so the trajectory is deterministic.
    """

    def __init__(self, next_token: int) -> None:
        self.next_token = next_token

    def _cache(self, length: int):
        return tuple(
            (
                torch.randn(1, NUM_KV_HEADS, length, HEAD_DIM),
                torch.randn(1, NUM_KV_HEADS, length, HEAD_DIM),
            )
            for _ in range(NUM_LAYERS)
        )

    def __call__(self, input_ids=None, past_key_values=None, use_cache=True):
        if past_key_values is None:
            length = input_ids.shape[1]
            query_len = length
        else:
            length = past_key_values[0][0].shape[2] + input_ids.shape[1]
            query_len = input_ids.shape[1]
        logits = torch.full((1, query_len, VOCAB), -10.0)
        logits[..., self.next_token] = 10.0
        return SimpleNamespace(past_key_values=self._cache(length), logits=logits)


def test_decode_trajectory_reader_grows_past_seq_len():
    prefill_seq, decode_steps, max_cache_len = 2, 3, 16
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    hf_model = _StubCausalLM(next_token=5)
    token_ids = [torch.tensor([[1, 2, 3, 4]])]  # truncated to prefill_seq=2

    reader = Qwen3DecodeTrajectoryCalibReader(
        hf_model,
        embed,
        _fake_config(),
        token_ids,
        prefill_seq=prefill_seq,
        max_cache_len=max_cache_len,
        decode_steps=decode_steps,
    )
    feeds = _drain(reader)

    assert len(feeds) == len(token_ids) * decode_steps

    # past_seq_len must grow monotonically from prefill_seq (real decode), not
    # stay pinned at 0 like the degenerate single-token reader.
    seq_lens = [int(f["past_seq_len"][0, 0]) for f in feeds]
    assert seq_lens == [prefill_seq, prefill_seq + 1, prefill_seq + 2]

    for f in feeds:
        # One token per decode step.
        assert f["input_hidden_states"].shape == (1, 1, HIDDEN)
        assert f["input_hidden_states"].dtype == np.float32
        cur_len = int(f["past_seq_len"][0, 0])
        for i in range(NUM_LAYERS):
            kv = f[f"past_keys_{i}"]
            assert kv.dtype == np.float16
            assert kv.shape == (1, NUM_KV_HEADS, max_cache_len, HEAD_DIM)
            # Positions beyond the valid context stay zero-padded.
            assert np.all(kv[:, :, cur_len:, :] == 0)


def test_decode_trajectory_reader_respects_max_cache():
    prefill_seq, decode_steps, max_cache_len = 4, 10, 6
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    hf_model = _StubCausalLM(next_token=2)
    token_ids = [torch.tensor([[1, 2, 3, 4, 5, 6]])]

    reader = Qwen3DecodeTrajectoryCalibReader(
        hf_model,
        embed,
        _fake_config(),
        token_ids,
        prefill_seq=prefill_seq,
        max_cache_len=max_cache_len,
        decode_steps=decode_steps,
    )
    feeds = _drain(reader)
    # Trajectory must stop once the cache is full (cur_len reaches max_cache_len).
    assert len(feeds) == max_cache_len - prefill_seq
    assert max(int(f["past_seq_len"][0, 0]) for f in feeds) == max_cache_len - 1


def test_finalize_pins_static_w8a8_scheme(tmp_path, monkeypatch):
    """The finalizer is authoritative over the precision policy.

    The precision-driven build keys the quantizer dispatch on ``config.mode``,
    so the transformer-only policy must pin ``mode="static"`` (QDQ) along with
    the reference-matched w8a8 dtypes/symmetry + GQA exclusion — even when the
    incoming config arrived as a non-QDQ mode (e.g. ``fp16``/``rtn``).
    """
    from winml.modelkit.quant import WinMLQuantizationConfig
    from winml.modelkit.quant.calibration import qwen3_transformer_only as mod

    p = tmp_path / "prefill.onnx"
    _build_tiny_onnx(p, seq_len=64, max_cache_len=128)

    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    fake_model = SimpleNamespace(
        config=_fake_config(),
        eval=lambda: None,
        get_input_embeddings=lambda: embed,
    )
    monkeypatch.setattr(
        mod, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=lambda *a, **k: fake_model)
    )
    monkeypatch.setattr(
        mod, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: object())
    )
    monkeypatch.setattr(mod, "_load_gsm8k_prompts", lambda n: ["hi"] * n)
    monkeypatch.setattr(
        mod,
        "_tokenize_prompts",
        lambda tok, prompts, n: [torch.zeros((1, 64), dtype=torch.long) for _ in range(n)],
    )

    # Hostile incoming mode: a non-QDQ policy that must be overridden.
    quant = WinMLQuantizationConfig(mode="fp16")
    quant.samples = 2

    result = mod.finalize_transformer_only_quant_config(quant, onnx_path=p)

    assert result.mode == "static"
    assert result.weight_type == "int8"
    assert result.activation_type == "uint8"
    assert result.weight_symmetric is True
    assert result.activation_symmetric is False
    assert result.calibration_method == "minmax"
    assert result.nodes_to_exclude == ["gqa_layer_0"]
    assert result.calibration_data is not None
