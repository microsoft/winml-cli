# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the Qwen3 genai config builder."""

from __future__ import annotations

from types import SimpleNamespace

from winml.modelkit.models.hf.qwen3.genai import (
    DEFAULT_CONTEXT_FILENAME,
    DEFAULT_EMBEDDINGS_FILENAME,
    DEFAULT_ITERATOR_FILENAME,
    DEFAULT_LM_HEAD_FILENAME,
    build_genai_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config(
    *,
    num_hidden_layers: int = 28,
    hidden_size: int = 1024,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    bos_token_id: int = 151643,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    vocab_size: int = 151936,
) -> SimpleNamespace:
    """Return a minimal stand-in for a HF PretrainedConfig."""
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildGenaiConfig:
    def setup_method(self) -> None:
        self.cfg = _mock_config()
        self.result = build_genai_config(self.cfg, max_cache_len=256, prefill_seq_len=64)

    def test_top_level_model_type(self) -> None:
        assert self.result["model"]["type"] == "decoder-pipeline"

    def test_token_ids(self) -> None:
        m = self.result["model"]
        assert m["bos_token_id"] == 151643
        assert m["eos_token_id"] == 151645
        assert m["pad_token_id"] == 151643
        assert m["vocab_size"] == 151936

    def test_context_length_equals_max_cache_len(self) -> None:
        assert self.result["model"]["context_length"] == 256

    def test_search_max_length_equals_context_length(self) -> None:
        assert self.result["search"]["max_length"] == self.result["model"]["context_length"]

    def test_search_past_present_share_buffer(self) -> None:
        assert self.result["search"]["past_present_share_buffer"] is True

    def test_decoder_architecture_params(self) -> None:
        dec = self.result["model"]["decoder"]
        assert dec["hidden_size"] == 1024
        assert dec["num_attention_heads"] == 16
        assert dec["num_key_value_heads"] == 8
        assert dec["num_hidden_layers"] == 28
        assert dec["head_size"] == 128

    def test_sliding_window_size_equals_prefill_seq_len(self) -> None:
        sw = self.result["model"]["decoder"]["sliding_window"]
        assert sw["window_size"] == 64
        assert sw["slide_inputs"] is True
        assert sw["slide_key_value_cache"] is False

    def test_decoder_io_tensor_names(self) -> None:
        inputs = self.result["model"]["decoder"]["inputs"]
        assert inputs["past_sequence_length"] == "past_seq_len"
        assert inputs["total_sequence_length"] == "total_seq_len"
        assert inputs["past_key_names"] == "past_keys_%d"
        assert inputs["past_value_names"] == "past_values_%d"
        outputs = self.result["model"]["decoder"]["outputs"]
        assert outputs["logits"] == "logits"
        assert outputs["present_key_names"] == "present_keys_%d"
        assert outputs["present_value_names"] == "present_values_%d"

    def test_pipeline_has_four_stages(self) -> None:
        pipeline = self.result["model"]["decoder"]["pipeline"]
        assert len(pipeline) == 4
        stage_names = [next(iter(s.keys())) for s in pipeline]
        assert stage_names == ["embeddings", "context", "iterator", "lm_head"]

    def test_embeddings_stage(self) -> None:
        stage = self.result["model"]["decoder"]["pipeline"][0]["embeddings"]
        assert stage["filename"] == DEFAULT_EMBEDDINGS_FILENAME
        assert stage["inputs"] == ["input_ids"]
        assert stage["outputs"] == ["input_hidden_states"]
        assert stage["run_on_prompt"] is True
        assert stage["run_on_token_gen"] is True

    def test_context_stage(self) -> None:
        stage = self.result["model"]["decoder"]["pipeline"][1]["context"]
        assert stage["filename"] == DEFAULT_CONTEXT_FILENAME
        assert "input_hidden_states" in stage["inputs"]
        assert "past_seq_len" in stage["inputs"]
        assert "total_seq_len" in stage["inputs"]
        assert stage["run_on_prompt"] is True
        assert stage["run_on_token_gen"] is False

    def test_iterator_stage(self) -> None:
        stage = self.result["model"]["decoder"]["pipeline"][2]["iterator"]
        assert stage["filename"] == DEFAULT_ITERATOR_FILENAME
        assert stage["run_on_prompt"] is False
        assert stage["run_on_token_gen"] is True

    def test_lm_head_stage(self) -> None:
        stage = self.result["model"]["decoder"]["pipeline"][3]["lm_head"]
        assert stage["filename"] == DEFAULT_LM_HEAD_FILENAME
        assert stage["inputs"] == ["output_hidden_states"]
        assert stage["outputs"] == ["logits"]
        assert stage["is_lm_head"] is True
        assert stage["run_on_prompt"] is True
        assert stage["run_on_token_gen"] is True

    def test_context_kv_inputs_count(self) -> None:
        """context.inputs must include all 28 past_keys + 28 past_values."""
        inputs = self.result["model"]["decoder"]["pipeline"][1]["context"]["inputs"]
        past_keys = [x for x in inputs if x.startswith("past_keys_")]
        past_values = [x for x in inputs if x.startswith("past_values_")]
        assert len(past_keys) == 28
        assert len(past_values) == 28
        # All layer indices present
        assert set(past_keys) == {f"past_keys_{i}" for i in range(28)}
        assert set(past_values) == {f"past_values_{i}" for i in range(28)}

    def test_context_outputs_kv_count(self) -> None:
        outputs = self.result["model"]["decoder"]["pipeline"][1]["context"]["outputs"]
        present_keys = [x for x in outputs if x.startswith("present_keys_")]
        present_values = [x for x in outputs if x.startswith("present_values_")]
        assert len(present_keys) == 28
        assert len(present_values) == 28

    def test_context_and_iterator_have_same_io(self) -> None:
        ctx = self.result["model"]["decoder"]["pipeline"][1]["context"]
        itr = self.result["model"]["decoder"]["pipeline"][2]["iterator"]
        assert ctx["inputs"] == itr["inputs"]
        assert ctx["outputs"] == itr["outputs"]

    def test_custom_filenames(self) -> None:
        result = build_genai_config(
            self.cfg,
            max_cache_len=512,
            prefill_seq_len=128,
            embeddings_filename="emb.onnx",
            context_filename="prefill.onnx",
            iterator_filename="decode.onnx",
            lm_head_filename="head.onnx",
        )
        pipeline = result["model"]["decoder"]["pipeline"]
        assert pipeline[0]["embeddings"]["filename"] == "emb.onnx"
        assert pipeline[1]["context"]["filename"] == "prefill.onnx"
        assert pipeline[2]["iterator"]["filename"] == "decode.onnx"
        assert pipeline[3]["lm_head"]["filename"] == "head.onnx"

    def test_eos_token_id_list_unpacked(self) -> None:
        cfg = _mock_config(eos_token_id=[151645, 151643])
        result = build_genai_config(cfg, max_cache_len=256, prefill_seq_len=64)
        assert result["model"]["eos_token_id"] == 151645

    def test_head_size_derived_when_head_dim_missing(self) -> None:
        cfg = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            # no head_dim attribute
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
            vocab_size=32000,
        )
        result = build_genai_config(cfg, max_cache_len=128, prefill_seq_len=32)
        # head_size = hidden_size // num_attention_heads = 512 // 8 = 64
        assert result["model"]["decoder"]["head_size"] == 64

    def test_pad_token_id_falls_back_to_bos(self) -> None:
        cfg = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=None,
            vocab_size=32000,
        )
        result = build_genai_config(cfg, max_cache_len=128, prefill_seq_len=32)
        assert result["model"]["pad_token_id"] == 0  # falls back to bos_token_id

    def test_different_layer_count(self) -> None:
        cfg = _mock_config(num_hidden_layers=4)
        result = build_genai_config(cfg, max_cache_len=128, prefill_seq_len=32)
        inputs = result["model"]["decoder"]["pipeline"][1]["context"]["inputs"]
        past_keys = [x for x in inputs if x.startswith("past_keys_")]
        assert len(past_keys) == 4
        assert {f"past_keys_{i}" for i in range(4)} == set(past_keys)
