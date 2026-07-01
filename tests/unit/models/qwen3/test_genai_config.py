# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the Qwen3 genai config builder."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from winml.modelkit.models.hf.qwen3.genai import (
    DEFAULT_CONTEXT_FILENAME,
    DEFAULT_EMBEDDINGS_FILENAME,
    DEFAULT_ITERATOR_FILENAME,
    DEFAULT_LM_HEAD_FILENAME,
    DecoderIOMapping,
    PipelineStage,
    build_genai_config,
    build_qwen3_transformer_only_stages,
)
from winml.modelkit.utils.genai import _detect_format_patterns


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


def _make_pipeline(
    num_layers: int = 28,
    *,
    emb_filename: str = DEFAULT_EMBEDDINGS_FILENAME,
    ctx_filename: str = DEFAULT_CONTEXT_FILENAME,
    iter_filename: str = DEFAULT_ITERATOR_FILENAME,
    lmh_filename: str = DEFAULT_LM_HEAD_FILENAME,
) -> list[PipelineStage]:
    """Build a standard 4-stage pipeline for use in unit tests."""
    ctx_inputs = [
        "input_hidden_states",
        "past_seq_len",
        "total_seq_len",
        *[f"past_keys_{i}" for i in range(num_layers)],
        *[f"past_values_{i}" for i in range(num_layers)],
    ]
    ctx_outputs = [
        "output_hidden_states",
        *[f"present_keys_{i}" for i in range(num_layers)],
        *[f"present_values_{i}" for i in range(num_layers)],
    ]
    return [
        PipelineStage(
            "embeddings", emb_filename, True, True, ["input_ids"], ["input_hidden_states"]
        ),
        PipelineStage("context", ctx_filename, True, False, ctx_inputs, ctx_outputs),
        PipelineStage("iterator", iter_filename, False, True, ctx_inputs, ctx_outputs),
        PipelineStage(
            "lm_head",
            lmh_filename,
            True,
            True,
            ["output_hidden_states"],
            ["logits"],
            is_lm_head=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests: build_genai_config
# ---------------------------------------------------------------------------


class TestBuildGenaiConfig:
    def setup_method(self) -> None:
        self.cfg = _mock_config()
        self.result = build_genai_config(
            self.cfg,
            max_cache_len=256,
            prefill_seq_len=64,
            pipeline=_make_pipeline(),
        )

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

    def test_sliding_window_present_when_prefill_seq_len_given(self) -> None:
        sw = self.result["model"]["decoder"]["sliding_window"]
        assert sw["window_size"] == 64
        assert sw["slide_inputs"] is True
        assert sw["slide_key_value_cache"] is False

    def test_sliding_window_absent_when_prefill_seq_len_none(self) -> None:
        result = build_genai_config(
            self.cfg,
            max_cache_len=256,
            prefill_seq_len=None,
            pipeline=_make_pipeline(),
        )
        assert "sliding_window" not in result["model"]["decoder"]

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

    def test_custom_decoder_io_mapping(self) -> None:
        custom_io = DecoderIOMapping(
            past_key_names="k_%d",
            past_value_names="v_%d",
            present_key_names="pk_%d",
            present_value_names="pv_%d",
        )
        result = build_genai_config(
            self.cfg,
            max_cache_len=256,
            prefill_seq_len=64,
            pipeline=_make_pipeline(),
            decoder_io=custom_io,
        )
        dec_inputs = result["model"]["decoder"]["inputs"]
        assert dec_inputs["past_key_names"] == "k_%d"
        assert dec_inputs["past_value_names"] == "v_%d"
        dec_outputs = result["model"]["decoder"]["outputs"]
        assert dec_outputs["present_key_names"] == "pk_%d"
        assert dec_outputs["present_value_names"] == "pv_%d"

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
            pipeline=_make_pipeline(
                emb_filename="emb.onnx",
                ctx_filename="prefill.onnx",
                iter_filename="decode.onnx",
                lmh_filename="head.onnx",
            ),
        )
        pipeline = result["model"]["decoder"]["pipeline"]
        assert pipeline[0]["embeddings"]["filename"] == "emb.onnx"
        assert pipeline[1]["context"]["filename"] == "prefill.onnx"
        assert pipeline[2]["iterator"]["filename"] == "decode.onnx"
        assert pipeline[3]["lm_head"]["filename"] == "head.onnx"

    def test_eos_token_id_list_preserved(self) -> None:
        cfg = _mock_config(eos_token_id=[151645, 151643])
        result = build_genai_config(
            cfg, max_cache_len=256, prefill_seq_len=64, pipeline=_make_pipeline()
        )
        # ORT genai accepts a list of EOS token IDs; all must be preserved so that
        # any secondary stop token (e.g. 151643 in some Qwen3 variants) is honoured.
        assert result["model"]["eos_token_id"] == [151645, 151643]

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
        result = build_genai_config(
            cfg, max_cache_len=128, prefill_seq_len=32, pipeline=_make_pipeline(2)
        )
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
        result = build_genai_config(
            cfg, max_cache_len=128, prefill_seq_len=32, pipeline=_make_pipeline(2)
        )
        assert result["model"]["pad_token_id"] == 0  # falls back to bos_token_id

    def test_different_layer_count(self) -> None:
        cfg = _mock_config(num_hidden_layers=4)
        result = build_genai_config(
            cfg, max_cache_len=128, prefill_seq_len=32, pipeline=_make_pipeline(4)
        )
        inputs = result["model"]["decoder"]["pipeline"][1]["context"]["inputs"]
        past_keys = [x for x in inputs if x.startswith("past_keys_")]
        assert len(past_keys) == 4
        assert {f"past_keys_{i}" for i in range(4)} == set(past_keys)


# ---------------------------------------------------------------------------
# Tests: _detect_format_patterns
# ---------------------------------------------------------------------------


class TestDetectFormatPatterns:
    def test_detects_two_kv_groups(self) -> None:
        names = [
            "input_hidden_states",
            "past_seq_len",
            "past_keys_0",
            "past_keys_1",
            "past_keys_2",
            "past_values_0",
            "past_values_1",
            "past_values_2",
        ]
        result = _detect_format_patterns(names, num_layers=3)
        assert result == {"past_keys_": "past_keys_%d", "past_values_": "past_values_%d"}

    def test_ignores_incomplete_index_range(self) -> None:
        # Missing index 1 — should not be detected
        names = ["prefix_0", "prefix_2"]
        result = _detect_format_patterns(names, num_layers=3)
        assert "prefix_" not in result

    def test_ignores_wrong_num_layers(self) -> None:
        # 3 entries but num_layers=5
        names = ["kv_0", "kv_1", "kv_2"]
        result = _detect_format_patterns(names, num_layers=5)
        assert len(result) == 0

    def test_empty_input(self) -> None:
        assert _detect_format_patterns([], num_layers=4) == {}

    def test_non_indexed_names_ignored(self) -> None:
        names = ["input_hidden_states", "past_seq_len", "total_seq_len"]
        result = _detect_format_patterns(names, num_layers=3)
        assert result == {}

    def test_single_layer_model(self) -> None:
        names = ["keys_0", "vals_0"]
        result = _detect_format_patterns(names, num_layers=1)
        assert result == {"keys_": "keys_%d", "vals_": "vals_%d"}


# ---------------------------------------------------------------------------
# Tests: build_qwen3_transformer_only_stages
# ---------------------------------------------------------------------------


class TestBuildQwen3TransformerOnlyStages:
    """Uses mocked onnx.load so no real ONNX files are required."""

    def _ctx_inputs(self, n: int = 4) -> list[str]:
        return [
            "input_hidden_states",
            "past_seq_len",
            "total_seq_len",
            *[f"past_keys_{i}" for i in range(n)],
            *[f"past_values_{i}" for i in range(n)],
        ]

    def _ctx_outputs(self, n: int = 4) -> list[str]:
        return [
            "output_hidden_states",
            *[f"present_keys_{i}" for i in range(n)],
            *[f"present_values_{i}" for i in range(n)],
        ]

    def _patch_onnx(self, n: int = 4):
        ctx_io = (self._ctx_inputs(n), self._ctx_outputs(n))
        iter_io = (self._ctx_inputs(n), self._ctx_outputs(n))
        return patch(
            "winml.modelkit.utils.genai._introspect_onnx_io",
            side_effect=[ctx_io, iter_io],
        )

    def test_returns_four_stages(self) -> None:
        with self._patch_onnx():
            stages, _ = build_qwen3_transformer_only_stages("ctx.onnx", "iter.onnx", num_layers=4)
        assert len(stages) == 4
        assert [s.name for s in stages] == ["embeddings", "context", "iterator", "lm_head"]

    def test_detected_kv_format_patterns(self) -> None:
        with self._patch_onnx():
            _, decoder_io = build_qwen3_transformer_only_stages(
                "ctx.onnx", "iter.onnx", num_layers=4
            )
        assert decoder_io.past_key_names == "past_keys_%d"
        assert decoder_io.past_value_names == "past_values_%d"
        assert decoder_io.present_key_names == "present_keys_%d"
        assert decoder_io.present_value_names == "present_values_%d"

    def test_detected_seq_len_names(self) -> None:
        with self._patch_onnx():
            _, decoder_io = build_qwen3_transformer_only_stages(
                "ctx.onnx", "iter.onnx", num_layers=4
            )
        assert decoder_io.past_sequence_length == "past_seq_len"
        assert decoder_io.total_sequence_length == "total_seq_len"

    def test_context_stage_inputs_from_onnx(self) -> None:
        with self._patch_onnx(n=4):
            stages, _ = build_qwen3_transformer_only_stages("ctx.onnx", "iter.onnx", num_layers=4)
        ctx_stage = next(s for s in stages if s.name == "context")
        assert "input_hidden_states" in ctx_stage.inputs
        assert "past_keys_0" in ctx_stage.inputs
        assert "past_values_3" in ctx_stage.inputs

    def test_custom_filenames(self) -> None:
        with self._patch_onnx():
            stages, _ = build_qwen3_transformer_only_stages(
                "ctx.onnx",
                "iter.onnx",
                num_layers=4,
                context_filename="prefill.onnx",
                iterator_filename="decode.onnx",
                embeddings_filename="emb.onnx",
                lm_head_filename="head.onnx",
            )
        names = {s.name: s.filename for s in stages}
        assert names["context"] == "prefill.onnx"
        assert names["iterator"] == "decode.onnx"
        assert names["embeddings"] == "emb.onnx"
        assert names["lm_head"] == "head.onnx"

    def test_roundtrip_with_build_genai_config(self) -> None:
        """build_qwen3_transformer_only_stages output feeds build_genai_config cleanly."""
        with self._patch_onnx(n=4):
            stages, decoder_io = build_qwen3_transformer_only_stages(
                "ctx.onnx", "iter.onnx", num_layers=4
            )
        cfg = _mock_config(num_hidden_layers=4)
        result = build_genai_config(
            cfg,
            max_cache_len=128,
            prefill_seq_len=32,
            pipeline=stages,
            decoder_io=decoder_io,
        )
        assert result["model"]["type"] == "decoder-pipeline"
        assert len(result["model"]["decoder"]["pipeline"]) == 4

    def test_cpu_ep_no_session_options(self) -> None:
        """Default cpu ep: context/iterator stages have no session_options."""
        with self._patch_onnx():
            stages, _ = build_qwen3_transformer_only_stages(
                "ctx.onnx", "iter.onnx", num_layers=4, ep="cpu"
            )
        ctx = next(s for s in stages if s.name == "context")
        itr = next(s for s in stages if s.name == "iterator")
        assert ctx.session_options is None
        assert itr.session_options is None

    def test_qnn_ep_injects_session_options(self) -> None:
        """ep='qnn': context/iterator get QNN session_options; emb/lm_head do not."""
        with self._patch_onnx():
            stages, _ = build_qwen3_transformer_only_stages(
                "ctx.onnx", "iter.onnx", num_layers=4, ep="qnn"
            )
        stage_map = {s.name: s for s in stages}
        assert stage_map["embeddings"].session_options is None
        assert stage_map["lm_head"].session_options is None
        ctx_opts = stage_map["context"].session_options
        itr_opts = stage_map["iterator"].session_options
        assert ctx_opts is not None
        assert itr_opts is not None
        assert ctx_opts["provider_options"][0]["qnn"]["backend_path"] == "QnnHtp.dll"
        assert itr_opts["log_id"] == "onnxruntime-genai.iterator"

    def test_qnn_session_options_in_serialized_config(self) -> None:
        """QNN session_options appear in genai_config.json pipeline output."""
        with self._patch_onnx():
            stages, decoder_io = build_qwen3_transformer_only_stages(
                "ctx.onnx", "iter.onnx", num_layers=4, ep="qnn"
            )
        cfg = build_genai_config(
            _mock_config(num_hidden_layers=4),
            max_cache_len=256,
            prefill_seq_len=64,
            pipeline=stages,
            decoder_io=decoder_io,
        )
        pipeline = cfg["model"]["decoder"]["pipeline"]
        ctx_dict = next(s for s in pipeline if "context" in s)["context"]
        itr_dict = next(s for s in pipeline if "iterator" in s)["iterator"]
        emb_dict = next(s for s in pipeline if "embeddings" in s)["embeddings"]
        assert "session_options" in ctx_dict
        assert "session_options" in itr_dict
        assert "session_options" not in emb_dict

    def test_custom_soc_model(self) -> None:
        """soc_model parameter propagates to QNN provider_options."""
        with self._patch_onnx():
            stages, _ = build_qwen3_transformer_only_stages(
                "ctx.onnx", "iter.onnx", num_layers=4, ep="qnn", soc_model="73"
            )
        ctx = next(s for s in stages if s.name == "context")
        assert ctx.session_options["provider_options"][0]["qnn"]["soc_model"] == "73"
