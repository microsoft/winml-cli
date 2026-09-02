# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for inference/pipeline.py adaptations.

Covers:
  - _HF_PIPELINE_TASK_MAP (sentence-similarity → feature-extraction)
  - _adapt_tokenizer_padding Pattern A / B / C detection
  - _adapt_image_processor_size multi-modal shape scanning
  - _detect_tokenizer_dict_param
"""

from __future__ import annotations

import inspect
import warnings
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from winml.modelkit.inference.engine import _discover_pipeline_params
from winml.modelkit.inference.pipeline import (
    _HF_PIPELINE_TASK_MAP,
    _adapt_image_processor_size,
    _adapt_tokenizer_padding,
    _detect_tokenizer_dict_param,
    _ExtractiveQuestionAnsweringPipeline,
    _pipeline_component_kwargs,
    _resolve_hf_pipeline_task,
    create_pipeline,
)


# ---------------------------------------------------------------------------
# _HF_PIPELINE_TASK_MAP
# ---------------------------------------------------------------------------


def _make_fast_qa_tokenizer(*, model_max_length: int = 7, padding_side: str = "right"):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    unknown_marker = "[UNK]"
    cls_marker = "[CLS]"
    separator_marker = "[SEP]"
    padding_marker = "[PAD]"
    vocabulary = {
        token: index
        for index, token in enumerate(
            [
                unknown_marker,
                cls_marker,
                separator_marker,
                padding_marker,
                "which",
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token=unknown_marker))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single=f"{cls_marker} $A {separator_marker}",
        pair=f"{cls_marker} $A {separator_marker} $B:1 {separator_marker}:1",
        special_tokens=[
            (cls_marker, vocabulary[cls_marker]),
            (separator_marker, vocabulary[separator_marker]),
        ],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=unknown_marker,
        cls_token=cls_marker,
        sep_token=separator_marker,
        pad_token=padding_marker,
        model_max_length=model_max_length,
        padding_side=padding_side,
    )


def _make_wordpiece_qa_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    unknown_marker = "[UNK]"
    cls_marker = "[CLS]"
    separator_marker = "[SEP]"
    padding_marker = "[PAD]"
    vocabulary = {
        token: index
        for index, token in enumerate(
            [
                unknown_marker,
                cls_marker,
                separator_marker,
                padding_marker,
                "which",
                "alpha",
                "play",
                "##ing",
                "omega",
            ]
        )
    }
    backend = Tokenizer(WordPiece(vocabulary, unk_token=unknown_marker))
    backend.pre_tokenizer = BertPreTokenizer()
    backend.post_processor = TemplateProcessing(
        single=f"{cls_marker} $A {separator_marker}",
        pair=f"{cls_marker} $A {separator_marker} $B:1 {separator_marker}:1",
        special_tokens=[
            (cls_marker, vocabulary[cls_marker]),
            (separator_marker, vocabulary[separator_marker]),
        ],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=unknown_marker,
        cls_token=cls_marker,
        sep_token=separator_marker,
        pad_token=padding_marker,
        model_max_length=16,
    )


def _make_qa_model(tokenizer, *, batch_size: int = 1) -> tuple[MagicMock, str]:
    model = MagicMock()
    input_name = tokenizer.model_input_names[0]
    model.io_config = {
        "input_names": [input_name],
        "input_shapes": [[batch_size, tokenizer.model_max_length]],
    }
    return model, input_name


def _create_qa_compat_pipe(tokenizer, model, task="question-answering"):
    with (
        patch("transformers.__version__", "5.0.0"),
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
    ):
        return create_pipeline(task, model, "test-model")


def test_document_question_answering_uses_native_transformers_pipeline() -> None:
    model = MagicMock()
    pipe = MagicMock(tokenizer=None, image_processor=None)

    with (
        patch("winml.modelkit.inference.pipeline._pipeline_component_kwargs", return_value={}),
        patch("transformers.pipeline", return_value=pipe) as pipeline,
    ):
        result = create_pipeline("document-question-answering", model, "test-model")

    assert result is pipe
    pipeline.assert_called_once_with("document-question-answering", model=model, device="cpu")


class TestHFPipelineTaskMap:
    def test_sentence_similarity_maps_to_feature_extraction(self) -> None:
        assert _HF_PIPELINE_TASK_MAP["sentence-similarity"] == "feature-extraction"

    def test_image_to_text_uses_transformers_4_native_name(self) -> None:
        with patch("transformers.__version__", "4.57.6"):
            assert _resolve_hf_pipeline_task("image-to-text") == "image-to-text"

    def test_image_to_text_maps_to_transformers_5_name(self) -> None:
        with patch("transformers.__version__", "5.0.0"):
            assert _resolve_hf_pipeline_task("image-to-text") == "image-text-to-text"

    @pytest.mark.parametrize(
        "task",
        ["sequence-classification", "next-sentence-prediction"],
    )
    def test_classification_aliases_map_to_transformers_task(self, task: str) -> None:
        assert _HF_PIPELINE_TASK_MAP[task] == "text-classification"

    def test_unknown_task_not_in_map(self) -> None:
        assert "image-classification" not in _HF_PIPELINE_TASK_MAP


class TestPipelineComponentKwargs:
    def test_omits_components_disabled_by_pipeline(self) -> None:
        class ProcessorOnlyPipeline:
            _load_tokenizer = False
            _load_feature_extractor = False
            _load_image_processor = False
            _load_processor = True

        with patch(
            "transformers.pipelines.check_task",
            return_value=("resolved", {"impl": ProcessorOnlyPipeline}, None),
        ):
            result = _pipeline_component_kwargs("test-task", "model-id")

        assert result == {"processor": "model-id"}


class TestCreatePipeline:
    def test_uses_native_question_answering_pipeline_with_transformers_4(self) -> None:
        model = MagicMock()
        pipe = MagicMock()
        pipe.tokenizer = None
        pipe.image_processor = None

        with (
            patch("transformers.__version__", "4.57.6"),
            patch(
                "winml.modelkit.inference.pipeline._pipeline_component_kwargs",
                return_value={},
            ),
            patch("transformers.pipeline", return_value=pipe) as pipeline,
        ):
            result = create_pipeline("question-answering", model, "test-model")

        assert result is pipe
        pipeline.assert_called_once_with("question-answering", model=model, device="cpu")

    def test_threads_trust_remote_code_to_pipeline(self) -> None:
        model = MagicMock()
        pipe = MagicMock()
        pipe.tokenizer = None
        pipe.image_processor = None

        with (
            patch(
                "winml.modelkit.inference.pipeline._pipeline_component_kwargs",
                return_value={},
            ),
            patch("transformers.pipeline", return_value=pipe) as pipeline,
        ):
            create_pipeline(
                "image-classification",
                model,
                "test-model",
                trust_remote_code=True,
            )

        assert pipeline.call_args.kwargs["trust_remote_code"] is True

    def test_places_native_pipeline_inputs_on_cuda(self) -> None:
        model = MagicMock()
        pipe = MagicMock()
        pipe.tokenizer = None
        pipe.image_processor = None

        with (
            patch(
                "winml.modelkit.inference.pipeline._pipeline_component_kwargs",
                return_value={},
            ),
            patch("transformers.pipeline", return_value=pipe) as pipeline,
        ):
            result = create_pipeline(
                "image-classification",
                model,
                "test-model",
                device="cuda",
            )

        assert result is pipe
        pipeline.assert_called_once_with(
            "image-classification",
            model=model,
            device="cuda",
        )

    def test_constructs_real_transformers_pipeline(self) -> None:
        from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor
        from transformers.pipelines import ImageClassificationPipeline

        config = ViTConfig(
            image_size=16,
            patch_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=2,
        )
        model = ViTForImageClassification(config)
        image_processor = ViTImageProcessor(size={"height": 16, "width": 16})

        with patch(
            "winml.modelkit.inference.pipeline._pipeline_component_kwargs",
            return_value={"image_processor": image_processor},
        ):
            result = create_pipeline("image-classification", model)

        assert isinstance(result, ImageClassificationPipeline)

    def test_constructs_removed_question_answering_pipeline(self) -> None:
        model = _make_model_with_shapes([[1, 16]])
        tokenizer = MagicMock()
        tokenizer.is_fast = True

        with (
            patch("transformers.__version__", "5.0.0"),
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as from_pretrained,
        ):
            result = create_pipeline("question-answering", model, "test-model")

        from_pretrained.assert_called_once_with("test-model", use_fast=True)
        assert result.task == "question-answering"
        assert result.tokenizer is tokenizer
        discovered_params = {item["name"]: item for item in _discover_pipeline_params(result)}
        assert set(discovered_params) == {
            "align_to_words",
            "doc_stride",
            "handle_impossible_answer",
            "max_answer_len",
            "top_k",
            "topk",
        }
        assert discovered_params["align_to_words"]["type"] == "boolean"
        assert discovered_params["align_to_words"]["sample_value"] == "True"
        assert discovered_params["topk"]["type"] == "integer"
        assert discovered_params["topk"]["sample_value"] == "5"

    def test_rejects_slow_question_answering_tokenizer_at_construction(self) -> None:
        model = _make_model_with_shapes([[1, 16]])
        tokenizer = MagicMock()
        tokenizer.is_fast = False

        with (
            patch("transformers.__version__", "5.0.0"),
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as from_pretrained,
            pytest.raises(
                ValueError,
                match="Extractive question answering requires a fast tokenizer "
                "with offset mappings",
            ),
        ):
            create_pipeline("question-answering", model, "test-model")

        from_pretrained.assert_called_once_with("test-model", use_fast=True)

    def test_aligns_subword_answers_to_words_by_default_and_on_request(self) -> None:
        tokenizer = _make_wordpiece_qa_tokenizer()
        target_token_id = tokenizer.convert_tokens_to_ids("##ing")
        model, input_name = _make_qa_model(tokenizer)

        def answer_continuation(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_continuation
        pipe = _create_qa_compat_pipe(tokenizer, model)
        context = "alpha playing omega"
        word_start = context.index("playing")
        subword_start = context.index("ing")

        default_result = pipe(question="which", context=context)
        aligned_result = pipe(question="which", context=context, align_to_words=True)
        raw_result = pipe(question="which", context=context, align_to_words=False)

        assert default_result["answer"] == aligned_result["answer"] == "playing"
        assert default_result["start"] == aligned_result["start"] == word_start
        assert default_result["end"] == aligned_result["end"] == word_start + len("playing")
        assert raw_result["answer"] == "ing"
        assert raw_result["start"] == subword_start
        assert raw_result["end"] == subword_start + len("ing")

    def test_sanitize_resolves_align_to_words_none_to_historical_default(self) -> None:
        pipe = _ExtractiveQuestionAnsweringPipeline(MagicMock(), MagicMock())

        _, _, default_postprocess = pipe._sanitize_parameters(align_to_words=None)
        _, _, raw_postprocess = pipe._sanitize_parameters(align_to_words=False)

        assert "align_to_words" not in default_postprocess
        assert raw_postprocess["align_to_words"] is False

    def test_extracts_question_answering_spans_for_single_and_batch_inputs(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        context = "alpha beta gamma delta epsilon"
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        model, input_name = _make_qa_model(tokenizer)

        def answer_target(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_target
        pipe = _create_qa_compat_pipe(tokenizer, model)

        single_result = pipe(question="which", context=context)
        batch_result = pipe(
            question=["which", "which"],
            context=[context, context],
        )

        expected_start = context.index("delta")
        expected_answer = context[expected_start : expected_start + len("delta")]
        assert single_result["answer"] == expected_answer
        assert single_result["start"] == expected_start
        assert single_result["end"] == expected_start + len(expected_answer)
        assert [item["answer"] for item in batch_result] == [expected_answer, expected_answer]
        assert all(set(call.kwargs) == {input_name} for call in model.call_args_list)

    def test_broadcasts_scalar_context_across_batched_questions(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        context = "alpha beta gamma delta epsilon"
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        model, input_name = _make_qa_model(tokenizer)

        def answer_target(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_target
        pipe = _create_qa_compat_pipe(tokenizer, model)

        result = pipe(
            question=["which", "which"],
            context=context,
        )

        assert [answer["answer"] for answer in result] == ["delta", "delta"]

    def test_validates_top_k(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        model, _ = _make_qa_model(tokenizer)
        pipe = _create_qa_compat_pipe(tokenizer, model)

        with pytest.raises(ValueError, match=r"top_k parameter should be >= 1 \(got 0\)"):
            pipe(
                question="which",
                context="alpha beta gamma delta epsilon",
                top_k=0,
            )

    def test_none_postprocess_overrides_use_defaults(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        model, input_name = _make_qa_model(tokenizer)

        def answer_target(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_target
        pipe = _create_qa_compat_pipe(tokenizer, model)

        result = pipe(
            question="which",
            context="alpha beta gamma delta epsilon",
            top_k=None,
            max_answer_len=None,
        )

        assert result["answer"] == "delta"

    def test_sanitize_omits_none_postprocess_overrides(self) -> None:
        pipe = _ExtractiveQuestionAnsweringPipeline(MagicMock(), MagicMock())

        preprocess, forward, postprocess = pipe._sanitize_parameters(
            top_k=None,
            max_answer_len=None,
        )

        assert preprocess == {}
        assert forward == {}
        assert postprocess == {"handle_impossible_answer": False}

    def test_deprecated_topk_alias_warns_and_controls_ranking(self) -> None:
        tokenizer = _make_fast_qa_tokenizer(model_max_length=16)
        start_token_id = tokenizer.convert_tokens_to_ids("delta")
        second_end_token_id = tokenizer.convert_tokens_to_ids("epsilon")
        model, input_name = _make_qa_model(tokenizer)

        def rank_answer_spans(**inputs):
            token_ids = inputs[input_name]
            start_logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            end_logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            start_logits[token_ids == start_token_id] = 10.0
            end_logits[token_ids == start_token_id] = 10.0
            end_logits[token_ids == second_end_token_id] = 8.0
            return SimpleNamespace(start_logits=start_logits, end_logits=end_logits)

        model.side_effect = rank_answer_spans
        pipe = _create_qa_compat_pipe(tokenizer, model)

        with pytest.warns(
            UserWarning,
            match="topk parameter is deprecated, use top_k instead",
        ):
            result = pipe(
                question="which",
                context="alpha beta gamma delta epsilon",
                topk=2,
            )

        assert [answer["answer"] for answer in result] == [
            "delta",
            "delta epsilon",
        ]

    def test_top_k_wins_over_deprecated_topk_without_warning(self) -> None:
        tokenizer = _make_fast_qa_tokenizer(model_max_length=16)
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        model, input_name = _make_qa_model(tokenizer)

        def answer_target(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_target
        pipe = _create_qa_compat_pipe(tokenizer, model)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = pipe(
                question="which",
                context="alpha beta gamma delta epsilon",
                topk=0,
                top_k=1,
            )

        assert result["answer"] == "delta"

    def test_sanitize_resolves_deprecated_topk_alias(self) -> None:
        pipe = _ExtractiveQuestionAnsweringPipeline(MagicMock(), MagicMock())

        with pytest.warns(
            UserWarning,
            match="topk parameter is deprecated, use top_k instead",
        ):
            _, _, alias_postprocess = pipe._sanitize_parameters(topk=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, _, explicit_postprocess = pipe._sanitize_parameters(topk=2, top_k=3)

        assert alias_postprocess["top_k"] == 2
        assert explicit_postprocess["top_k"] == 3

    def test_rejects_non_positive_deprecated_topk(self) -> None:
        pipe = _ExtractiveQuestionAnsweringPipeline(MagicMock(), MagicMock())

        with (
            pytest.warns(
                UserWarning,
                match="topk parameter is deprecated, use top_k instead",
            ),
            pytest.raises(
                ValueError,
                match=r"top_k parameter should be >= 1 \(got 0\)",
            ),
        ):
            pipe(question="which", context="context", topk=0)

    @pytest.mark.parametrize("max_answer_len", [0, -1])
    def test_rejects_non_positive_max_answer_len_before_tokenization(
        self,
        max_answer_len: int,
    ) -> None:
        tokenizer = MagicMock()
        pipe = _ExtractiveQuestionAnsweringPipeline(MagicMock(), tokenizer)

        with pytest.raises(
            ValueError,
            match=rf"max_answer_len parameter should be >= 1 \(got {max_answer_len}",
        ):
            pipe(
                question="which",
                context="context",
                max_answer_len=max_answer_len,
            )

        tokenizer.assert_not_called()

    def test_accepts_two_positional_strings_with_future_warning(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        model, input_name = _make_qa_model(tokenizer)

        def answer_target(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_target
        pipe = _create_qa_compat_pipe(tokenizer, model)

        with pytest.warns(FutureWarning, match="Passing a list of SQuAD examples"):
            result = pipe("which", "alpha beta gamma delta epsilon")

        assert result["answer"] == "delta"

    @pytest.mark.parametrize(
        ("question", "context", "message"),
        [
            ("", "context", "`question` cannot be empty"),
            ("question", "", "`context` cannot be empty"),
        ],
    )
    def test_rejects_empty_question_or_context(
        self,
        question: str,
        context: str,
        message: str,
    ) -> None:
        tokenizer = MagicMock()
        pipe = _ExtractiveQuestionAnsweringPipeline(MagicMock(), tokenizer)

        with pytest.raises(ValueError, match=message):
            pipe(question=question, context=context)

        tokenizer.assert_not_called()

    def test_preserves_mapping_and_mapping_batch_inputs(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        model, input_name = _make_qa_model(tokenizer)

        def answer_target(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_target
        pipe = _create_qa_compat_pipe(tokenizer, model)
        item = {
            "question": "which",
            "context": "alpha beta gamma delta epsilon",
        }

        with pytest.warns(FutureWarning) as warning_records:
            single_result = pipe(item)
            batch_result = pipe([item, item])

        assert single_result["answer"] == "delta"
        assert [answer["answer"] for answer in batch_result] == ["delta", "delta"]
        assert len(warning_records) == 2

    def test_preserves_lazy_generator_result_shape_and_order(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        delta_token_id = tokenizer.convert_tokens_to_ids("delta")
        epsilon_token_id = tokenizer.convert_tokens_to_ids("epsilon")
        model, input_name = _make_qa_model(tokenizer)
        yielded_items: list[str] = []

        def answer_context_target(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == delta_token_id] = 10.0
            logits[token_ids == epsilon_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        def qa_inputs():
            for answer in ("delta", "epsilon"):
                yielded_items.append(answer)
                yield {"question": "which", "context": f"alpha {answer}"}

        model.side_effect = answer_context_target
        pipe = _create_qa_compat_pipe(tokenizer, model)

        with pytest.warns(FutureWarning) as warning_records:
            results = pipe(qa_inputs())

        assert len(warning_records) == 1
        assert iter(results) is results
        assert yielded_items == []
        first_result = next(results)
        assert first_result["answer"] == "delta"
        assert yielded_items == ["delta"]
        assert [first_result["answer"], *(item["answer"] for item in results)] == [
            "delta",
            "epsilon",
        ]
        assert yielded_items == ["delta", "epsilon"]

    @pytest.mark.parametrize(
        ("invalid_item", "error_type", "message"),
        [
            ("not-a-mapping", TypeError, "must be a mapping or sequence of mappings"),
            (
                {"question": 1, "context": "alpha delta"},
                TypeError,
                "question and context values must be strings",
            ),
            (
                {"question": "", "context": "alpha delta"},
                ValueError,
                "`question` cannot be empty",
            ),
        ],
    )
    def test_validates_each_iterator_item(
        self,
        invalid_item: Any,
        error_type: type[Exception],
        message: str,
    ) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        model, input_name = _make_qa_model(tokenizer)

        def answer_target(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_target
        pipe = _create_qa_compat_pipe(tokenizer, model)
        valid_item = {"question": "which", "context": "alpha delta"}

        with pytest.warns(FutureWarning):
            results = pipe(iter([valid_item, invalid_item]))

        assert next(results)["answer"] == "delta"
        with pytest.raises(error_type, match=message):
            next(results)

    def test_returns_ranked_top_k_answers_with_compatible_shapes(self) -> None:
        tokenizer = _make_fast_qa_tokenizer(model_max_length=16)
        start_token_id = tokenizer.convert_tokens_to_ids("delta")
        second_end_token_id = tokenizer.convert_tokens_to_ids("epsilon")
        model, input_name = _make_qa_model(tokenizer)

        def rank_answer_spans(**inputs):
            token_ids = inputs[input_name]
            start_logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            end_logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            start_logits[token_ids == start_token_id] = 10.0
            end_logits[token_ids == start_token_id] = 10.0
            end_logits[token_ids == second_end_token_id] = 8.0
            return SimpleNamespace(start_logits=start_logits, end_logits=end_logits)

        model.side_effect = rank_answer_spans
        pipe = _create_qa_compat_pipe(tokenizer, model)
        context = "alpha beta gamma delta epsilon"

        single_result = pipe(question="which", context=context, top_k=1)
        ranked_result = pipe(question="which", context=context, top_k=2)

        assert single_result["answer"] == "delta"
        assert [answer["answer"] for answer in ranked_result] == [
            "delta",
            "delta epsilon",
        ]
        assert ranked_result[0]["score"] > ranked_result[1]["score"]

    def test_ranks_one_impossible_answer_within_top_k(self) -> None:
        tokenizer = _make_fast_qa_tokenizer(model_max_length=16)
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        model, input_name = _make_qa_model(tokenizer)

        def rank_null_first(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == tokenizer.cls_token_id] = 10.0
            logits[token_ids == target_token_id] = 8.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = rank_null_first
        pipe = _create_qa_compat_pipe(tokenizer, model)
        context = "alpha beta gamma delta epsilon"

        answerable_result = pipe(
            question="which",
            context=context,
            top_k=2,
        )
        impossible_result = pipe(
            question="which",
            context=context,
            top_k=2,
            handle_impossible_answer=True,
        )

        assert len(answerable_result) == len(impossible_result) == 2
        assert all(answer["answer"] for answer in answerable_result)
        assert [answer["answer"] for answer in impossible_result] == ["", "delta"]

    def test_returns_ranked_top_k_answers_for_each_batch_item(self) -> None:
        tokenizer = _make_fast_qa_tokenizer(model_max_length=16)
        start_token_id = tokenizer.convert_tokens_to_ids("delta")
        second_end_token_id = tokenizer.convert_tokens_to_ids("epsilon")
        model, input_name = _make_qa_model(tokenizer)

        def rank_answer_spans(**inputs):
            token_ids = inputs[input_name]
            start_logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            end_logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            start_logits[token_ids == start_token_id] = 10.0
            end_logits[token_ids == start_token_id] = 10.0
            end_logits[token_ids == second_end_token_id] = 8.0
            return SimpleNamespace(start_logits=start_logits, end_logits=end_logits)

        model.side_effect = rank_answer_spans
        pipe = _create_qa_compat_pipe(tokenizer, model)
        context = "alpha beta gamma delta epsilon"

        result = pipe(
            question=["which", "which"],
            context=[context, context],
            top_k=2,
        )

        assert [[answer["answer"] for answer in item_answers] for item_answers in result] == [
            ["delta", "delta epsilon"],
            ["delta", "delta epsilon"],
        ]

    def test_deduplicates_same_span_across_overflow_without_summing_scores(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        target_token_id = tokenizer.convert_tokens_to_ids("gamma")
        model, input_name = _make_qa_model(tokenizer)

        def rank_overlapping_target(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = rank_overlapping_target
        pipe = _create_qa_compat_pipe(tokenizer, model)

        result = pipe(
            question="which",
            context="alpha beta gamma delta epsilon",
            top_k=20,
            handle_impossible_answer=True,
        )

        answers = [answer["answer"].lower() for answer in result]
        gamma_candidates = [answer for answer in result if answer["answer"].lower() == "gamma"]
        assert answers[0] == "gamma"
        assert len(gamma_candidates) == 1
        assert gamma_candidates[0]["score"] <= 1.0
        assert answers.count("") == 1

    def test_preserves_identical_answer_text_at_distinct_context_spans(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        target_token_id = tokenizer.convert_tokens_to_ids("gamma")
        model, input_name = _make_qa_model(tokenizer)

        def rank_repeated_target(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = rank_repeated_target
        pipe = _create_qa_compat_pipe(tokenizer, model)
        context = "alpha beta gamma delta gamma epsilon"

        result = pipe(
            question="which",
            context=context,
            top_k=20,
        )

        gamma_candidates = [answer for answer in result if answer["answer"].lower() == "gamma"]
        expected_spans = {
            (context.index("gamma"), context.index("gamma") + len("gamma")),
            (context.rindex("gamma"), context.rindex("gamma") + len("gamma")),
        }
        assert {(answer["start"], answer["end"]) for answer in gamma_candidates} == expected_spans
        assert all(answer["score"] <= 1.0 for answer in gamma_candidates)

    def test_returns_impossible_answer_from_cls_score(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        model, input_name = _make_qa_model(tokenizer)

        def answer_cls(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == tokenizer.cls_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_cls
        pipe = _create_qa_compat_pipe(tokenizer, model)
        no_answer_result = pipe(
            question="which",
            context="alpha beta gamma delta epsilon",
            handle_impossible_answer=True,
        )

        assert no_answer_result["answer"] == ""
        assert no_answer_result["start"] == no_answer_result["end"] == 0

    def test_answer_window_beats_null_scores_from_other_overflow_windows(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        context = "alpha beta gamma delta epsilon"
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        model, input_name = _make_qa_model(tokenizer)

        def answer_target_or_cls(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            if torch.any(token_ids == target_token_id):
                logits[token_ids == target_token_id] = 10.0
            else:
                logits[token_ids == tokenizer.cls_token_id] = 12.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_target_or_cls
        pipe = _create_qa_compat_pipe(tokenizer, model)
        overflow_answer = pipe(
            question="which",
            context=context,
            handle_impossible_answer=True,
        )

        assert overflow_answer["answer"] == "delta"

    def test_non_context_logits_do_not_distort_overflow_span_ranking(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        alpha_token_id = tokenizer.convert_tokens_to_ids("alpha")
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        question_token_id = tokenizer.convert_tokens_to_ids("which")
        model, input_name = _make_qa_model(tokenizer)

        def rank_answer_spans(**inputs):
            token_ids = inputs[input_name]
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            if torch.any(token_ids == target_token_id):
                logits[token_ids == target_token_id] = 6.0
                logits[token_ids == question_token_id] = 10.0
            else:
                logits[token_ids == alpha_token_id] = 4.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = rank_answer_spans
        pipe = _create_qa_compat_pipe(tokenizer, model)

        result = pipe(
            question="which",
            context="alpha beta gamma delta epsilon",
        )

        assert result["answer"] == "delta"

    def test_pads_overflow_features_to_the_models_static_batch(self) -> None:
        tokenizer = _make_fast_qa_tokenizer()
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        model, input_name = _make_qa_model(tokenizer, batch_size=2)
        observed_batch_sizes: list[int] = []

        def answer_static_batch(**inputs):
            token_ids = inputs[input_name]
            observed_batch_sizes.append(token_ids.shape[0])
            if token_ids.shape[0] != 2:
                raise AssertionError("QA features must match the model's static batch")
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_static_batch
        pipe = _create_qa_compat_pipe(tokenizer, model)

        result = pipe(
            question="which",
            context="alpha beta gamma delta epsilon alpha beta",
        )

        assert result["answer"] == "delta"
        assert observed_batch_sizes
        assert set(observed_batch_sizes) == {2}

    def test_places_context_first_for_left_padding_tokenizers(self) -> None:
        tokenizer = _make_fast_qa_tokenizer(padding_side="left")
        target_token_id = tokenizer.convert_tokens_to_ids("delta")
        question_token_id = tokenizer.convert_tokens_to_ids("which")
        context_token_ids = {
            tokenizer.convert_tokens_to_ids(token)
            for token in ("alpha", "beta", "gamma", "delta", "epsilon")
        }
        model, input_name = _make_qa_model(tokenizer)

        def answer_left_padded(**inputs):
            token_ids = inputs[input_name]
            for row in token_ids:
                question_position = int((row == question_token_id).nonzero()[0].item())
                context_positions = [
                    index
                    for index, token_id in enumerate(row.tolist())
                    if token_id in context_token_ids
                ]
                if not context_positions or max(context_positions) >= question_position:
                    raise AssertionError("Left-padded QA inputs must place context before question")
            logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            logits[token_ids == target_token_id] = 10.0
            return SimpleNamespace(start_logits=logits, end_logits=logits)

        model.side_effect = answer_left_padded
        pipe = _create_qa_compat_pipe(tokenizer, model)

        result = pipe(
            question="which",
            context="alpha beta gamma delta epsilon",
        )

        assert result["answer"] == "delta"

    def test_preserves_requested_stride_for_boundary_spanning_answers(self) -> None:
        tokenizer = _make_fast_qa_tokenizer(model_max_length=10)
        start_token_id = tokenizer.convert_tokens_to_ids("beta")
        end_token_id = tokenizer.convert_tokens_to_ids("delta")
        model, input_name = _make_qa_model(tokenizer)

        def answer_boundary_span(**inputs):
            token_ids = inputs[input_name]
            start_logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            end_logits = torch.zeros(token_ids.shape, dtype=torch.float32)
            start_logits[token_ids == start_token_id] = 10.0
            end_logits[token_ids == end_token_id] = 10.0
            return SimpleNamespace(
                start_logits=start_logits,
                end_logits=end_logits,
            )

        model.side_effect = answer_boundary_span
        pipe = _create_qa_compat_pipe(tokenizer, model)
        context = "alpha alpha alpha beta gamma gamma delta epsilon epsilon epsilon"

        result = pipe(
            question="which",
            context=context,
            doc_stride=4,
            max_answer_len=4,
        )

        assert result["answer"] == "beta gamma gamma delta"


# ---------------------------------------------------------------------------
# _detect_tokenizer_dict_param
# ---------------------------------------------------------------------------


def _make_pipe_with_preprocess(preprocess_fn: Any) -> MagicMock:
    """Create a mock pipeline whose type has the given preprocess method."""
    pipe = MagicMock()
    # Set the type's preprocess method
    pipe_type = type(pipe)
    pipe_type.preprocess = preprocess_fn
    return pipe


class TestDetectTokenizerDictParam:
    def test_named_param_tokenizer_kwargs(self) -> None:
        """FillMask-style: preprocess(self, inputs, tokenizer_kwargs=None)."""

        def preprocess(self, inputs, tokenizer_kwargs=None, **kwargs):
            pass

        pipe = _make_pipe_with_preprocess(preprocess)
        sig = inspect.signature(preprocess)
        result = _detect_tokenizer_dict_param(pipe, sig.parameters)
        assert result == "tokenizer_kwargs"

    def test_no_tokenizer_param(self) -> None:
        """Simple pipeline with no tokenizer dict param."""

        def preprocess(self, inputs, **kwargs):
            pass

        pipe = _make_pipe_with_preprocess(preprocess)
        sig = inspect.signature(preprocess)
        result = _detect_tokenizer_dict_param(pipe, sig.parameters)
        assert result is None


# ---------------------------------------------------------------------------
# _adapt_tokenizer_padding
# ---------------------------------------------------------------------------


def _make_model_with_shapes(shapes: list[list[int]]) -> MagicMock:
    model = MagicMock()
    model.io_config = {"input_shapes": shapes}
    return model


def _make_tokenizer_pipe(preprocess_fn: Any) -> MagicMock:
    """Build a mock pipeline with tokenizer and preprocess."""
    pipe = MagicMock()
    pipe._preprocess_params = {}
    pipe.tokenizer = MagicMock()
    pipe.tokenizer.model_max_length = 512
    pipe_type = type(pipe)
    pipe_type.preprocess = preprocess_fn
    return pipe


class TestAdaptTokenizerPadding:
    def test_pattern_a_varkw_sets_top_level(self) -> None:
        """Pattern A: **kwargs forwarded → top-level padding/max_length."""

        def preprocess(self, inputs, **kwargs):
            pass

        pipe = _make_tokenizer_pipe(preprocess)
        model = _make_model_with_shapes([[1, 128]])
        _adapt_tokenizer_padding(pipe, "text-classification", model)

        assert pipe._preprocess_params["padding"] == "max_length"
        assert pipe._preprocess_params["max_length"] == 128
        assert pipe._preprocess_params["truncation"] is True
        assert pipe.tokenizer.model_max_length == 128

    def test_pattern_b_tokenizer_kwargs(self) -> None:
        """Pattern B: named tokenizer_kwargs param → nested dict."""

        def preprocess(self, inputs, tokenizer_kwargs=None, **kwargs):
            pass

        pipe = _make_tokenizer_pipe(preprocess)
        model = _make_model_with_shapes([[1, 64]])
        _adapt_tokenizer_padding(pipe, "fill-mask", model)

        tok = pipe._preprocess_params["tokenizer_kwargs"]
        assert tok["padding"] == "max_length"
        assert tok["max_length"] == 64

    def test_pattern_c_explicit_params_only(self) -> None:
        """Pattern C: no **kwargs, only explicit named params."""

        def preprocess(self, inputs, max_seq_len=None, padding=None):
            pass

        pipe = _make_tokenizer_pipe(preprocess)
        model = _make_model_with_shapes([[1, 256]])
        _adapt_tokenizer_padding(pipe, "question-answering", model)

        assert pipe._preprocess_params.get("max_seq_len") == 256
        assert pipe._preprocess_params.get("padding") == "max_length"

    def test_multi_modal_finds_2d_shape(self) -> None:
        """Multi-modal models: should find the 2-D text shape among 4-D image shapes."""

        def preprocess(self, inputs, **kwargs):
            pass

        pipe = _make_tokenizer_pipe(preprocess)
        # First shape is 4-D (image), second is 2-D (text)
        model = _make_model_with_shapes([[1, 3, 224, 224], [1, 77]])
        _adapt_tokenizer_padding(pipe, "clip", model)

        assert pipe._preprocess_params["max_length"] == 77

    def test_no_2d_shape_skips(self) -> None:
        """No 2-D shape → no tokenizer adaptation."""

        def preprocess(self, inputs, **kwargs):
            pass

        pipe = _make_tokenizer_pipe(preprocess)
        model = _make_model_with_shapes([[1, 3, 224, 224]])
        pipe._preprocess_params.clear()
        _adapt_tokenizer_padding(pipe, "image-classification", model)

        assert "max_length" not in pipe._preprocess_params

    def test_no_tokenizer_skips(self) -> None:
        """Pipeline with tokenizer=None should return early."""
        pipe = MagicMock()
        pipe.tokenizer = None
        model = _make_model_with_shapes([[1, 128]])
        pipe._preprocess_params = {}
        _adapt_tokenizer_padding(pipe, "text-classification", model)
        # No params should be set
        assert "max_length" not in pipe._preprocess_params


# ---------------------------------------------------------------------------
# _adapt_image_processor_size
# ---------------------------------------------------------------------------


class TestAdaptImageProcessorSize:
    def test_height_width_format(self) -> None:
        """Standard processors use {"height": h, "width": w}."""
        pipe = MagicMock()
        pipe.image_processor.size = {"height": 224, "width": 224}
        pipe.image_processor.do_pad = True
        model = _make_model_with_shapes([[1, 3, 384, 384]])
        _adapt_image_processor_size(pipe, "image-classification", model)

        assert pipe.image_processor.size == {"height": 384, "width": 384}
        assert pipe.image_processor.do_pad is False

    def test_shortest_edge_format(self) -> None:
        """ConvNeXt-style processors use {"shortest_edge": N}."""
        pipe = MagicMock()
        pipe.image_processor.size = {"shortest_edge": 224}
        model = _make_model_with_shapes([[1, 3, 384, 384]])
        _adapt_image_processor_size(pipe, "image-classification", model)

        assert pipe.image_processor.size == {"shortest_edge": 384}

    def test_multi_modal_finds_4d_shape(self) -> None:
        """Multi-modal: should find the 4-D image shape among 2-D text shapes."""
        pipe = MagicMock()
        pipe.image_processor.size = {"height": 224, "width": 224}
        # First shape is 2-D (text), second is 4-D (image)
        model = _make_model_with_shapes([[1, 77], [1, 3, 336, 336]])
        _adapt_image_processor_size(pipe, "clip", model)

        assert pipe.image_processor.size == {"height": 336, "width": 336}

    def test_no_4d_shape_skips(self) -> None:
        """No 4-D shape → no image processor adaptation."""
        pipe = MagicMock()
        pipe.image_processor.size = {"height": 224, "width": 224}
        model = _make_model_with_shapes([[1, 128]])
        _adapt_image_processor_size(pipe, "text-classification", model)

        # Size should be unchanged
        assert pipe.image_processor.size == {"height": 224, "width": 224}

    def test_no_image_processor_skips(self) -> None:
        """Pipeline without image_processor should be skipped."""
        pipe = MagicMock(spec=[])  # no 'image_processor' attribute
        model = _make_model_with_shapes([[1, 3, 224, 224]])
        _adapt_image_processor_size(pipe, "image-classification", model)
