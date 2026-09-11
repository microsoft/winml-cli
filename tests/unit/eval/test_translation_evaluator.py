# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for translation evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from winml.modelkit.eval.translation_evaluator import WinMLTranslationEvaluator
from winml.modelkit.utils.eval_utils import DatasetValidationError


def make_evaluator(rows, columns_mapping=None, pipeline=None):
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    dataset = Dataset.from_list(rows)
    model = MagicMock()
    model.config.label2id = None
    model.max_encoder_length = 512
    model.max_decode_length = 512
    config = WinMLEvaluationConfig(
        model_id="example/translation-model",
        task="translation",
        dataset=DatasetConfig(
            path="example/parallel-corpus",
            samples=len(rows),
            shuffle=False,
            columns_mapping=columns_mapping or {},
        ),
    )
    with (
        patch("datasets.load_dataset", return_value=dataset),
        patch(
            "winml.modelkit.eval.base_evaluator.WinMLEvaluator.prepare_pipeline",
            return_value=pipeline or MagicMock(),
        ),
    ):
        return WinMLTranslationEvaluator(config, model)


class TestTranslationMetric:
    def test_perfect_corpus_scores_100(self):
        from winml.modelkit.eval.metrics.translation import TranslationMetric

        metric = TranslationMetric()
        sentence = "This is a complete test sentence."
        metric.update(sentence, sentence)

        assert metric.compute() == {
            "sacrebleu_13a_0_100": 100.0,
            "chrf2_0_100": 100.0,
            "n_samples": 1,
        }

    def test_empty_corpus_has_no_scores(self):
        from winml.modelkit.eval.metrics.translation import TranslationMetric

        assert TranslationMetric().compute() == {
            "sacrebleu_13a_0_100": None,
            "chrf2_0_100": None,
            "n_samples": 0,
        }

    def test_multiple_references_use_corpus_metrics(self):
        from winml.modelkit.eval.metrics.translation import TranslationMetric

        metric = TranslationMetric()
        prediction = "A sufficiently long reference sentence is here."
        metric.update(prediction, ["Another complete reference is here.", prediction])

        assert metric.compute()["sacrebleu_13a_0_100"] == 100.0

    def test_empty_references_are_rejected(self):
        from winml.modelkit.eval.metrics.translation import TranslationMetric

        with pytest.raises(ValueError, match="at least one non-empty"):
            TranslationMetric().update("prediction", [])


class TestTranslationEvaluator:
    def test_nested_translation_uses_explicit_direction_and_bounded_generation(self):
        evaluator = make_evaluator(
            [{"translation": {"source": "Bonjour le monde", "target": "Hello world"}}],
            {"source_lang": "source", "target_lang": "target"},
        )
        assert evaluator.pipe.tokenizer.model_max_length == 128
        assert evaluator.pipe.generation_config.max_new_tokens is None
        evaluator.pipe = MagicMock(return_value=[{"translation_text": "Hello world"}])

        result = evaluator.compute()

        evaluator.pipe.assert_called_once_with(
            "Bonjour le monde",
            do_sample=False,
            max_new_tokens=64,
            num_beams=1,
            num_return_sequences=1,
            truncation=True,
            src_lang="source",
            tgt_lang="target",
        )
        assert result["chrf2_0_100"] == 100.0
        assert result["attempted"] == result["evaluated"] == 1
        assert result["skipped"] == 0

    def test_explicit_caps_clamp_to_static_model_capacities(self):
        pipeline = MagicMock()
        evaluator = make_evaluator(
            [{"source": "texte", "reference": "text"}],
            {
                "source_column": "source",
                "reference_column": "reference",
                "max_source_tokens": "1024",
                "max_new_tokens": "1024",
            },
            pipeline,
        )
        evaluator.pipe = MagicMock(return_value={"generated_text": "text"})

        evaluator.compute()

        assert pipeline.tokenizer.model_max_length == 512
        assert evaluator.pipe.call_args.kwargs["max_new_tokens"] == 511

    @pytest.mark.parametrize(
        ("name", "value"),
        [("max_source_tokens", "0"), ("max_new_tokens", "bad")],
    )
    def test_invalid_token_bounds_are_rejected(self, name, value):
        with pytest.raises(DatasetValidationError, match=f"{name} must be a positive integer"):
            make_evaluator(
                [{"source": "texte", "reference": "text"}],
                {"source_column": "source", "reference_column": "reference", name: value},
            )

    @pytest.mark.parametrize("name", ["num_beams", "num_return_sequences"])
    def test_static_batch_contract_rejects_generation_fanout(self, name):
        with pytest.raises(DatasetValidationError, match="static batch-one"):
            make_evaluator(
                [{"source": "texte", "reference": "text"}],
                {"source_column": "source", "reference_column": "reference", name: "2"},
            )

    def test_pipeline_without_tokenizer_is_supported(self):
        pipeline = MagicMock()
        pipeline.tokenizer = None

        evaluator = make_evaluator(
            [{"source": "texte", "reference": "text"}],
            {"source_column": "source", "reference_column": "reference"},
            pipeline,
        )

        assert evaluator.pipe.tokenizer is None

    def test_flat_columns_and_multiple_references(self):
        evaluator = make_evaluator(
            [{"source": "Une phrase source", "references": ["A source sentence", "A sentence"]}],
            {"source_column": "source", "reference_column": "references"},
        )
        evaluator.pipe = MagicMock(return_value={"generated_text": "A source sentence"})

        assert evaluator.compute()["n_samples"] == 1

    def test_missing_language_key_is_skipped_with_exact_accounting(self):
        evaluator = make_evaluator(
            [
                {"translation": {"source": "Valide", "target": "Valid"}},
                {"translation": {"source": "Invalide", "other": "Invalid"}},
            ],
            {"source_lang": "source", "target_lang": "target"},
        )
        evaluator.pipe = MagicMock(return_value=[{"translation_text": "Valid"}])

        result = evaluator.compute()

        assert result["attempted"] == 2
        assert result["evaluated"] == 1
        assert result["skipped"] == 1
        evaluator.pipe.assert_called_once()

    def test_nested_translation_requires_explicit_direction(self):
        evaluator = make_evaluator(
            [{"translation": {"source": "Bonjour", "target": "Hello"}}]
        )

        with pytest.raises(DatasetValidationError, match="provide --column source_lang"):
            evaluator.compute()

    def test_tokenizer_languages_and_prefix_are_independent_of_dataset_keys(self):
        evaluator = make_evaluator(
            [{"translation": {"dataset_source": "Bonjour", "dataset_target": "Hello"}}],
            {
                "source_lang": "dataset_source",
                "target_lang": "dataset_target",
                "tokenizer_source_lang": "tokenizer_source",
                "tokenizer_target_lang": "tokenizer_target",
                "source_prefix": "translate: ",
            },
        )
        evaluator.pipe = MagicMock(return_value=[{"translation_text": "Hello"}])

        evaluator.compute()

        assert evaluator.pipe.call_args.args == ("translate: Bonjour",)
        assert evaluator.pipe.call_args.kwargs["src_lang"] == "tokenizer_source"
        assert evaluator.pipe.call_args.kwargs["tgt_lang"] == "tokenizer_target"

    def test_runtime_failure_propagates(self):
        evaluator = make_evaluator(
            [{"source": "un", "reference": "one"}],
            {"source_column": "source", "reference_column": "reference"},
        )
        evaluator.pipe = MagicMock(side_effect=RuntimeError("decoder failed"))

        with pytest.raises(RuntimeError, match="decoder failed"):
            evaluator.compute()

    @pytest.mark.parametrize("output", [[], [{}], [{"translation_text": ""}]])
    def test_invalid_output_fails_closed_when_no_rows_are_evaluated(self, output):
        evaluator = make_evaluator(
            [{"source": "texte", "reference": "text"}],
            {"source_column": "source", "reference_column": "reference"},
        )
        evaluator.pipe = MagicMock(return_value=output)

        with pytest.raises(DatasetValidationError, match="No valid translation samples"):
            evaluator.compute()


class TestTranslationRegistration:
    def test_registry_and_schema_are_present(self):
        from winml.modelkit.eval import WinMLEvaluationConfig, get_evaluator_class
        from winml.modelkit.utils.eval_utils import TASK_SCHEMAS

        assert get_evaluator_class(WinMLEvaluationConfig(task="translation")) is (
            WinMLTranslationEvaluator
        )
        assert TASK_SCHEMAS["translation"].roles == ("encoder", "decoder")
