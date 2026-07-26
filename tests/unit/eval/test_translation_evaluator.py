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


def make_evaluator(rows, columns_mapping=None):
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    dataset = Dataset.from_list(rows)
    model = MagicMock()
    model.config.label2id = None
    model.max_encoder_length = 512
    model.max_decode_length = 512
    config = WinMLEvaluationConfig(
        model_id="Helsinki-NLP/opus-mt-fr-en",
        task="translation",
        dataset=DatasetConfig(
            path="wmt14",
            samples=len(rows),
            shuffle=False,
            columns_mapping=columns_mapping or {},
        ),
    )
    with (
        patch("datasets.load_dataset", return_value=dataset),
        patch(
            "winml.modelkit.eval.base_evaluator.WinMLEvaluator.prepare_pipeline",
            return_value=MagicMock(),
        ),
    ):
        return WinMLTranslationEvaluator(config, model)


class TestTranslationMetric:
    def test_perfect_corpus_scores_100(self):
        from winml.modelkit.eval.metrics.translation import TranslationMetric

        metric = TranslationMetric()
        metric.update("This is a complete test sentence.", "This is a complete test sentence.")
        result = metric.compute()

        assert result == {"sacrebleu": 100.0, "chrf": 100.0, "n_samples": 1}

    def test_empty_corpus_has_no_scores(self):
        from winml.modelkit.eval.metrics.translation import TranslationMetric

        assert TranslationMetric().compute() == {
            "sacrebleu": None,
            "chrf": None,
            "n_samples": 0,
        }

    def test_multiple_references_are_supported(self):
        from winml.modelkit.eval.metrics.translation import TranslationMetric

        metric = TranslationMetric()
        metric.update(
            "A sufficiently long reference sentence is here.",
            [
                "Another reference sentence is also supplied.",
                "A sufficiently long reference sentence is here.",
            ],
        )
        assert metric.compute()["sacrebleu"] == 100.0

    def test_multiple_perfect_samples_have_perfect_corpus_scores(self):
        from winml.modelkit.eval.metrics.translation import TranslationMetric

        metric = TranslationMetric()
        first = "The first complete sentence is correct."
        second = "The second complete sentence is correct."
        metric.update(first, first)
        metric.update(second, second)
        result = metric.compute()

        assert result["sacrebleu"] == 100.0
        assert result["chrf"] == 100.0

    def test_empty_references_are_rejected(self):
        from winml.modelkit.eval.metrics.translation import TranslationMetric

        with pytest.raises(ValueError, match="at least one non-empty"):
            TranslationMetric().update("prediction", [])


class TestTranslationEvaluator:
    def test_nested_translation_dict_uses_explicit_language_keys(self):
        evaluator = make_evaluator(
            [{"translation": {"fr": "Bonjour le monde", "en": "Hello world"}}],
            {"source_lang": "fr", "target_lang": "en"},
        )
        assert evaluator.pipe.tokenizer.model_max_length == 512
        assert evaluator.pipe.generation_config.max_new_tokens is None
        assert evaluator.pipe.generation_config.num_beams == 1
        evaluator.pipe = MagicMock(return_value=[{"translation_text": "Hello world"}])

        result = evaluator.compute()

        evaluator.pipe.assert_called_once_with(
            "Bonjour le monde",
            num_beams=1,
            truncation=True,
            src_lang="fr",
            tgt_lang="en",
            max_new_tokens=511,
        )
        assert result["n_samples"] == 1
        assert result["chrf"] == 100.0
        assert result["attempted"] == 1
        assert result["evaluated"] == 1
        assert result["skipped"] == 0

    def test_flat_custom_columns_need_no_language_keys(self):
        evaluator = make_evaluator(
            [{"french": "Une phrase source", "english": "A source sentence"}],
            {"source_column": "french", "reference_column": "english"},
        )
        evaluator.pipe = MagicMock(return_value={"generated_text": "A source sentence"})

        result = evaluator.compute()

        assert result["n_samples"] == 1
        assert result["chrf"] == 100.0

    def test_missing_language_key_rejects_row_with_accounting(self):
        evaluator = make_evaluator(
            [
                {"translation": {"fr": "Valide", "en": "Valid"}},
                {"translation": {"fr": "Invalide", "de": "Ungültig"}},
            ],
            {"source_lang": "fr", "target_lang": "en"},
        )
        evaluator.pipe = MagicMock(return_value=[{"translation_text": "Valid"}])

        result = evaluator.compute()

        assert result["n_samples"] == 1
        assert result["attempted"] == 2
        assert result["evaluated"] == 1
        assert result["skipped"] == 1
        evaluator.pipe.assert_called_once()

    def test_nested_translation_requires_explicit_direction(self):
        evaluator = make_evaluator(
            [{"translation": {"fr": "Bonjour", "en": "Hello"}}],
        )

        with pytest.raises(DatasetValidationError, match="provide --column source_lang"):
            evaluator.compute()

    def test_pipeline_runtime_failure_propagates(self):
        evaluator = make_evaluator(
            [
                {"source": "un", "reference": "one"},
                {"source": "deux", "reference": "two"},
            ],
            {"source_column": "source", "reference_column": "reference"},
        )
        evaluator.pipe = MagicMock(
            side_effect=[
                [{"translation_text": "one"}],
                RuntimeError("decoder failed"),
            ]
        )

        with pytest.raises(RuntimeError, match="decoder failed"):
            evaluator.compute()

    def test_all_rejected_fails_closed(self):
        evaluator = make_evaluator(
            [{"source": "texte", "reference": "text"}],
            {"source_column": "source", "reference_column": "reference"},
        )
        evaluator.pipe = MagicMock(return_value=[{"translation_text": ""}])

        with pytest.raises(DatasetValidationError, match="No valid translation samples"):
            evaluator.compute()

    @pytest.mark.parametrize("output", [[], [{}], [{"translation_text": ""}]])
    def test_empty_pipeline_output_is_rejected(self, output):
        evaluator = make_evaluator(
            [{"source": "texte", "reference": "text"}],
            {"source_column": "source", "reference_column": "reference"},
        )
        evaluator.pipe = MagicMock(return_value=output)

        with pytest.raises(DatasetValidationError, match="No valid translation samples"):
            evaluator.compute()

    def test_empty_reference_list_is_rejected(self):
        evaluator = make_evaluator(
            [{"source": "texte", "references": []}],
            {"source_column": "source", "reference_column": "references"},
        )

        with pytest.raises(DatasetValidationError, match="No valid translation samples"):
            evaluator.compute()

    def test_non_mapping_row_is_rejected_with_accounting(self):
        evaluator = make_evaluator(
            [{"source": "valide", "reference": "valid"}],
            {"source_column": "source", "reference_column": "reference"},
        )
        evaluator.data = [None, {"source": "valide", "reference": "valid"}]
        evaluator.pipe = MagicMock(return_value=[{"translation_text": "valid"}])

        result = evaluator.compute()

        assert result["attempted"] == 2
        assert result["evaluated"] == 1
        assert result["skipped"] == 1

    def test_tokenizer_language_ids_and_source_prefix_are_independent_of_dataset_keys(self):
        evaluator = make_evaluator(
            [{"translation": {"fra_Latn": "Bonjour", "eng_Latn": "Hello"}}],
            {
                "source_lang": "fra_Latn",
                "target_lang": "eng_Latn",
                "tokenizer_source_lang": "fra_Latn",
                "tokenizer_target_lang": "eng_Latn",
                "source_prefix": "translate French to English: ",
            },
        )
        evaluator.pipe = MagicMock(return_value=[{"translation_text": "Hello"}])

        evaluator.compute()

        evaluator.pipe.assert_called_once_with(
            "translate French to English: Bonjour",
            num_beams=1,
            truncation=True,
            src_lang="fra_Latn",
            tgt_lang="eng_Latn",
            max_new_tokens=511,
        )


class TestTranslationRegistration:
    def test_registry_and_schema_are_present(self):
        from winml.modelkit.eval import WinMLEvaluationConfig, get_evaluator_class
        from winml.modelkit.utils.eval_utils import TASK_SCHEMAS

        assert get_evaluator_class(WinMLEvaluationConfig(task="translation")) is (
            WinMLTranslationEvaluator
        )
        assert TASK_SCHEMAS["translation"].roles == ("encoder", "decoder")
