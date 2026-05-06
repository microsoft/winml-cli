# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLZeroShotClassificationEvaluator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from datasets import ClassLabel, Dataset, Features, Value

from winml.modelkit.eval import (
    WinMLEvaluationConfig,
    WinMLZeroShotClassificationEvaluator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CANDIDATE_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def _make_classlabel_dataset(
    texts: list[str],
    labels: list[int],
    class_names: list[str] | None = None,
    input_col: str = "text",
    label_col: str = "label",
) -> Dataset:
    names = class_names or CANDIDATE_LABELS
    features = Features(
        {
            input_col: Value("string"),
            label_col: ClassLabel(names=names),
        }
    )
    return Dataset.from_dict({input_col: texts, label_col: labels}, features=features)


def _make_string_dataset(
    texts: list[str],
    labels: list[str],
    input_col: str = "text",
    label_col: str = "label",
) -> Dataset:
    features = Features(
        {
            input_col: Value("string"),
            label_col: Value("string"),
        }
    )
    return Dataset.from_dict({input_col: texts, label_col: labels}, features=features)


def make_evaluator(
    dataset: Dataset,
    columns_mapping: dict[str, str] | None = None,
    pipe: MagicMock | None = None,
) -> WinMLZeroShotClassificationEvaluator:
    """Construct an evaluator without going through HF loading."""
    from winml.modelkit.datasets import DatasetConfig

    mapping = columns_mapping or {"input_column": "text", "label_column": "label"}

    if pipe is None:
        pipe = MagicMock()
        pipe.tokenizer = None

    model = MagicMock()
    model.config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    model.io_config = {}

    config = WinMLEvaluationConfig(
        model_id="test/model",
        task="zero-shot-classification",
        dataset=DatasetConfig(
            path="dummy",
            columns_mapping=mapping,
            samples=len(dataset),
            shuffle=False,
        ),
    )

    with (
        patch.object(
            WinMLZeroShotClassificationEvaluator,
            "prepare_data",
            return_value=dataset,
        ),
        patch.object(
            WinMLZeroShotClassificationEvaluator,
            "prepare_pipeline",
            return_value=pipe,
        ),
    ):
        return WinMLZeroShotClassificationEvaluator(config, model)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_evaluator_registered(self) -> None:
        from winml.modelkit.eval.evaluate import _EVALUATOR_REGISTRY

        assert "zero-shot-classification" in _EVALUATOR_REGISTRY
        assert (
            _EVALUATOR_REGISTRY["zero-shot-classification"] is WinMLZeroShotClassificationEvaluator
        )

    def test_default_dataset_registered(self) -> None:
        from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS

        cfg = _DEFAULT_DATASETS["zero-shot-classification"]
        assert cfg.path is not None
        assert cfg.columns_mapping.get("input_column") is not None
        assert cfg.columns_mapping.get("label_column") is not None

    def test_exported_from_package(self) -> None:
        from winml.modelkit import eval as eval_pkg

        assert "WinMLZeroShotClassificationEvaluator" in eval_pkg.__all__


# ---------------------------------------------------------------------------
# _FixedShapeZeroShotPipeline
# ---------------------------------------------------------------------------


class TestFixedShapePipeline:
    """The subclass delegates resizing to the evaluator's pad/truncate helper."""

    def test_calls_evaluator_pad_or_truncate(self) -> None:
        from transformers.pipelines.zero_shot_classification import (
            ZeroShotClassificationPipeline,
        )

        from winml.modelkit.eval.zero_shot_classification_evaluator import (
            _FixedShapeZeroShotPipeline,
        )

        captured: dict = {}
        sentinel_encoding = {"input_ids": MagicMock()}
        padded = {"input_ids": MagicMock()}

        def _fake_super_parse(self, sequence_pairs, **kwargs):
            captured.update(kwargs)
            return sentinel_encoding

        evaluator = MagicMock()
        evaluator._pad_or_truncate.return_value = padded

        instance = _FixedShapeZeroShotPipeline.__new__(_FixedShapeZeroShotPipeline)
        instance._winml_evaluator = evaluator
        instance.tokenizer = MagicMock()
        with patch.object(ZeroShotClassificationPipeline, "_parse_and_tokenize", _fake_super_parse):
            result = instance._parse_and_tokenize([("a", "b")])

        assert captured["truncation"] is True
        assert captured["padding"] is True
        evaluator._pad_or_truncate.assert_called_once_with(sentinel_encoding, instance.tokenizer)
        assert result is padded

    def test_passthrough_when_evaluator_unset(self) -> None:
        from transformers.pipelines.zero_shot_classification import (
            ZeroShotClassificationPipeline,
        )

        from winml.modelkit.eval.zero_shot_classification_evaluator import (
            _FixedShapeZeroShotPipeline,
        )

        sentinel_encoding = {"input_ids": MagicMock()}

        def _fake_super_parse(self, sequence_pairs, **kwargs):
            return sentinel_encoding

        instance = _FixedShapeZeroShotPipeline.__new__(_FixedShapeZeroShotPipeline)
        instance._winml_evaluator = None
        instance.tokenizer = MagicMock()
        with patch.object(ZeroShotClassificationPipeline, "_parse_and_tokenize", _fake_super_parse):
            result = instance._parse_and_tokenize([("a", "b")])

        assert result is sentinel_encoding


# ---------------------------------------------------------------------------
# prepare_pipeline
# ---------------------------------------------------------------------------


def _make_mock_pipe_with_tokenizer(model_input_names: list[str] | None = None) -> MagicMock:
    pipe = MagicMock()
    pipe.tokenizer = MagicMock()
    pipe.tokenizer.model_max_length = 0
    pipe.tokenizer.model_input_names = model_input_names or [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]
    return pipe


class TestPreparePipeline:
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_sets_model_max_length_from_io_config(self, mock_load_ds, mock_pipeline) -> None:
        from winml.modelkit.datasets import DatasetConfig

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 2
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_ds.column_names = ["text", "label"]
        mock_load_ds.return_value = mock_ds

        mock_pipe = _make_mock_pipe_with_tokenizer()
        mock_pipeline.return_value = mock_pipe

        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [[1, 256]]}

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="zero-shot-classification",
            dataset=DatasetConfig(path="dummy"),
        )

        with patch.object(
            WinMLZeroShotClassificationEvaluator,
            "align_labels",
            side_effect=lambda dataset, ds_config: dataset,
        ):
            WinMLZeroShotClassificationEvaluator(config, model)

        assert mock_pipe.tokenizer.model_max_length == 256

    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_filters_tokenizer_input_names(self, mock_load_ds, mock_pipeline) -> None:
        from winml.modelkit.datasets import DatasetConfig

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 2
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_ds.column_names = ["text", "label"]
        mock_load_ds.return_value = mock_ds

        mock_pipe = _make_mock_pipe_with_tokenizer(
            model_input_names=["input_ids", "attention_mask", "token_type_ids"],
        )
        mock_pipeline.return_value = mock_pipe

        model = MagicMock()
        model.config.label2id = None
        model.io_config = {
            "input_shapes": [[1, 128]],
            "input_names": ["input_ids", "attention_mask"],
        }

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="zero-shot-classification",
            dataset=DatasetConfig(path="dummy"),
        )

        with patch.object(
            WinMLZeroShotClassificationEvaluator,
            "align_labels",
            side_effect=lambda dataset, ds_config: dataset,
        ):
            WinMLZeroShotClassificationEvaluator(config, model)

        assert mock_pipe.tokenizer.model_input_names == ["input_ids", "attention_mask"]

    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_no_tokenizer_change_without_io_config(self, mock_load_ds, mock_pipeline) -> None:
        from winml.modelkit.datasets import DatasetConfig

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 2
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_ds.column_names = ["text", "label"]
        mock_load_ds.return_value = mock_ds

        mock_pipe = _make_mock_pipe_with_tokenizer()
        original_names = list(mock_pipe.tokenizer.model_input_names)
        mock_pipeline.return_value = mock_pipe

        model = MagicMock()
        model.config.label2id = None
        model.io_config = {}

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="zero-shot-classification",
            dataset=DatasetConfig(path="dummy"),
        )

        with patch.object(
            WinMLZeroShotClassificationEvaluator,
            "align_labels",
            side_effect=lambda dataset, ds_config: dataset,
        ):
            WinMLZeroShotClassificationEvaluator(config, model)

        # model_max_length untouched and input_names unchanged.
        assert mock_pipe.tokenizer.model_max_length == 0
        assert mock_pipe.tokenizer.model_input_names == original_names


# ---------------------------------------------------------------------------
# schema_info
# ---------------------------------------------------------------------------


class TestSchemaInfo:
    def test_schema_has_input_and_label(self) -> None:
        cols = WinMLZeroShotClassificationEvaluator.schema_info()
        overrides = {c.override for c in cols if c.override}
        assert "input_column" in overrides
        assert "label_column" in overrides

    def test_schema_has_optional_overrides(self) -> None:
        cols = WinMLZeroShotClassificationEvaluator.schema_info()
        override_to_required = {c.override: c.required for c in cols if c.override}
        assert override_to_required.get("candidate_labels") is False
        assert override_to_required.get("hypothesis_template") is False


# ---------------------------------------------------------------------------
# align_labels / schema validation
# ---------------------------------------------------------------------------


class TestAlignLabels:
    def test_valid_dataset_passes(self) -> None:
        ds = _make_classlabel_dataset(["a", "b"], [0, 1])
        ev = make_evaluator(ds)
        out = ev.align_labels(ds, ev.config.dataset)
        assert out is ds

    def test_missing_input_column_raises(self) -> None:
        ds = _make_classlabel_dataset(["a"], [0])
        ev = make_evaluator(
            ds,
            columns_mapping={"input_column": "nope", "label_column": "label"},
        )
        with pytest.raises(ValueError, match="Column 'nope'"):
            ev.align_labels(ds, ev.config.dataset)

    def test_missing_label_column_raises(self) -> None:
        ds = _make_classlabel_dataset(["a"], [0])
        ev = make_evaluator(
            ds,
            columns_mapping={"input_column": "text", "label_column": "missing"},
        )
        with pytest.raises(ValueError, match="Column 'missing'"):
            ev.align_labels(ds, ev.config.dataset)

    def test_no_alignment_against_nli_label2id(self) -> None:
        """Regression: base-class alignment must not be applied."""
        ds = _make_classlabel_dataset(["a", "b"], [0, 1])
        ev = make_evaluator(ds)
        out = ev.align_labels(ds, ev.config.dataset)
        # Labels unchanged — still 0 and 1 (not remapped to NLI ids)
        assert out[0]["label"] == 0
        assert out[1]["label"] == 1


# ---------------------------------------------------------------------------
# _resolve_candidate_labels
# ---------------------------------------------------------------------------


class TestResolveCandidateLabels:
    def test_user_override_comma_separated(self) -> None:
        ds = _make_classlabel_dataset(["a"], [0])
        ev = make_evaluator(
            ds,
            columns_mapping={
                "input_column": "text",
                "label_column": "label",
                "candidate_labels": "politics, sports ,tech",
            },
        )
        labels = ev._resolve_candidate_labels(ds)
        assert labels == ["politics", "sports", "tech"]

    def test_auto_from_classlabel(self) -> None:
        ds = _make_classlabel_dataset(["a"], [0])
        ev = make_evaluator(ds)
        labels = ev._resolve_candidate_labels(ds)
        assert labels == CANDIDATE_LABELS

    def test_string_label_without_override_raises(self) -> None:
        ds = _make_string_dataset(["a"], ["World"])
        ev = make_evaluator(ds)
        with pytest.raises(ValueError, match="not a ClassLabel"):
            ev._resolve_candidate_labels(ds)

    def test_empty_override_raises(self) -> None:
        ds = _make_classlabel_dataset(["a"], [0])
        ev = make_evaluator(
            ds,
            columns_mapping={
                "input_column": "text",
                "label_column": "label",
                "candidate_labels": ", , ",
            },
        )
        with pytest.raises(ValueError, match="empty"):
            ev._resolve_candidate_labels(ds)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------


def _pipe_returning(predictions: list[str]) -> MagicMock:
    """Build a MagicMock pipeline that emits predictions in order."""
    pipe = MagicMock()
    pipe.tokenizer = None
    state = {"i": 0}

    def _call(text: str, candidate_labels: list[str], **kwargs: object):
        idx = state["i"]
        state["i"] += 1
        top = predictions[idx]
        ordered = [top] + [c for c in candidate_labels if c != top]
        scores = [0.9] + [0.1 / max(1, len(ordered) - 1)] * (len(ordered) - 1)
        return {"sequence": text, "labels": ordered, "scores": scores}

    pipe.side_effect = _call
    return pipe


class TestCompute:
    def test_perfect_accuracy_and_f1(self) -> None:
        ds = _make_classlabel_dataset(
            ["a", "b", "c", "d"],
            [0, 1, 2, 3],
        )
        # Predictions exactly match gold labels.
        pipe = _pipe_returning(["World", "Sports", "Business", "Sci/Tech"])
        ev = make_evaluator(ds, pipe=pipe)
        metrics = ev.compute()
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_half_accuracy(self) -> None:
        ds = _make_classlabel_dataset(
            ["a", "b", "c", "d"],
            [0, 1, 2, 3],
        )
        # 2 out of 4 correct.
        pipe = _pipe_returning(["World", "Business", "Business", "World"])
        ev = make_evaluator(ds, pipe=pipe)
        metrics = ev.compute()
        assert metrics["accuracy"] == pytest.approx(0.5)

    def test_custom_hypothesis_template_passed(self) -> None:
        ds = _make_classlabel_dataset(["a"], [0])
        pipe = _pipe_returning(["World"])
        ev = make_evaluator(
            ds,
            columns_mapping={
                "input_column": "text",
                "label_column": "label",
                "hypothesis_template": "The topic is {}.",
            },
            pipe=pipe,
        )
        ev.compute()
        _, call_kwargs = pipe.call_args
        assert call_kwargs["hypothesis_template"] == "The topic is {}."

    def test_default_template_not_passed_when_unset(self) -> None:
        ds = _make_classlabel_dataset(["a"], [0])
        pipe = _pipe_returning(["World"])
        ev = make_evaluator(ds, pipe=pipe)
        ev.compute()
        _, call_kwargs = pipe.call_args
        # Template is omitted so pipeline uses its own default.
        assert "hypothesis_template" not in call_kwargs

    def test_candidate_labels_passed_to_pipe(self) -> None:
        ds = _make_classlabel_dataset(["a"], [0])
        pipe = _pipe_returning(["World"])
        ev = make_evaluator(ds, pipe=pipe)
        ev.compute()
        _, call_kwargs = pipe.call_args
        assert call_kwargs["candidate_labels"] == CANDIDATE_LABELS

    def test_f1_zero_division_handled(self) -> None:
        """Macro F1 should not crash when a class has no predictions."""
        ds = _make_classlabel_dataset(
            ["a", "b"],
            [0, 1],
        )
        # Both predictions collapse to one class — other classes have no preds.
        pipe = _pipe_returning(["World", "World"])
        ev = make_evaluator(ds, pipe=pipe)
        metrics = ev.compute()
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_string_labels_compute_end_to_end(self) -> None:
        ds = _make_string_dataset(
            ["a", "b"],
            ["World", "Sports"],
        )
        pipe = _pipe_returning(["World", "Sports"])
        ev = make_evaluator(
            ds,
            columns_mapping={
                "input_column": "text",
                "label_column": "label",
                "candidate_labels": ",".join(CANDIDATE_LABELS),
            },
            pipe=pipe,
        )
        metrics = ev.compute()
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_override_remaps_classlabel_references(self) -> None:
        """Override replaces ClassLabel names positionally for references."""
        ds = _make_classlabel_dataset(
            ["a", "b", "c", "d"],
            [0, 1, 2, 3],
        )
        override = ["politics", "sports", "technology", "science"]
        # Predictions in override vocab, perfectly aligned with gold IDs.
        pipe = _pipe_returning(override)
        ev = make_evaluator(
            ds,
            columns_mapping={
                "input_column": "text",
                "label_column": "label",
                "candidate_labels": ",".join(override),
            },
            pipe=pipe,
        )
        metrics = ev.compute()
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_override_length_mismatch_raises(self) -> None:
        """Override length must match ClassLabel cardinality."""
        ds = _make_classlabel_dataset(
            ["a", "b", "c", "d"],
            [0, 1, 2, 3],
        )
        pipe = _pipe_returning(["politics", "sports"])
        ev = make_evaluator(
            ds,
            columns_mapping={
                "input_column": "text",
                "label_column": "label",
                "candidate_labels": "politics,sports",
            },
            pipe=pipe,
        )
        with pytest.raises(ValueError, match="one override label per class"):
            ev.compute()
