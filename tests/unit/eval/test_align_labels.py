# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for align_labels and _filter_unsupported_labels."""

from unittest.mock import patch

import pytest
from datasets import ClassLabel, Dataset, Features, Sequence, Value

from winml.modelkit.eval import DatasetConfig, WinMLEvaluator
from winml.modelkit.utils.eval_utils import DatasetValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockConfig:
    def __init__(self, label2id=None, id2label=None):
        self.label2id = label2id
        self.id2label = id2label


class MockModel:
    def __init__(self, label2id=None, id2label=None):
        if label2id and not id2label:
            id2label = {v: k for k, v in label2id.items()}
        if id2label and not label2id:
            label2id = {v: int(k) for k, v in id2label.items()}
        self.config = MockConfig(label2id, id2label)


def make_evaluator(label2id=None, id2label=None, task="image-classification"):
    ev = object.__new__(WinMLEvaluator)
    ev.model = MockModel(label2id, id2label)
    ev.config = type("Cfg", (), {"task": task})()
    return ev


def make_dataset(labels, names):
    features = Features({"text": Value("string"), "label": ClassLabel(names=names)})
    return Dataset.from_dict(
        {"text": [f"t{i}" for i in range(len(labels))], "label": labels},
        features=features,
    )


def make_ner_dataset(tag_seqs, names):
    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=names)),
        }
    )
    tokens = [["w"] * len(seq) for seq in tag_seqs]
    return Dataset.from_dict(
        {"tokens": tokens, "ner_tags": tag_seqs},
        features=features,
    )


# ---------------------------------------------------------------------------
# _get_label_mapping priority
# ---------------------------------------------------------------------------


class TestGetLabelMappingPriority:
    def test_user_mapping_wins(self):
        ev = make_evaluator(label2id={"cat": 0, "dog": 1})
        ds_config = DatasetConfig(
            path="timm/mini-imagenet",
            label_mapping={"custom": 99},
        )
        result = ev._get_label_mapping(ds_config)
        assert result == {"custom": 99}

    @patch("winml.modelkit.datasets.label_utils.should_align_labels", return_value=True)
    @patch("winml.modelkit.datasets.label_utils.get_label_mapping", return_value={"syn": 0})
    def test_known_dataset_fallback(self, mock_get, mock_should):
        ev = make_evaluator(label2id={"cat": 0})
        ds_config = DatasetConfig(path="timm/mini-imagenet")
        result = ev._get_label_mapping(ds_config)
        assert result == {"syn": 0}

    def test_model_label2id_fallback(self):
        ev = make_evaluator(label2id={"cat": 0, "dog": 1})
        ds_config = DatasetConfig(path="some/dataset")
        result = ev._get_label_mapping(ds_config)
        assert result == {"cat": 0, "dog": 1}

    def test_no_mapping_available(self):
        ev = make_evaluator()
        ds_config = DatasetConfig(path="some/dataset")
        result = ev._get_label_mapping(ds_config)
        assert result is None


# ---------------------------------------------------------------------------
# align_labels guards
# ---------------------------------------------------------------------------


class TestAlignLabelsGuards:
    def test_skip_if_label_column_missing(self):
        ev = make_evaluator(label2id={"cat": 0})
        ds = Dataset.from_dict({"text": ["a", "b"]})
        ds_config = DatasetConfig(path="test")
        result = ev.align_labels(ds, ds_config)
        assert len(result) == 2

    def test_skip_if_sequence_classlabel(self):
        ev = make_evaluator(label2id={"O": 0, "B-PER": 1})
        ds = make_ner_dataset([[0, 1], [0, 0]], names=["O", "B-PER"])
        ds_config = DatasetConfig(
            path="test",
            columns_mapping={"label_column": "ner_tags"},
        )
        result = ev.align_labels(ds, ds_config)
        assert len(result) == 2
        assert result["ner_tags"] == [[0, 1], [0, 0]]

    def test_skip_if_no_mapping(self):
        ev = make_evaluator()
        ds = make_dataset([0, 1], names=["cat", "dog"])
        ds_config = DatasetConfig(path="test")
        result = ev.align_labels(ds, ds_config)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# align_labels end-to-end
# ---------------------------------------------------------------------------


class TestAlignLabelsEndToEnd:
    def test_all_labels_match(self):
        ev = make_evaluator(label2id={"cat": 0, "dog": 1})
        ds = make_dataset([0, 1, 0], names=["cat", "dog"])
        ds_config = DatasetConfig(path="test")
        result = ev.align_labels(ds, ds_config)
        assert len(result) == 3

    def test_model_has_extra_labels(self):
        ev = make_evaluator(label2id={"cat": 0, "dog": 1, "bird": 2})
        ds = make_dataset([0, 1, 0], names=["cat", "dog"])
        ds_config = DatasetConfig(path="test")
        result = ev.align_labels(ds, ds_config)
        assert len(result) == 3

    def test_dataset_has_extra_labels_raises(self):
        """Model label2id missing dataset labels → alignment fails."""
        ev = make_evaluator(label2id={"cat": 0, "dog": 1})
        ds = make_dataset([0, 1, 2, 0, 2], names=["cat", "dog", "bird"])
        ds_config = DatasetConfig(path="test")
        with pytest.raises(DatasetValidationError):
            ev.align_labels(ds, ds_config)

    def test_zero_overlap_raises(self):
        ev = make_evaluator(label2id={"fish": 0, "shark": 1})
        ds = make_dataset([0, 1], names=["cat", "dog"])
        ds_config = DatasetConfig(path="test")
        with pytest.raises(DatasetValidationError):
            ev.align_labels(ds, ds_config)

    def test_different_ordering_reordered(self):
        ev = make_evaluator(label2id={"dog": 0, "cat": 1})
        ds = make_dataset([0, 1, 0], names=["cat", "dog"])
        ds_config = DatasetConfig(path="test")
        result = ev.align_labels(ds, ds_config)
        # cat was ds_id=0, model wants cat=1
        assert result["label"][0] == 1
        # dog was ds_id=1, model wants dog=0
        assert result["label"][1] == 0

    def test_user_mapping_applied(self):
        ev = make_evaluator(label2id={"a": 0, "b": 1})
        ds = make_dataset([0, 1, 0], names=["a", "b"])
        # User provides explicit mapping with different ordering
        ds_config = DatasetConfig(
            path="test",
            label_mapping={"a": 1, "b": 0},
        )
        result = ev.align_labels(ds, ds_config)
        # a was ds_id=0, user maps a→1
        assert result["label"][0] == 1
        # b was ds_id=1, user maps b→0
        assert result["label"][1] == 0


# ---------------------------------------------------------------------------
# _filter_unsupported_labels
# ---------------------------------------------------------------------------


class TestFilterUnsupportedLabels:
    def test_no_id2label_skips(self):
        ev = make_evaluator()
        ds = make_dataset([0, 1], names=["cat", "dog"])
        result = ev._filter_unsupported_labels(ds, "label")
        assert len(result) == 2

    def test_all_supported_no_filter(self):
        ev = make_evaluator(id2label={0: "cat", 1: "dog"})
        ds = make_dataset([0, 1, 0], names=["cat", "dog"])
        result = ev._filter_unsupported_labels(ds, "label")
        assert len(result) == 3

    def test_partial_filter(self):
        ev = make_evaluator(id2label={0: "cat", 1: "dog"})
        ds = make_dataset([0, 1, 5, 0, 5], names=["cat", "dog", "x", "y", "z", "bird"])
        result = ev._filter_unsupported_labels(ds, "label")
        assert len(result) == 3

    def test_zero_remaining_raises(self):
        ev = make_evaluator(id2label={99: "x"})
        ds = make_dataset([0, 1], names=["cat", "dog"])
        with pytest.raises(DatasetValidationError, match="No samples remain"):
            ev._filter_unsupported_labels(ds, "label")

    def test_custom_label_column(self):
        features = Features({"text": Value("string"), "tag": Value("int64")})
        ds = Dataset.from_dict(
            {"text": ["a", "b", "c"], "tag": [0, 1, 5]},
            features=features,
        )
        ev = make_evaluator(id2label={0: "a", 1: "b"})
        result = ev._filter_unsupported_labels(ds, "tag")
        assert len(result) == 2
