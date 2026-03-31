# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLObjectDetectionEvaluator schema validation and label alignment."""

import pytest
from datasets import ClassLabel, Dataset, Features, Sequence, Value

from winml.modelkit.datasets import DatasetConfig
from winml.modelkit.eval import WinMLObjectDetectionEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockConfig:
    def __init__(self, label2id=None, id2label=None):
        self.label2id = label2id
        self.id2label = id2label


class MockModel:
    def __init__(self, label2id=None):
        id2label = {v: k for k, v in label2id.items()} if label2id else None
        self.config = MockConfig(label2id, id2label)


def make_evaluator(label2id=None, columns_mapping=None):
    """Create evaluator without triggering __init__ data loading."""
    ev = object.__new__(WinMLObjectDetectionEvaluator)
    ev.model = MockModel(label2id)
    ev._annotation_col = "objects"
    ev._bbox_key = "bbox"
    ev._category_key = "category"
    if columns_mapping:
        ev._annotation_col = columns_mapping.get("annotation_column", "objects")
        ev._bbox_key = columns_mapping.get("bbox_key", "bbox")
        ev._category_key = columns_mapping.get("category_key", "category")
    return ev


def make_od_dataset(category_ids, class_names):
    """Build a dataset with object-detection annotation structure."""
    features = Features(
        {
            "image": Value("string"),
            "objects": {
                "bbox": Sequence(Sequence(Value("float32"), length=4)),
                "category": Sequence(ClassLabel(names=class_names)),
            },
        }
    )
    return Dataset.from_dict(
        {
            "image": [f"img_{i}.jpg" for i in range(len(category_ids))],
            "objects": [
                {"bbox": [[0, 0, 1, 1]] * len(ids), "category": ids} for ids in category_ids
            ],
        },
        features=features,
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_valid_schema_passes(self):
        ev = make_evaluator()
        ds = make_od_dataset([[0]], ["cat"])
        ev._validate_schema(ds)  # should not raise

    def test_missing_annotation_column_raises(self):
        ev = make_evaluator()
        ds = Dataset.from_dict({"image": ["a.jpg"], "label": [0]})
        with pytest.raises(ValueError, match="No column 'objects'"):
            ev._validate_schema(ds)

    def test_missing_bbox_key_raises(self):
        ev = make_evaluator()
        features = Features(
            {
                "image": Value("string"),
                "objects": {"category": Sequence(Value("int32"))},
            }
        )
        ds = Dataset.from_dict(
            {"image": ["a.jpg"], "objects": [{"category": [0]}]},
            features=features,
        )
        with pytest.raises(ValueError, match="has no key 'bbox'"):
            ev._validate_schema(ds)

    def test_missing_category_key_raises(self):
        ev = make_evaluator()
        features = Features(
            {
                "image": Value("string"),
                "objects": {
                    "bbox": Sequence(Sequence(Value("float32"), length=4)),
                },
            }
        )
        ds = Dataset.from_dict(
            {"image": ["a.jpg"], "objects": [{"bbox": [[0, 0, 1, 1]]}]},
            features=features,
        )
        with pytest.raises(ValueError, match="has no key 'category'"):
            ev._validate_schema(ds)


# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------


class TestAlignLabels:
    def test_already_aligned_skips(self):
        """When dataset IDs match model IDs, no remapping occurs."""
        ev = make_evaluator(label2id={"cat": 0, "dog": 1})
        ds = make_od_dataset([[0, 1]], ["cat", "dog"])
        ds_config = DatasetConfig(path="test")

        result = ev.align_labels(ds, ds_config)
        assert result["objects"][0]["category"] == [0, 1]

    def test_remaps_ids(self):
        """Dataset cat=0,dog=1 remapped to model N/A=0,cat=1,dog=2 (DETR-style)."""
        ev = make_evaluator(label2id={"N/A": 0, "cat": 1, "dog": 2})
        ds = make_od_dataset([[0, 1], [1, 0]], ["cat", "dog"])
        ds_config = DatasetConfig(path="test")

        result = ev.align_labels(ds, ds_config)
        assert result["objects"][0]["category"] == [1, 2]
        assert result["objects"][1]["category"] == [2, 1]

    def test_unknown_label_raises(self):
        """Dataset has label not in model's label2id."""
        ev = make_evaluator(label2id={"cat": 0})
        ds = make_od_dataset([[0, 1]], ["cat", "dog"])
        ds_config = DatasetConfig(path="test")

        with pytest.raises(ValueError, match="Dataset label 'dog' not in"):
            ev.align_labels(ds, ds_config)

    def test_no_label2id_warns_and_skips(self):
        """No label2id available — warns and returns dataset unchanged."""
        ev = make_evaluator(label2id=None)
        ds = make_od_dataset([[0]], ["cat"])
        ds_config = DatasetConfig(path="test")

        result = ev.align_labels(ds, ds_config)
        assert result["objects"][0]["category"] == [0]

    def test_not_classlabel_warns_and_skips(self):
        """Category is plain int, not ClassLabel — warns and skips."""
        ev = make_evaluator(label2id={"cat": 0})
        features = Features(
            {
                "image": Value("string"),
                "objects": {
                    "bbox": Sequence(Sequence(Value("float32"), length=4)),
                    "category": Sequence(Value("int32")),
                },
            }
        )
        ds = Dataset.from_dict(
            {
                "image": ["a.jpg"],
                "objects": [{"bbox": [[0, 0, 1, 1]], "category": [0]}],
            },
            features=features,
        )
        ds_config = DatasetConfig(path="test")

        result = ev.align_labels(ds, ds_config)
        assert result["objects"][0]["category"] == [0]

    def test_user_mapping_overrides_model(self):
        """User-provided label_mapping takes priority."""
        ev = make_evaluator(
            label2id={"N/A": 0, "cat": 1, "dog": 2},
        )
        ds = make_od_dataset([[0, 1]], ["cat", "dog"])
        ds_config = DatasetConfig(
            path="test",
            label_mapping={"cat": 1, "dog": 2},
        )

        result = ev.align_labels(ds, ds_config)
        assert result["objects"][0]["category"] == [1, 2]
