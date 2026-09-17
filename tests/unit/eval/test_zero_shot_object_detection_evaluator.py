# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for generalized zero-shot object detection evaluation."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import ClassLabel, Dataset, Features, Image, Sequence, Value

from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.eval.zero_shot_object_detection_evaluator import (
    QueryChunk,
    WinMLZeroShotObjectDetectionEvaluator,
    _apply_query_budget,
    _extract_category_vocabulary,
    _filter_annotation_to_vocabulary,
    _make_query_chunks,
    _normalize_max_queries,
    _query_capacity,
    _remap_grounded_output,
    _select_category_covering_rows,
    _validate_prompt_template,
)
from winml.modelkit.utils.eval_utils import TASK_SCHEMAS, DatasetValidationError


def _features(category_feature=None):
    category_feature = category_feature or ClassLabel(names=["cat", "dog", "bird"])
    return Features(
        {
            "image": Image(),
            "image_id": Value("int64"),
            "objects": {
                "bbox": Sequence(Sequence(Value("float32"), length=4)),
                "category": Sequence(category_feature),
            },
        }
    )


class TestRegistrationAndSchema:
    def test_distinct_schema_and_lazy_registry_entry(self):
        evaluate_module = importlib.import_module("winml.modelkit.eval.evaluate")

        sys.modules.pop("winml.modelkit.eval.zero_shot_object_detection_evaluator", None)
        task = "zero-shot-object-detection"
        assert task in TASK_SCHEMAS
        assert task in evaluate_module._EVALUATOR_REGISTRY
        assert "winml.modelkit.eval.zero_shot_object_detection_evaluator" not in sys.modules

        cls = evaluate_module.get_evaluator_class(WinMLEvaluationConfig(task=task))
        assert cls.__name__ == "WinMLZeroShotObjectDetectionEvaluator"
        assert cls.__module__ == "winml.modelkit.eval.zero_shot_object_detection_evaluator"

    def test_default_dataset_is_pinned_coco_xywh(self):
        from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS

        default = _DEFAULT_DATASETS["zero-shot-object-detection"]
        assert default["path"] == "detection-datasets/coco"
        assert default["split"] == "val"
        assert default["revision"] == "cf0b22332314a937e9dc8a1957b21725430bb41d"
        assert default["max_queries"] == 16
        assert default["columns_mapping"]["box_format"] == "xywh"
        assert default["columns_mapping"]["prompt_template"] == "a photo of a {}"

    def test_schema_documents_authoritative_categories_and_prompt(self):
        schema = TASK_SCHEMAS["zero-shot-object-detection"]
        params = {item.name: item for item in schema.params}
        assert "authoritative category names" in params["category_key"].description
        assert params["prompt_template"].default == "a photo of a {}"
        assert params["max_queries"].default == "16"

    def test_prepare_pipeline_requires_model_id(self):
        evaluator = object.__new__(WinMLZeroShotObjectDetectionEvaluator)
        evaluator.config = WinMLEvaluationConfig(
            model_id=None,
            task="zero-shot-object-detection",
        )
        evaluator.model = MagicMock()

        with (
            patch("winml.modelkit.eval.base_evaluator.WinMLEvaluator.prepare_pipeline"),
            pytest.raises(ValueError, match="requires --model-id"),
        ):
            evaluator.prepare_pipeline()

    def test_prepare_pipeline_threads_trust_remote_code(self):
        evaluator = object.__new__(WinMLZeroShotObjectDetectionEvaluator)
        evaluator.config = WinMLEvaluationConfig(
            model_id="google/owlv2-base-patch16-finetuned",
            task="zero-shot-object-detection",
            trust_remote_code=True,
        )
        evaluator.model = MagicMock()
        pipe = SimpleNamespace(tokenizer=MagicMock(), image_processor=MagicMock())

        with (
            patch(
                "winml.modelkit.eval.base_evaluator.WinMLEvaluator.prepare_pipeline",
                return_value=pipe,
            ),
            patch("transformers.AutoProcessor.from_pretrained") as mock_from_pretrained,
        ):
            mock_from_pretrained.return_value = SimpleNamespace()
            evaluator.prepare_pipeline()

        assert mock_from_pretrained.call_args.kwargs["trust_remote_code"] is True


class TestVocabularyAndPrompt:
    def test_extracts_classlabel_names_without_model_label_map(self):
        assert _extract_category_vocabulary(_features(), "objects", "category") == [
            (0, "cat"),
            (1, "dog"),
            (2, "bird"),
        ]

    def test_plain_integer_categories_fail_closed(self):
        with pytest.raises(DatasetValidationError, match="authoritative names"):
            _extract_category_vocabulary(_features(Value("int64")), "objects", "category")

    def test_explicit_mapping_is_validated_and_sorted_by_id(self):
        result = _extract_category_vocabulary(
            _features(Value("int64")),
            "objects",
            "category",
            {"dog": 4, "cat": 2},
        )
        assert result == [(2, "cat"), (4, "dog")]
        with pytest.raises(DatasetValidationError, match="unique"):
            _extract_category_vocabulary(
                _features(Value("int64")),
                "objects",
                "category",
                {"dog": 1, "cat": 1},
            )

    @pytest.mark.parametrize("template", ["no field", "{} and {}", "{name}"])
    def test_prompt_requires_one_positional_field(self, template):
        with pytest.raises(DatasetValidationError, match="exactly one"):
            _validate_prompt_template(template)

    def test_default_contract_renders_category_name_only(self):
        chunks = _make_query_chunks([(17, "cat")], "a photo of a {}", 1)
        assert chunks == [QueryChunk(("a photo of a cat",), (17,))]


class TestCapacityChunkingAndMapping:
    def test_capacity_is_read_by_input_name_not_position(self):
        io_config = {
            "input_names": ["pixel_values", "input_ids", "attention_mask"],
            "input_shapes": [[1, 3, 960, 960], [2, 16], [2, 16]],
        }
        assert _query_capacity(io_config) == 2

    def test_static_capacity_one_chunks_every_category(self):
        chunks = _make_query_chunks([(4, "cat"), (9, "dog"), (12, "bird")], "{}", 1)
        assert [chunk.category_ids for chunk in chunks] == [(4,), (9,), (12,)]

    def test_multi_query_capacity_replicates_each_category_without_padding(self):
        chunks = _make_query_chunks([(4, "cat"), (9, "dog"), (12, "bird")], "{}", 2)
        assert chunks == [
            QueryChunk(("cat", "cat"), (4, 4)),
            QueryChunk(("dog", "dog"), (9, 9)),
            QueryChunk(("bird", "bird"), (12, 12)),
        ]

    def test_query_budget_truncates_deterministically_by_authoritative_order(self):
        vocabulary = [(2, "cat"), (4, "dog"), (9, "bird")]
        used, accounting = _apply_query_budget(vocabulary, 2)
        assert used == [(2, "cat"), (4, "dog")]
        assert accounting == {
            "query_requested": 2,
            "query_available": 3,
            "query_used": 2,
            "query_truncated": 1,
        }

    def test_query_budget_none_keeps_full_vocabulary(self):
        vocabulary = [(2, "cat"), (4, "dog")]
        used, accounting = _apply_query_budget(vocabulary, None)
        assert used == vocabulary
        assert accounting["query_requested"] == 2
        assert accounting["query_truncated"] == 0

    @pytest.mark.parametrize("invalid", [0, -1, "3", True])
    def test_max_queries_rejects_invalid_values(self, invalid):
        with pytest.raises(DatasetValidationError, match="max_queries"):
            _normalize_max_queries(invalid)

    def test_competing_slot_logits_preserve_replicated_real_query_detections(self):
        from transformers import Owlv2ImageProcessor

        chunk = _make_query_chunks([(12, "bird")], "{}", 2)[0]
        outputs = SimpleNamespace(
            # Transformers takes max over the final axis. Each slot wins once.
            logits=torch.tensor([[[0.25, 0.75], [0.90, 0.10]]]),
            pred_boxes=torch.tensor([[[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]]]),
        )
        grounded = Owlv2ImageProcessor().post_process_object_detection(
            outputs,
            threshold=0.0,
            target_sizes=torch.tensor([[10, 20]]),
        )[0]

        assert grounded["labels"].tolist() == [1, 0]
        result = _remap_grounded_output(grounded, chunk)
        assert result["labels"] == [12, 12]
        assert len(result["boxes"]) == 2
        assert result["scores"] == pytest.approx(
            [torch.sigmoid(torch.tensor(0.75)).item(), torch.sigmoid(torch.tensor(0.90)).item()]
        )


class TestSelection:
    def test_greedy_selection_is_category_covering_and_deterministic(self):
        rows = [
            (0, 30, [0]),
            (1, 20, [0, 1]),
            (2, 10, [2]),
            (3, 5, [1, 2]),
        ]
        selected, covered = _select_category_covering_rows(rows, {0, 1, 2}, 3)
        assert selected == [3, 1]
        assert covered == {0, 1, 2}

    def test_cap_is_respected_when_full_coverage_is_impossible(self):
        selected, covered = _select_category_covering_rows([(0, 0, [0]), (1, 1, [1])], {0, 1}, 1)
        assert selected == [0]
        assert covered == {0}

    def test_out_of_cap_annotation_entries_are_filtered_not_crashing(self):
        annotation = {
            "bbox": [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]],
            "category": [0, 2],
            "area": [1.0, 4.0],
        }
        filtered, categories = _filter_annotation_to_vocabulary(annotation, "category", {0})
        assert categories == [0]
        assert filtered["category"] == [0]
        assert filtered["bbox"] == [[0.0, 0.0, 1.0, 1.0]]
        assert filtered["area"] == [1.0]


def _compute_evaluator(*, model_side_effect=None, empty_annotations=False):
    evaluator = object.__new__(WinMLZeroShotObjectDetectionEvaluator)
    evaluator._vocabulary = [(3, "cat"), (8, "dog"), (13, "bird")]
    evaluator._prompt_template = "a photo of a {}"
    evaluator._image_col = "image"
    evaluator._annotation_col = "objects"
    evaluator._bbox_key = "bbox"
    evaluator._category_key = "category"
    evaluator._box_format = "xywh"
    evaluator._box_coords = "absolute"
    evaluator._selection_accounting = {
        "requested": 2,
        "requested_cap": 2,
        "source_rows_scanned": 10,
        "selected": 2,
        "category_count": 3,
        "categories_covered": 3,
        "query_requested": 3,
        "query_available": 3,
        "query_used": 3,
        "query_truncated": 0,
    }
    from PIL import Image as PILImage

    annotation = (
        {"bbox": [], "category": []}
        if empty_annotations
        else {"bbox": [[1.0, 2.0, 3.0, 4.0]], "category": [3]}
    )
    evaluator.data = [
        {"image": PILImage.new("RGB", (20, 10)), "objects": annotation},
        {"image": PILImage.new("RGB", (20, 10)), "objects": annotation},
    ]
    model = MagicMock()
    model.io_config = {
        "input_names": ["input_ids", "pixel_values", "attention_mask"],
        "input_shapes": [[2, 16], [1, 3, 960, 960], [2, 16]],
    }
    if model_side_effect is None:
        model.return_value = SimpleNamespace(
            logits=torch.tensor([[[0.25, 0.75]]]),
            pred_boxes=torch.tensor([[[0.5, 0.5, 0.2, 0.4]]]),
        )
    else:
        model.side_effect = model_side_effect
    evaluator.model = model
    processor = MagicMock()
    processor.tokenizer.return_value = {
        "input_ids": torch.zeros((2, 16), dtype=torch.int64),
        "attention_mask": torch.ones((2, 16), dtype=torch.int64),
    }
    processor.image_processor.return_value = {"pixel_values": torch.zeros((1, 3, 960, 960))}
    from transformers import Owlv2ImageProcessor

    grounded_processor = Owlv2ImageProcessor()
    processor.post_process_grounded_object_detection.side_effect = (
        lambda outputs, threshold, target_sizes: grounded_processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )
    )
    evaluator._processor = processor
    return evaluator, model, processor


class TestComputeAndAccounting:
    def test_every_image_uses_complete_vocabulary_and_threshold_zero(self):
        evaluator, model, processor = _compute_evaluator()
        metric = MagicMock(return_value={"map": 0.1, "num_images": 2})
        with patch(
            "winml.modelkit.eval.metrics.mean_average_precision.MAPMetric.compute",
            metric,
        ):
            result = evaluator.compute()

        # Each category gets an independent pass; static slots contain replicas.
        assert model.call_count == 6
        prompts = [call.args[0] for call in processor.tokenizer.call_args_list]
        assert prompts == [
            ["a photo of a cat", "a photo of a cat"],
            ["a photo of a dog", "a photo of a dog"],
            ["a photo of a bird", "a photo of a bird"],
            ["a photo of a cat", "a photo of a cat"],
            ["a photo of a dog", "a photo of a dog"],
            ["a photo of a bird", "a photo of a bird"],
        ]
        for call in processor.post_process_grounded_object_detection.call_args_list:
            assert call.kwargs["threshold"] == 0.0
            torch.testing.assert_close(call.kwargs["target_sizes"], torch.tensor([[10, 20]]))
        assert result == {
            "map": 0.1,
            "num_images": 2,
            "requested": 2,
            "requested_cap": 2,
            "source_rows_scanned": 10,
            "selected": 2,
            "category_count": 3,
            "categories_covered": 3,
            "query_requested": 3,
            "query_available": 3,
            "query_used": 3,
            "query_truncated": 0,
            "decoded": 2,
            "skipped": 0,
            "processed": 2,
            "failed": 0,
            "query_count": 3,
            "query_passes": 6,
            "query_capacity": 2,
        }

        predictions = metric.call_args.kwargs["predictions"]
        references = metric.call_args.kwargs["references"]
        assert predictions[0]["boxes"] == [[8.0, 6.0, 12.0, 14.0]] * 3
        assert predictions[0]["labels"] == [3, 8, 13]
        assert predictions[0]["scores"] == pytest.approx(
            [torch.sigmoid(torch.tensor(0.75)).item()] * 3
        )
        assert references[0] == {
            "boxes": [[1.0, 2.0, 3.0, 4.0]],
            "labels": [3],
        }
        assert metric.call_args.kwargs["box_format"] == "xywh"
        assert metric.call_args.kwargs["box_coords"] == "absolute"

    def test_empty_detections_and_annotations_are_valid_metric_inputs(self):
        evaluator, _model, processor = _compute_evaluator(empty_annotations=True)
        processor.post_process_grounded_object_detection.side_effect = None
        processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
        ]
        metric = MagicMock(return_value={"map": -1.0})
        with patch(
            "winml.modelkit.eval.metrics.mean_average_precision.MAPMetric.compute",
            metric,
        ):
            evaluator.compute()
        assert metric.call_args.kwargs["predictions"][0]["boxes"] == []
        assert metric.call_args.kwargs["references"][0]["boxes"] == []

    def test_zero_processed_images_fail_closed_without_computing_metric(self):
        evaluator, _model, _processor = _compute_evaluator()
        evaluator.data = []
        evaluator._selection_accounting["selected"] = 0
        metric = MagicMock(return_value={"map": -1.0})

        with (
            patch(
                "winml.modelkit.eval.metrics.mean_average_precision.MAPMetric.compute",
                metric,
            ),
            pytest.raises(DatasetValidationError, match="processed no images"),
        ):
            evaluator.compute()

        metric.assert_not_called()

    def test_runtime_failure_propagates_instead_of_becoming_skipped(self):
        evaluator, _model, _processor = _compute_evaluator(
            model_side_effect=RuntimeError("ORT inference failed")
        )
        with pytest.raises(RuntimeError, match="ORT inference failed"):
            evaluator.compute()


class TestSchemaValidation:
    def _evaluator(self):
        evaluator = object.__new__(WinMLZeroShotObjectDetectionEvaluator)
        evaluator._image_col = "image"
        evaluator._annotation_col = "objects"
        evaluator._bbox_key = "bbox"
        evaluator._category_key = "category"
        return evaluator

    def test_missing_image_column_is_actionable(self):
        dataset = Dataset.from_dict(
            {"objects": [{"bbox": [], "category": []}]},
            features=Features(
                {
                    "objects": {
                        "bbox": Sequence(Sequence(Value("float32"), length=4)),
                        "category": Sequence(ClassLabel(names=["cat"])),
                    }
                }
            ),
        )
        with pytest.raises(DatasetValidationError, match="missing required column 'image'"):
            self._evaluator()._validate_schema(dataset)

    def test_missing_bbox_field_is_actionable(self):
        dataset = Dataset.from_dict(
            {"image": [None], "objects": [{"category": []}]},
            features=Features(
                {
                    "image": Image(),
                    "objects": {"category": Sequence(ClassLabel(names=["cat"]))},
                }
            ),
        )
        with pytest.raises(DatasetValidationError, match="bbox"):
            self._evaluator()._validate_schema(dataset)


def test_model_wrapper_routes_text_and_image_inputs():
    from winml.modelkit.models.winml import get_winml_class
    from winml.modelkit.models.winml.object_detection import WinMLModelForObjectDetection

    assert get_winml_class("owlv2", "zero-shot-object-detection") is WinMLModelForObjectDetection
    model = object.__new__(WinMLModelForObjectDetection)
    model._session = MagicMock()
    model._session.io_config = {"input_names": ["input_ids", "pixel_values", "attention_mask"]}
    model._session.run.return_value = {
        "logits": torch.zeros((1, 1, 1)).numpy(),
        "pred_boxes": torch.zeros((1, 1, 4)).numpy(),
    }
    model(
        input_ids=torch.zeros((1, 16), dtype=torch.int64),
        pixel_values=torch.zeros((1, 3, 2, 2)),
        attention_mask=torch.ones((1, 16), dtype=torch.int64),
    )
    assert set(model._session.run.call_args.args[0]) == {
        "input_ids",
        "pixel_values",
        "attention_mask",
    }


def test_dataset_config_roundtrip_preserves_explicit_label_mapping():
    config = WinMLEvaluationConfig(
        model_id="test/model",
        task="zero-shot-object-detection",
        dataset=DatasetConfig(path="test/data", label_mapping={"cat": 4}),
    )
    restored = WinMLEvaluationConfig.from_dict(config.to_dict())
    assert restored.dataset.label_mapping == {"cat": 4}


def test_dataset_config_roundtrip_preserves_max_queries_override():
    config = WinMLEvaluationConfig(
        model_id="test/model",
        task="zero-shot-object-detection",
        dataset=DatasetConfig(path="test/data", max_queries=3),
    )
    restored = WinMLEvaluationConfig.from_dict(config.to_dict())
    assert restored.dataset.max_queries == 3
