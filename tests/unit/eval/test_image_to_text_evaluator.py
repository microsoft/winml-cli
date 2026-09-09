# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLImageToTextEvaluator."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from transformers.pipelines import ImageTextToTextPipeline, ImageToTextPipeline

from winml.modelkit.eval import WinMLImageToTextEvaluator
from winml.modelkit.inference.pipeline import _HF_PIPELINE_TASK_MAP


def make_evaluator(columns_mapping=None):
    """Instantiate evaluator with mocked dataset + pipeline."""
    import transformers

    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    mapping = columns_mapping or {}

    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: 0
    mock_ds.shuffle.return_value = mock_ds
    mock_ds.select.return_value = mock_ds
    mock_ds.column_names = [
        mapping.get("input_column", "image"),
        mapping.get("label_column", "text"),
    ]

    mock_pipe = MagicMock()
    model = MagicMock()
    model.config.label2id = None

    config = WinMLEvaluationConfig(
        model_id="microsoft/trocr-base-handwritten",
        task="image-to-text",
        dataset=DatasetConfig(path="Teklia/IAM-line", columns_mapping=mapping),
    )

    # Resolve the lazy Transformers export before patching it.
    assert hasattr(transformers, "pipeline")
    with (
        patch("datasets.load_dataset", return_value=mock_ds),
        patch("transformers.pipelines.pipeline", return_value=mock_pipe),
        patch.object(transformers, "pipeline", return_value=mock_pipe),
    ):
        return WinMLImageToTextEvaluator(config, model)


class TestInit:
    def test_uses_transformers_pipeline_task_name(self):
        assert _HF_PIPELINE_TASK_MAP["image-to-text"] == "image-text-to-text"

    def test_default_columns(self):
        ev = make_evaluator()
        assert ev._image_col == "image"
        assert ev._label_col == "text"

    def test_custom_columns(self):
        ev = make_evaluator(
            columns_mapping={
                "input_column": "img",
                "label_column": "caption",
            }
        )
        assert ev._image_col == "img"
        assert ev._label_col == "caption"


class TestAlignLabels:
    def test_align_labels_is_noop(self):
        ev = make_evaluator()
        mock_dataset = MagicMock()
        result = ev.align_labels(mock_dataset, MagicMock())
        assert result is mock_dataset


class TestRegistry:
    def test_registered(self):
        from winml.modelkit.eval import WinMLEvaluationConfig
        from winml.modelkit.eval.evaluate import _EVALUATOR_REGISTRY, get_evaluator_class

        assert "image-to-text" in _EVALUATOR_REGISTRY
        # Registry stores "module:Class" strings now (lazy resolution).
        assert (
            get_evaluator_class(WinMLEvaluationConfig(task="image-to-text"))
            is WinMLImageToTextEvaluator
        )


class TestCompute:
    """compute() iterates samples through the pipeline and aggregates metrics."""

    def test_perfect_predictions(self):
        """When pipeline returns exactly the reference, CER should be 0."""
        ev = make_evaluator()
        ev.data = [
            {"image": "img1", "text": "HELLO"},
            {"image": "img2", "text": "WORLD"},
        ]
        ev.pipe = MagicMock(
            side_effect=[
                [{"generated_text": "HELLO"}],
                [{"generated_text": "WORLD"}],
            ]
        )

        result = ev.compute()

        ev.pipe.assert_any_call("img1")
        assert result["cer"] == 0.0
        assert result["n_samples"] == 2
        assert "cider" in result

    @pytest.mark.parametrize("pipeline_class", [ImageToTextPipeline, ImageTextToTextPipeline])
    def test_real_pipeline_call_contract(self, pipeline_class):
        """Exercise real argument validation without loading model weights."""
        pipe = object.__new__(pipeline_class)
        pipe._num_workers = None
        pipe._batch_size = None
        pipe._preprocess_params = {}
        pipe._forward_params = {}
        pipe._postprocess_params = {}
        pipe.call_count = 0
        pipe.framework = "pt"
        pipe.assistant_model = None
        pipe.assistant_tokenizer = None
        pipe.processor = SimpleNamespace(chat_template=None)

        def predict(inputs, *_args):
            if isinstance(inputs, dict):
                assert inputs["text"] == ""
                image = inputs["images"]
            else:
                image = inputs
            return [{"generated_text": str(image.getpixel((0, 0)))}]

        pipe.run_single = MagicMock(side_effect=predict)
        ev = make_evaluator()
        images = [Image.new("RGB", (2, 2), color=(index, 0, 0)) for index in range(2)]
        ev.data = [{"image": image, "text": str(image.getpixel((0, 0)))} for image in images]
        ev.pipe = pipe

        result = ev.compute()

        assert pipe.run_single.call_count == len(images)
        assert result["n_samples"] == len(images)
        assert result["cer"] == 0.0
        assert "skipped" not in result

    def test_dict_output_shape(self):
        """Pipeline may also return a single dict (not a list)."""
        ev = make_evaluator()
        ev.data = [{"image": "img1", "text": "HELLO"}]
        ev.pipe = MagicMock(return_value={"generated_text": "HELLO"})

        result = ev.compute()
        assert result["cer"] == 0.0
        assert result["n_samples"] == 1

    def test_skips_samples_with_missing_data(self):
        """None image or None text → skipped, n_samples reflects actual count."""
        ev = make_evaluator()
        ev.data = [
            {"image": "img1", "text": "abc"},
            {"image": None, "text": "skipped"},
            {"image": "img2", "text": None},
            {"image": "img3", "text": "abc"},
        ]
        ev.pipe = MagicMock(
            side_effect=[
                [{"generated_text": "abc"}],
                [{"generated_text": "abc"}],
            ]
        )

        result = ev.compute()

        assert ev.pipe.call_count == 2
        assert result["n_samples"] == 2
        assert result["cer"] == 0.0
        assert result.get("skipped") == 2

    def test_pipeline_exception_skipped(self):
        """If the pipeline raises, the sample is skipped (not fatal)."""
        ev = make_evaluator()
        ev.data = [
            {"image": "img1", "text": "abc"},
            {"image": "img2", "text": "abc"},
        ]
        ev.pipe = MagicMock(
            side_effect=[
                [{"generated_text": "abc"}],
                RuntimeError("model crashed"),
            ]
        )

        result = ev.compute()

        assert result["n_samples"] == 1
        assert result["cer"] == 0.0
        assert result.get("skipped") == 1

    @pytest.mark.parametrize("error_class", [TypeError, RuntimeError])
    def test_all_pipeline_exceptions_raise(self, error_class):
        """A broken pipeline must not return successful empty metrics."""
        ev = make_evaluator()
        ev.data = [{"image": f"image-{index}", "text": str(index)} for index in range(2)]
        error = error_class("pipeline failed")
        ev.pipe = MagicMock(side_effect=error)

        with pytest.raises(RuntimeError, match="produced no predictions: pipeline failed") as exc:
            ev.compute()

        assert exc.value.__cause__ is error
        assert ev.pipe.call_count == len(ev.data)

    def test_uses_custom_columns(self):
        """Image and label columns from columns_mapping are honoured."""
        ev = make_evaluator(
            columns_mapping={
                "input_column": "img",
                "label_column": "caption",
            }
        )
        ev.data = [{"img": "x", "caption": "abc"}]
        ev.pipe = MagicMock(return_value=[{"generated_text": "abc"}])

        result = ev.compute()
        assert result["cer"] == 0.0
        assert result["n_samples"] == 1

    def test_empty_dataset(self):
        """Empty data returns metric dict with n_samples=0 and Nones."""
        ev = make_evaluator()
        ev.data = []
        ev.pipe = MagicMock()

        result = ev.compute()
        assert result["n_samples"] == 0
        assert result["cer"] is None
        assert result["cider"] is None
