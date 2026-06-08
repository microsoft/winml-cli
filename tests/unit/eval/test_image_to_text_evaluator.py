# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLImageToTextEvaluator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from winml.modelkit.eval.image_to_text_evaluator import WinMLImageToTextEvaluator


def make_evaluator(columns_mapping=None):
    """Instantiate evaluator with mocked dataset + pipeline."""
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

    with patch("datasets.load_dataset", return_value=mock_ds), \
         patch("transformers.pipeline", return_value=mock_pipe):
        return WinMLImageToTextEvaluator(config, model)


class TestInit:
    def test_default_columns(self):
        ev = make_evaluator()
        assert ev._image_col == "image"
        assert ev._label_col == "text"

    def test_custom_columns(self):
        ev = make_evaluator(columns_mapping={
            "input_column": "img",
            "label_column": "caption",
        })
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
        ev.pipe = MagicMock(side_effect=[
            [{"generated_text": "HELLO"}],
            [{"generated_text": "WORLD"}],
        ])

        result = ev.compute()

        assert result["cer"] == 0.0
        assert result["n_samples"] == 2
        assert "cider" in result

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
        ev.pipe = MagicMock(side_effect=[
            [{"generated_text": "abc"}],
            [{"generated_text": "abc"}],
        ])

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
        ev.pipe = MagicMock(side_effect=[
            [{"generated_text": "abc"}],
            RuntimeError("model crashed"),
        ])

        result = ev.compute()

        assert result["n_samples"] == 1
        assert result["cer"] == 0.0
        assert result.get("skipped") == 1

    def test_uses_custom_columns(self):
        """Image and label columns from columns_mapping are honoured."""
        ev = make_evaluator(columns_mapping={
            "input_column": "img",
            "label_column": "caption",
        })
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
