# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLImageToTextEvaluator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from winml.modelkit.eval.image_to_text_evaluator import WinMLImageToTextEvaluator
from winml.modelkit.inference.pipeline import _HF_PIPELINE_TASK_MAP
from winml.modelkit.models.winml.image_to_text import (
    MgpstrImageToTextPipeline,
    WinMLModelForMgpstrSceneTextRecognition,
)


class _MgpstrEvaluationModel(WinMLModelForMgpstrSceneTextRecognition):
    def __init__(self) -> None:
        self.config = SimpleNamespace(model_type="mgp-str")
        self._io_config = {
            "input_names": ["pixel_values"],
            "input_shapes": [[1, 3, 32, 128]],
        }

    @property
    def io_config(self) -> dict:
        return self._io_config

    def _run_inference(self, inputs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        pixel_values = inputs["pixel_values"]
        assert pixel_values.shape == (1, 3, 32, 128)
        return {
            "char_logits": torch.zeros((1, 2, 3)),
            "bpe_logits": torch.zeros((1, 2, 3)),
            "wp_logits": torch.zeros((1, 2, 3)),
        }

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self._run_inference(self._format_inputs(**kwargs))


class _MgpstrEvaluationProcessor:
    def __call__(self, *, images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert images.mode == "RGB"
        resized = images.resize((128, 32))
        pixels = np.asarray(resized, dtype=np.float32).copy()
        return {"pixel_values": torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)}

    def batch_decode(self, logits: tuple[torch.Tensor, ...]) -> dict[str, list[Any]]:
        assert len(logits) == 3
        return {"generated_text": ["TEXT"], "scores": [torch.tensor(1.0)]}


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

        ev.pipe.assert_any_call("img1", text="")
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

    def test_mgpstr_evaluation_normalizes_grayscale_and_decodes_wrapper_output(self):
        """Exercise evaluator -> MGP pipeline -> wrapper without mocks."""
        pipe = object.__new__(MgpstrImageToTextPipeline)
        pipe.model = _MgpstrEvaluationModel()
        pipe.processor = _MgpstrEvaluationProcessor()

        evaluator = object.__new__(WinMLImageToTextEvaluator)
        evaluator._image_col = "image"
        evaluator._label_col = "text"
        evaluator.pipe = pipe
        evaluator.data = [
            {"image": Image.new("L", (16, 8)), "text": "TEXT"},
            {"image": np.zeros((8, 16), dtype=np.uint8), "text": "TEXT"},
            {"image": torch.zeros((1, 8, 16), dtype=torch.uint8), "text": "TEXT"},
        ]

        result = evaluator.compute()

        assert result["cer"] == 0.0
        assert result["n_samples"] == 3
        assert "skipped" not in result
