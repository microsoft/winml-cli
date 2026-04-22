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
from typing import Any
from unittest.mock import MagicMock

from winml.modelkit.inference.pipeline import (
    _HF_PIPELINE_TASK_MAP,
    _adapt_image_processor_size,
    _adapt_tokenizer_padding,
    _detect_tokenizer_dict_param,
)


# ---------------------------------------------------------------------------
# _HF_PIPELINE_TASK_MAP
# ---------------------------------------------------------------------------


class TestHFPipelineTaskMap:
    def test_sentence_similarity_maps_to_feature_extraction(self) -> None:
        assert _HF_PIPELINE_TASK_MAP["sentence-similarity"] == "feature-extraction"

    def test_unknown_task_not_in_map(self) -> None:
        assert "image-classification" not in _HF_PIPELINE_TASK_MAP


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
