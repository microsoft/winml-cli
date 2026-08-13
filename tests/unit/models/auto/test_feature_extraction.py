# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for WinMLModelForFeatureExtraction.

Validates forward pass I/O contract: accepts arbitrary **kwargs (architecture-agnostic),
returns a ModelOutput subclass whose fields mirror the ONNX exporter's declared
output names and order.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers.utils import ModelOutput


def create_mock_model():
    """Create WinMLModelForFeatureExtraction with a mocked session."""
    from winml.modelkit.models.winml import WinMLModelForFeatureExtraction

    model = WinMLModelForFeatureExtraction.__new__(WinMLModelForFeatureExtraction)
    mock_session = MagicMock()
    mock_session.io_config = {
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
        "input_types": [np.dtype("int32"), np.dtype("int32"), np.dtype("int32")],
        "input_shapes": [[1, 8], [1, 8], [1, 8]],
        "output_names": ["last_hidden_state"],
    }
    mock_session.run.return_value = {
        "last_hidden_state": np.random.randn(1, 8, 384).astype(np.float32),
    }
    mock_session.device = "cpu"
    model._session = mock_session
    model.config = MagicMock()
    model._onnx_path = "mock.onnx"
    model._device = "cpu"
    return model


class TestWinMLModelForFeatureExtractionBasic:
    def test_class_importable(self):
        from winml.modelkit.models.winml import WinMLModelForFeatureExtraction

        assert WinMLModelForFeatureExtraction is not None

    def test_inherits_from_base(self):
        from winml.modelkit.models.winml import WinMLModelForFeatureExtraction, WinMLPreTrainedModel

        assert issubclass(WinMLModelForFeatureExtraction, WinMLPreTrainedModel)

    def test_exported_from_winml_package(self):
        from winml.modelkit.models.winml import WinMLModelForFeatureExtraction

        assert WinMLModelForFeatureExtraction is not None


class TestForwardLastHiddenState:
    def test_returns_model_output(self):
        model = create_mock_model()
        input_ids = torch.ones(1, 8, dtype=torch.long)
        result = model.forward(input_ids=input_ids)
        assert isinstance(result, ModelOutput)

    def test_last_hidden_state_shape(self):
        model = create_mock_model()
        model._session.run.return_value = {
            "last_hidden_state": np.zeros((1, 8, 384), dtype=np.float32),
        }
        result = model.forward(input_ids=torch.ones(1, 8, dtype=torch.long))
        assert result.last_hidden_state.shape == (1, 8, 384)
        assert result[0].shape == (1, 8, 384)

    def test_optional_inputs_forwarded(self):
        model = create_mock_model()
        input_ids = torch.ones(1, 8, dtype=torch.long)
        attention_mask = torch.ones(1, 8, dtype=torch.long)
        token_type_ids = torch.zeros(1, 8, dtype=torch.long)

        model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        call_kwargs = model._session.run.call_args[0][0]
        assert "attention_mask" in call_kwargs
        assert "token_type_ids" in call_kwargs

    def test_missing_token_type_ids_synthesized_with_required_shape_and_dtype(self):
        model = create_mock_model()
        model.forward(
            input_ids=torch.ones(1, 8, dtype=torch.long),
            attention_mask=torch.ones(1, 8, dtype=torch.long),
        )

        call_kwargs = model._session.run.call_args[0][0]
        np.testing.assert_array_equal(call_kwargs["token_type_ids"], np.zeros((1, 8)))
        assert call_kwargs["token_type_ids"].dtype == np.int32

    def test_provided_token_type_ids_preserved(self):
        model = create_mock_model()
        provided = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int64)

        model.forward(
            input_ids=torch.ones(1, 8, dtype=torch.long),
            attention_mask=torch.ones(1, 8, dtype=torch.long),
            token_type_ids=provided,
        )

        call_kwargs = model._session.run.call_args[0][0]
        np.testing.assert_array_equal(call_kwargs["token_type_ids"], provided.numpy())
        assert call_kwargs["token_type_ids"].dtype == provided.numpy().dtype

    def test_unrelated_missing_required_input_still_fails(self):
        from winml.modelkit.session.session import WinMLSession

        model = create_mock_model()
        model._session.run.side_effect = lambda inputs: (
            WinMLSession._validate_inputs(model._session, inputs)
        )

        with pytest.raises(ValueError, match="attention_mask"):
            model.forward(input_ids=torch.ones(1, 8, dtype=torch.long))

    def test_incompatible_static_token_type_shape_is_not_hidden(self):
        from winml.modelkit.session.session import WinMLSession

        model = create_mock_model()
        model._session.io_config["input_shapes"][2] = [1, 16]
        model._session.run.side_effect = lambda inputs: (
            WinMLSession._validate_inputs(model._session, inputs)
        )

        with pytest.raises(ValueError, match="token_type_ids"):
            model.forward(
                input_ids=torch.ones(1, 8, dtype=torch.long),
                attention_mask=torch.ones(1, 8, dtype=torch.long),
            )

    def test_sentence_similarity_feature_path_synthesizes_each_single_segment(self):
        model = create_mock_model()

        for token_id in (1, 2):
            model.forward(
                input_ids=torch.full((1, 8), token_id, dtype=torch.long),
                attention_mask=torch.ones(1, 8, dtype=torch.long),
            )

        assert model._session.run.call_count == 2
        for call in model._session.run.call_args_list:
            np.testing.assert_array_equal(call.args[0]["token_type_ids"], np.zeros((1, 8)))


class TestForwardPreservesOnnxOutputNames:
    """ONNX output names and shapes are exposed verbatim (no rename, no unsqueeze)."""

    def test_pre_pooled_output_preserved(self):
        from winml.modelkit.models.winml import WinMLModelForFeatureExtraction

        model = WinMLModelForFeatureExtraction.__new__(WinMLModelForFeatureExtraction)
        mock_session = MagicMock()
        mock_session.io_config = {
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["sentence_embedding"],
        }
        mock_session.run.return_value = {
            "sentence_embedding": np.zeros((1, 384), dtype=np.float32),
        }
        mock_session.device = "cpu"
        model._session = mock_session
        model.config = MagicMock()
        model._onnx_path = "mock.onnx"
        model._device = "cpu"

        result = model.forward(input_ids=torch.ones(1, 8, dtype=torch.long))

        assert result.sentence_embedding.shape == (1, 384)
        assert result[0].shape == (1, 384)

    def test_multi_output_preserves_order_and_names(self):
        """CLIP-style export with projected embedding first, hidden states second."""
        from winml.modelkit.models.winml import WinMLModelForFeatureExtraction

        model = WinMLModelForFeatureExtraction.__new__(WinMLModelForFeatureExtraction)
        mock_session = MagicMock()
        mock_session.io_config = {
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["text_embeds", "last_hidden_state"],
        }
        mock_session.run.return_value = {
            "text_embeds": np.zeros((1, 768), dtype=np.float32),
            "last_hidden_state": np.zeros((1, 77, 768), dtype=np.float32),
        }
        mock_session.device = "cpu"
        model._session = mock_session
        model.config = MagicMock()
        model._onnx_path = "mock.onnx"
        model._device = "cpu"

        result = model.forward(input_ids=torch.ones(1, 77, dtype=torch.long))

        assert result.text_embeds.shape == (1, 768)
        assert result.last_hidden_state.shape == (1, 77, 768)
        # HF pipelines consume output[0]; must match exporter's first output.
        assert result[0].shape == (1, 768)
