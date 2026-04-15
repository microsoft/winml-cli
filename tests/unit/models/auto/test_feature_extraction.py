# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for WinMLModelForFeatureExtraction.

Validates forward pass I/O contract: accepts arbitrary **kwargs (architecture-agnostic),
returns BaseModelOutput with last_hidden_state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput


def create_mock_model():
    """Create WinMLModelForFeatureExtraction with a mocked session."""
    from winml.modelkit.models.winml import WinMLModelForFeatureExtraction

    model = WinMLModelForFeatureExtraction.__new__(WinMLModelForFeatureExtraction)
    mock_session = MagicMock()
    mock_session.io_config = {
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
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
    def test_returns_base_model_output(self):
        model = create_mock_model()
        input_ids = torch.ones(1, 8, dtype=torch.long)
        result = model.forward(input_ids=input_ids)
        assert isinstance(result, BaseModelOutput)

    def test_last_hidden_state_shape(self):
        model = create_mock_model()
        model._session.run.return_value = {
            "last_hidden_state": np.zeros((1, 8, 384), dtype=np.float32),
        }
        result = model.forward(input_ids=torch.ones(1, 8, dtype=torch.long))
        assert result.last_hidden_state.shape == (1, 8, 384)

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

    def test_none_inputs_excluded(self):
        model = create_mock_model()
        model.forward(input_ids=torch.ones(1, 8, dtype=torch.long))

        call_kwargs = model._session.run.call_args[0][0]
        assert "attention_mask" not in call_kwargs
        assert "token_type_ids" not in call_kwargs


class TestForwardSentenceEmbedding:
    """When ONNX exports a pre-pooled sentence_embedding, it should be wrapped."""

    def test_sentence_embedding_unsqueezed(self):
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

        # [B, hidden_dim] -> [B, 1, hidden_dim]
        assert result.last_hidden_state.shape == (1, 1, 384)

    def test_generic_2d_output_unsqueezed(self):
        """Any unknown 2-D output is wrapped as [B, 1, H]."""
        from winml.modelkit.models.winml import WinMLModelForFeatureExtraction

        model = WinMLModelForFeatureExtraction.__new__(WinMLModelForFeatureExtraction)
        mock_session = MagicMock()
        mock_session.io_config = {
            "input_names": ["input_ids"],
            "output_names": ["pooler_output"],
        }
        mock_session.run.return_value = {
            "pooler_output": np.zeros((1, 768), dtype=np.float32),
        }
        mock_session.device = "cpu"
        model._session = mock_session
        model.config = MagicMock()
        model._onnx_path = "mock.onnx"
        model._device = "cpu"

        result = model.forward(input_ids=torch.ones(1, 8, dtype=torch.long))
        assert result.last_hidden_state.shape == (1, 1, 768)
