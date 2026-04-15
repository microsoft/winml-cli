# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Tests for WinMLModelForSequenceClassification.

Tests the sequence classification model in modelkit/models/winml/sequence_classification.py
following the design specifications in docs/design/automodel/CORELOOP.md.

Acceptance Criteria (from design):
- AC-1: Class exists and inherits from WinMLPreTrainedModel
- AC-2: forward() accepts input_ids, attention_mask, token_type_ids
- AC-3: Returns SequenceClassifierOutput with logits
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch


def create_mock_model(num_labels: int = 2):
    """Create a WinMLModelForSequenceClassification with mocked session.

    Sets up all attributes that the base class expects:
    - _session with io_config, run()
    - config with num_labels
    - _onnx_path, _device
    """
    from winml.modelkit.models.winml import WinMLModelForSequenceClassification

    model = WinMLModelForSequenceClassification.__new__(WinMLModelForSequenceClassification)
    mock_session = MagicMock()
    mock_session.run.return_value = {"logits": np.random.randn(1, num_labels).astype(np.float32)}
    mock_session.io_config = {
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
        "output_names": ["logits"],
    }
    mock_session.device = "cpu"
    model._session = mock_session
    model.config = MagicMock()
    model.config.num_labels = num_labels
    model._onnx_path = "mock.onnx"
    model._device = "cpu"
    return model


class TestWinMLModelForSequenceClassificationBasic:
    """Basic functionality tests."""

    def test_class_exists(self):
        """Test that the class exists and is importable."""
        from winml.modelkit.models.winml import WinMLModelForSequenceClassification

        assert WinMLModelForSequenceClassification is not None

    def test_inherits_from_base(self):
        """Test class inherits from WinMLPreTrainedModel."""
        from winml.modelkit.models.winml import (
            WinMLModelForSequenceClassification,
            WinMLPreTrainedModel,
        )

        assert issubclass(WinMLModelForSequenceClassification, WinMLPreTrainedModel)

    def test_has_forward_method(self):
        """Test class has forward method."""
        from winml.modelkit.models.winml import WinMLModelForSequenceClassification

        assert hasattr(WinMLModelForSequenceClassification, "forward")
        assert callable(WinMLModelForSequenceClassification.forward)


class TestForwardMethod:
    """Test forward() method."""

    def test_forward_accepts_input_ids(self):
        """AC-2: forward() accepts input_ids."""
        model = create_mock_model()

        input_ids = torch.randint(0, 30522, (1, 128))
        model.forward(input_ids=input_ids)

        model._session.run.assert_called_once()

    def test_forward_accepts_attention_mask(self):
        """AC-2: forward() accepts attention_mask."""
        model = create_mock_model()

        input_ids = torch.randint(0, 30522, (1, 128))
        attention_mask = torch.ones((1, 128), dtype=torch.long)
        model.forward(input_ids=input_ids, attention_mask=attention_mask)

        model._session.run.assert_called()

    def test_forward_accepts_token_type_ids(self):
        """AC-2: forward() accepts token_type_ids."""
        model = create_mock_model()

        input_ids = torch.randint(0, 30522, (1, 128))
        attention_mask = torch.ones((1, 128), dtype=torch.long)
        token_type_ids = torch.zeros((1, 128), dtype=torch.long)

        model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        model._session.run.assert_called()

    def test_forward_returns_sequence_classifier_output(self):
        """AC-3: forward() returns SequenceClassifierOutput."""
        model = create_mock_model()

        input_ids = torch.randint(0, 30522, (1, 128))
        output = model.forward(input_ids=input_ids)

        # Should have logits
        assert hasattr(output, "logits")

    def test_forward_logits_shape(self):
        """AC-3: Output logits have correct shape."""
        model = create_mock_model(num_labels=2)

        input_ids = torch.randint(0, 30522, (1, 128))
        output = model.forward(input_ids=input_ids)

        # Logits shape should be (batch_size, num_labels)
        assert output.logits.shape == (1, 2)

    def test_forward_loss_is_none(self):
        """forward() does not compute loss (thin wrapper)."""
        model = create_mock_model()

        input_ids = torch.randint(0, 30522, (1, 128))
        output = model.forward(input_ids=input_ids)

        # Thin wrapper does not compute loss
        assert output.loss is None


class TestProperties:
    """Test model properties."""

    def test_num_labels_from_config(self):
        """num_labels property reads from config."""
        model = create_mock_model(num_labels=2)
        model.config.num_labels = 2

        assert model.num_labels == 2

    def test_device_property(self):
        """device property returns current device."""
        model = create_mock_model()

        assert model.device == "cpu"

    def test_dtype_property(self):
        """dtype property returns float32."""
        model = create_mock_model()

        assert model.dtype == torch.float32
