# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for _detect_task_from_config function.

Tests _detect_task_from_config independently from the full resolution flow.
"""

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.loader.task import _detect_task_from_config


class TestDetectTaskFromConfig:
    """Tests for _detect_task_from_config function."""

    def test_known_architecture_returns_correct_task(self):
        """Known architecture returns correct task via TasksManager."""
        config = MagicMock()
        config.architectures = ["ResNetForImageClassification"]

        task = _detect_task_from_config(config)
        assert task == "image-classification"

    def test_bert_architecture_returns_fill_mask(self):
        """BertForMaskedLM returns fill-mask task."""
        config = MagicMock()
        config.architectures = ["BertForMaskedLM"]

        task = _detect_task_from_config(config)
        assert task == "fill-mask"

    def test_missing_architectures_none_raises_error(self):
        """ValueError when config.architectures is None."""
        config = MagicMock()
        config.architectures = None

        with pytest.raises(ValueError, match="no 'architectures' field"):
            _detect_task_from_config(config)

    def test_missing_architectures_empty_list_raises_error(self):
        """ValueError when config.architectures is empty list."""
        config = MagicMock()
        config.architectures = []

        with pytest.raises(ValueError, match="no 'architectures' field"):
            _detect_task_from_config(config)

    def test_non_importable_architecture_raises_error(self):
        """ValueError when architecture cannot be imported from transformers."""
        config = MagicMock()
        config.architectures = ["NonExistentModelClass"]

        with patch("winml.modelkit.loader.task.importlib.import_module") as mock_import:
            mock_transformers = MagicMock()
            del mock_transformers.NonExistentModelClass
            mock_import.return_value = mock_transformers

            with pytest.raises(ValueError, match="Cannot import NonExistentModelClass"):
                _detect_task_from_config(config)

    def test_uses_first_architecture_only(self):
        """Uses architectures[0] when multiple architectures present."""
        config = MagicMock()
        config.architectures = [
            "ResNetForImageClassification",
            "SomeOtherClass",
        ]

        task = _detect_task_from_config(config)
        assert task == "image-classification"
