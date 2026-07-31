# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for zero-shot-classification calibration datasets."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import torch
from datasets.features import ClassLabel, Value


class _FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.features = {
            "text": Value("string"),
            "label": ClassLabel(names=["World", "Sports", "Business", "Sci/Tech"]),
        }

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._rows[idx]

    def select(self, indices: list[int]) -> _FakeDataset:
        return _FakeDataset([self._rows[i] for i in indices])


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, premise: str, hypothesis: str, **_: Any) -> dict[str, torch.Tensor]:
        self.calls.append((premise, hypothesis))
        return {
            "input_ids": torch.ones(1, 8, dtype=torch.int64),
            "attention_mask": torch.ones(1, 8, dtype=torch.int64),
        }


class TestZeroShotClassificationDataset:
    @patch("winml.modelkit.datasets.zero_shot_classification.load_dataset")
    @patch("winml.modelkit.datasets.zero_shot_classification.AutoTokenizer")
    def test_defaults_use_ag_news_sentence_pairs(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        from winml.modelkit.datasets import ZeroShotClassificationDataset
        from winml.modelkit.datasets.zero_shot_classification import (
            DEFAULT_ZERO_SHOT_CLASSIFICATION_DATASET,
            DEFAULT_ZERO_SHOT_CLASSIFICATION_SPLIT,
        )

        tokenizer = _FakeTokenizer()
        mock_tokenizer_cls.from_pretrained.return_value = tokenizer
        mock_load_dataset.return_value = _FakeDataset(
            [{"text": "Markets rallied after the earnings report.", "label": 2}]
        )

        dataset = ZeroShotClassificationDataset(
            model_name="cross-encoder/nli-deberta-v3-small",
            max_samples=4,
        )

        assert dataset.dataset_name == DEFAULT_ZERO_SHOT_CLASSIFICATION_DATASET
        assert dataset.data_split == DEFAULT_ZERO_SHOT_CLASSIFICATION_SPLIT
        mock_load_dataset.assert_called_once_with(
            DEFAULT_ZERO_SHOT_CLASSIFICATION_DATASET,
            split=DEFAULT_ZERO_SHOT_CLASSIFICATION_SPLIT,
        )
        assert tokenizer.calls == [
            ("Markets rallied after the earnings report.", "This text is about World."),
            ("Markets rallied after the earnings report.", "This text is about Sports."),
            ("Markets rallied after the earnings report.", "This text is about Business."),
            ("Markets rallied after the earnings report.", "This text is about Sci/Tech."),
        ]
        assert len(dataset) == 4
        assert set(dataset[0]) == {"input_ids", "attention_mask"}

    def test_zero_shot_task_uses_specialized_dataset(self) -> None:
        from winml.modelkit.datasets import TASK_DATASET_MAPPING, ZeroShotClassificationDataset

        assert TASK_DATASET_MAPPING["zero-shot-classification"] is ZeroShotClassificationDataset
