# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for CrossEntropyMetric class."""

from __future__ import annotations

import math

import pytest
import torch

from winml.modelkit.eval import CrossEntropyMetric


class TestCrossEntropyMetricBasic:
    """Basic functionality of the incremental CE metric."""

    def test_single_update_known_loss(self) -> None:
        """Hand-computed CE for a trivial 3-class example."""
        metric = CrossEntropyMetric()

        # logits: one masked position, vocab_size=3
        # logits = [2.0, 1.0, 0.0] → softmax = [0.665, 0.245, 0.090]
        # label = 0 → CE = -log(0.665) ≈ 0.4076
        logits = torch.tensor([[2.0, 1.0, 0.0]])  # [1, 3]
        labels = torch.tensor([0])                  # [1]

        metric.update(logits, labels)
        result = metric.compute()

        expected_ce = -math.log(math.exp(2.0) / (math.exp(2.0) + math.exp(1.0) + math.exp(0.0)))
        assert result["cross_entropy"] == pytest.approx(expected_ce, abs=1e-3)

    def test_multiple_updates_accumulate(self) -> None:
        """Two updates should micro-average across all tokens."""
        metric = CrossEntropyMetric()

        # Update 1: 1 masked token
        logits1 = torch.tensor([[10.0, 0.0, 0.0]])  # strongly predicts class 0
        labels1 = torch.tensor([0])
        metric.update(logits1, labels1)

        # Update 2: 1 masked token
        logits2 = torch.tensor([[0.0, 0.0, 10.0]])  # strongly predicts class 2
        labels2 = torch.tensor([2])
        metric.update(logits2, labels2)

        result = metric.compute()
        # Both predictions are confident and correct → CE ≈ 0
        assert result["cross_entropy"] == pytest.approx(0.0, abs=0.1)

    def test_ignored_positions_skipped(self) -> None:
        """Positions with label -100 should not contribute to the metric."""
        metric = CrossEntropyMetric()

        # seq_len=4, vocab_size=3
        logits = torch.tensor([
            [2.0, 0.0, 0.0],  # position 0: masked
            [0.0, 0.0, 0.0],  # position 1: not masked (label=-100)
            [0.0, 0.0, 0.0],  # position 2: not masked (label=-100)
            [0.0, 2.0, 0.0],  # position 3: masked
        ])
        labels = torch.tensor([0, -100, -100, 1])

        metric.update(logits, labels)
        assert metric.total_tokens == 2

    def test_all_ignored_no_contribution(self) -> None:
        """If all labels are -100, update should be a no-op."""
        metric = CrossEntropyMetric()

        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        labels = torch.tensor([-100, -100])

        metric.update(logits, labels)
        assert metric.total_tokens == 0


class TestCrossEntropyMetricOutputFormat:
    """Verify output dict structure and types."""

    def test_output_keys(self) -> None:
        metric = CrossEntropyMetric()
        metric.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
        result = metric.compute()

        assert "cross_entropy" in result

    def test_output_types(self) -> None:
        metric = CrossEntropyMetric()
        metric.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
        result = metric.compute()

        assert isinstance(result["cross_entropy"], float)

    def test_values_rounded_to_4_decimals(self) -> None:
        metric = CrossEntropyMetric()
        metric.update(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([0]))
        result = metric.compute()

        ce_str = str(result["cross_entropy"])
        # At most 4 decimal places
        if "." in ce_str:
            assert len(ce_str.split(".")[1]) <= 4


class TestCrossEntropyMetricReset:
    """Verify reset clears accumulated state."""

    def test_reset_clears_state(self) -> None:
        metric = CrossEntropyMetric()
        metric.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
        metric.reset()

        assert metric._total_loss == 0.0
        assert metric.total_tokens == 0

    def test_compute_after_reset_raises(self) -> None:
        metric = CrossEntropyMetric()
        metric.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
        metric.reset()

        with pytest.raises(ValueError, match="No masked tokens"):
            metric.compute()


class TestCrossEntropyMetricEdgeCases:
    """Edge cases and error handling."""

    def test_no_updates_raises(self) -> None:
        metric = CrossEntropyMetric()
        with pytest.raises(ValueError, match="No masked tokens"):
            metric.compute()

    def test_large_batch_micro_averages(self) -> None:
        """Micro-average: every token weighted equally regardless of sample."""
        metric = CrossEntropyMetric()

        # Sample 1: 1 masked token, CE ≈ 0 (correct prediction)
        metric.update(
            torch.tensor([[10.0, 0.0]]),
            torch.tensor([0]),
        )
        # Sample 2: 3 masked tokens, CE is higher (wrong predictions)
        logits = torch.tensor([
            [0.0, 10.0],  # predicts class 1
            [0.0, 10.0],  # predicts class 1
            [0.0, 10.0],  # predicts class 1
        ])
        labels = torch.tensor([0, 0, 0])  # but truth is class 0
        metric.update(logits, labels)

        assert metric.total_tokens == 4
        # Mean CE should be dominated by the 3 wrong predictions
        result = metric.compute()
        assert result["cross_entropy"] > 1.0
