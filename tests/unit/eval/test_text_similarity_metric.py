# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for TextSimilarityMetric (CER + CIDEr)."""

from __future__ import annotations

import pytest

from winml.modelkit.eval.metrics.text_similarity import (
    TextSimilarityMetric,
    _levenshtein,
)


class TestLevenshtein:
    """Cover edge cases of the Levenshtein helper."""

    def test_identical(self) -> None:
        assert _levenshtein("hello", "hello") == 0

    def test_empty_a(self) -> None:
        assert _levenshtein("", "abc") == 3

    def test_empty_b(self) -> None:
        assert _levenshtein("abc", "") == 3

    def test_both_empty(self) -> None:
        assert _levenshtein("", "") == 0

    def test_substitution(self) -> None:
        assert _levenshtein("cat", "bat") == 1  # 1 sub

    def test_insertion(self) -> None:
        assert _levenshtein("cat", "cats") == 1  # 1 ins

    def test_deletion(self) -> None:
        assert _levenshtein("cats", "cat") == 1  # 1 del

    def test_classic_kitten_sitting(self) -> None:
        # Standard wikipedia example: kitten -> sitting is 3 edits.
        assert _levenshtein("kitten", "sitting") == 3


class TestEmptyMetric:
    def test_no_samples_returns_none(self) -> None:
        m = TextSimilarityMetric()
        result = m.compute()
        assert result == {"cer": None, "cider": None, "n_samples": 0}

    def test_empty_references_skipped(self) -> None:
        m = TextSimilarityMetric()
        m.update("anything", [])  # no refs => skipped
        assert m.compute()["n_samples"] == 0


class TestCER:
    """CER edge cases."""

    def test_perfect_match_cer_zero(self) -> None:
        m = TextSimilarityMetric()
        m.update("HELLO WORLD", "HELLO WORLD")
        assert m.compute()["cer"] == 0.0

    def test_one_substitution(self) -> None:
        # "HELLO WORLD" vs "HELLO WORLE": 1 char diff over 11 ref chars = 0.0909
        m = TextSimilarityMetric()
        m.update("HELLO WORLE", "HELLO WORLD")
        cer = m.compute()["cer"]
        assert cer == pytest.approx(1 / 11, abs=1e-4)

    def test_aggregate_cer_across_samples(self) -> None:
        # Sample 1: "abc" vs "abc" -> 0 edits, 3 ref chars
        # Sample 2: "abd" vs "abc" -> 1 edit, 3 ref chars
        # Aggregate CER = 1 / 6 ≈ 0.1667
        m = TextSimilarityMetric()
        m.update("abc", "abc")
        m.update("abd", "abc")
        assert m.compute()["cer"] == pytest.approx(1 / 6, abs=1e-4)

    def test_multi_reference_picks_minimum_distance(self) -> None:
        # Pred matches one of two references exactly => 0 edits.
        m = TextSimilarityMetric()
        m.update("hello", ["world", "hello"])
        assert m.compute()["cer"] == 0.0

    def test_garbage_pred_high_cer(self) -> None:
        # Pred completely unlike ref -> CER >= 1.0
        m = TextSimilarityMetric()
        m.update("xxxxxxxxx", "HELLO")
        cer = m.compute()["cer"]
        assert cer >= 1.0  # at least full replacement cost


class TestCIDEr:
    """Sanity checks for the pycocoevalcap-backed CIDEr metric."""

    def test_cider_perfect_match_positive(self) -> None:
        # With identical pred==ref across multiple samples, CIDEr is positive.
        m = TextSimilarityMetric()
        m.update("a cat sitting on a rock", ["a cat sitting on a rock"])
        m.update("the dog runs in the park", ["the dog runs in the park"])
        m.update("a man riding a bicycle", ["a man riding a bicycle"])
        cider = m.compute()["cider"]
        assert cider is not None
        assert cider > 0

    def test_cider_disjoint_pred_is_zero(self) -> None:
        # Predictions share no n-grams with references.
        m = TextSimilarityMetric()
        m.update("xxx yyy zzz", ["a cat on a rock"])
        m.update("foo bar baz", ["a dog in the park"])
        cider = m.compute()["cider"]
        assert cider == pytest.approx(0.0, abs=1e-4)


class TestComputeShape:
    """compute() returns the expected dict shape."""

    def test_keys_present(self) -> None:
        m = TextSimilarityMetric()
        m.update("hello world", "hello world")
        result = m.compute()
        assert set(result.keys()) == {"cer", "cider", "n_samples"}

    def test_n_samples_counts_updates(self) -> None:
        m = TextSimilarityMetric()
        for _ in range(5):
            m.update("a", "a")
        assert m.compute()["n_samples"] == 5
