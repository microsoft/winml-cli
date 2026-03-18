# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Attention pattern family cross-matching and rewriting tests.

Verifies that ExpandedAttention and TransposeAttention patterns do not
cross-match, and that BERT Tiny attention patterns can be rewritten.
"""

import numpy as np
import onnx
import pytest

from winml.modelkit.onnx.domains import ONNXDomain
from winml.modelkit.pattern import (
    ExpandedAttentionPattern,
    PatternMatcher,
    TransposeAttentionPattern,
)

from .conftest import TEST_DOMAIN_VERSIONS


_OPSET_23_DOMAIN_VERSIONS = {ONNXDomain.AI_ONNX: 23}


def _create_expanded_attention_model(dtype=np.float32):
    head_size = 16
    inputs = {
        "Q": np.random.randn(1, 2, 8, head_size).astype(dtype),
        "K": np.random.randn(1, 8, 2, head_size).astype(dtype),
        "V": np.random.randn(1, 2, 8, head_size).astype(dtype),
        "attn_mask": np.random.randn(1, 2, 8, 8).astype(dtype),
    }
    attributes = {"scale": 1.0 / head_size}
    is_constant_map = {"Q": False, "K": False, "V": False, "attn_mask": False}
    output_dtypes = ["tensor(float)" if dtype == np.float32 else "tensor(float16)"]
    return ExpandedAttentionPattern().get_onnx_model(
        inputs, attributes, is_constant_map, output_dtypes, TEST_DOMAIN_VERSIONS
    )


def _create_transpose_attention_model(dtype=np.float32, scale=None):
    inputs = {
        "Q": np.random.randn(1, 2, 8, 16).astype(dtype),
        "K": np.random.randn(1, 8, 2, 16).astype(dtype),
        "V": np.random.randn(1, 2, 8, 16).astype(dtype),
        "attn_mask": np.random.randn(1, 2, 8, 8).astype(dtype),
    }
    is_constant_map = {"Q": False, "K": False, "V": False, "attn_mask": False}
    output_dtypes = ["tensor(float)" if dtype == np.float32 else "tensor(float16)"]
    attributes = {"scale": scale} if scale is not None else {}
    return TransposeAttentionPattern().get_onnx_model(
        inputs, attributes, is_constant_map, output_dtypes, _OPSET_23_DOMAIN_VERSIONS
    )


class TestAttentionPatternCrossMatching:
    """ExpandedAttention and TransposeAttention should not cross-match."""

    def test_expanded_attention_does_not_match_transpose_pattern(self) -> None:
        model = _create_expanded_attention_model()
        matcher = PatternMatcher(model)
        matcher.register_pattern(TransposeAttentionPattern())
        assert len(matcher.match()) == 0, (
            "TransposeAttention should not match ExpandedAttention model"
        )

    def test_transpose_attention_does_not_match_expanded_pattern(self) -> None:
        model = _create_transpose_attention_model()
        matcher = PatternMatcher(model)
        matcher.register_pattern(ExpandedAttentionPattern())
        assert len(matcher.match()) == 0, (
            "ExpandedAttention should not match TransposeAttention model"
        )

    def test_both_patterns_registered_only_correct_one_matches(self) -> None:
        model = _create_expanded_attention_model()
        matcher = PatternMatcher(model)
        matcher.register_pattern(ExpandedAttentionPattern())
        matcher.register_pattern(TransposeAttentionPattern())
        results = matcher.match()

        assert len(results) == 1
        assert (
            type(results[0].skeleton_match_result.pattern).__name__
            == "ExpandedAttentionPattern"
        )


class TestBertTinyAttentionPatternRewriting:
    """Match and rewrite attention patterns in BERT Tiny model."""

    @pytest.fixture
    def bert_tiny_model(self):
        from pathlib import Path

        fixtures = Path(__file__).parent / "../../../fixtures"
        model_path = fixtures / "nsp_b0ee7fae871bae40_opt_opset23.onnx"
        if not model_path.exists():
            pytest.skip(f"BERT Tiny model not found: {model_path}")

        model = onnx.load(str(model_path))

        # Keep only opset imports for domains actually used by nodes
        used_domains = {n.domain for n in model.graph.node}
        opset_imports = [
            opset for opset in model.opset_import if opset.domain in used_domains
        ]
        del model.opset_import[:]
        model.opset_import.extend(opset_imports)

        return model

    def test_bert_tiny_has_expected_attention_patterns(self, bert_tiny_model) -> None:
        pattern = ExpandedAttentionPattern()
        matcher = PatternMatcher(bert_tiny_model)
        matcher.register_pattern(pattern)
        results = matcher.match()

        assert len(results) == 2, (
            f"Expected 2 ExpandedAttentionPattern matches, found {len(results)}"
        )

    def test_bert_tiny_attention_patterns_are_removable(self, bert_tiny_model) -> None:
        pattern = ExpandedAttentionPattern()
        matcher = PatternMatcher(bert_tiny_model)
        matcher.register_pattern(pattern)
        for i, result in enumerate(matcher.match()):
            assert result.skeleton_match_result.removable, f"Match {i} is not removable"

    def test_bert_tiny_rewrite_to_transpose_attention(self, bert_tiny_model) -> None:
        from winml.modelkit.pattern import PatternRewriter

        expanded_pattern = ExpandedAttentionPattern()
        matcher = PatternMatcher(bert_tiny_model)
        matcher.register_pattern(expanded_pattern)
        results = matcher.match()
        assert len(results) == 2

        rewriter = PatternRewriter(bert_tiny_model)
        rewritten_model = rewriter.rewrite([(results, TransposeAttentionPattern)])

        attention_count = sum(
            1 for n in rewritten_model.graph.node if n.op_type == "Attention"
        )
        assert attention_count == 2

    def test_bert_tiny_rewritten_model_has_transpose_attention(
        self, bert_tiny_model
    ) -> None:
        from winml.modelkit.pattern import PatternRewriter

        expanded_pattern = ExpandedAttentionPattern()
        matcher = PatternMatcher(bert_tiny_model)
        matcher.register_pattern(expanded_pattern)
        results = matcher.match()

        rewriter = PatternRewriter(bert_tiny_model)
        rewritten_model = rewriter.rewrite([(results, TransposeAttentionPattern)])

        new_matcher = PatternMatcher(rewritten_model)
        new_matcher.register_pattern(TransposeAttentionPattern())
        new_results = new_matcher.match()

        assert len(new_results) == 2

    def test_bert_tiny_rewritten_model_no_expanded_patterns(
        self, bert_tiny_model
    ) -> None:
        from winml.modelkit.pattern import PatternRewriter

        expanded_pattern = ExpandedAttentionPattern()
        matcher = PatternMatcher(bert_tiny_model)
        matcher.register_pattern(expanded_pattern)
        results = matcher.match()

        rewriter = PatternRewriter(bert_tiny_model)
        rewritten_model = rewriter.rewrite([(results, TransposeAttentionPattern)])

        new_matcher = PatternMatcher(rewritten_model)
        new_matcher.register_pattern(expanded_pattern)
        assert len(new_matcher.match()) == 0
