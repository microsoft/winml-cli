# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for `inspect.resolver.resolve_processor`'s Strategy-2 gating.

Strategy 2 (Auto* class instantiation) is the most expensive part of
``resolve_processor`` — each Auto* call does its own Hub I/O and class
init (AutoProcessor and AutoFeatureExtractor can each cost several
seconds on warm cache). These tests pin the gating logic:

* Strategy 2 must be skipped entirely when Strategies 0+1 populated all
  four processor fields.
* When some fields are still unknown, only the relevant Auto* lookups
  may fire; the ones whose field is already set must be skipped.
"""

from __future__ import annotations

from unittest.mock import patch

from winml.modelkit.inspect.resolver import resolve_processor


def _all_filled_hub_result() -> tuple[str, str, str, str]:
    """Strategy 1 returns all four processor types."""
    return ("BertProcessor", "BertTokenizer", "BertImageProcessor", "BertFeatureExtractor")


class TestResolveProcessorStrategy2Gating:
    def test_strategy2_skipped_when_all_fields_filled_by_strategy1(self) -> None:
        """All four classes already known → Strategy 2 must not run at all."""
        with (
            patch(
                "winml.modelkit.inspect.resolver._resolve_processor_from_hub_configs",
                return_value=_all_filled_hub_result(),
            ),
            patch(
                "winml.modelkit.inspect.resolver._resolve_processor_from_auto_classes",
            ) as mock_auto,
        ):
            info = resolve_processor("some/model", model_type="bert")

        assert mock_auto.call_count == 0, (
            "Strategy 2 must be skipped when Strategies 0+1 already populated all fields"
        )
        # Values must come from hub_config strategy
        assert info.processor_class == "BertProcessor"
        assert info.tokenizer_class == "BertTokenizer"
        assert info.image_processor_class == "BertImageProcessor"
        assert info.feature_extractor_class == "BertFeatureExtractor"

    def test_strategy2_called_with_per_field_flags(self) -> None:
        """Only the fields still missing after Strategy 1 should have try_*=True."""
        # Strategy 1 fills only image_processor and feature_extractor.
        hub_result = (None, None, "ConvNextImageProcessor", "ConvNextFeatureExtractor")

        with (
            patch(
                "winml.modelkit.inspect.resolver._resolve_processor_from_hub_configs",
                return_value=hub_result,
            ),
            patch(
                "winml.modelkit.inspect.resolver._resolve_processor_from_auto_classes",
                return_value=(None, None, None, None),
            ) as mock_auto,
        ):
            resolve_processor("some/model", model_type="resnet")

        assert mock_auto.call_count == 1
        kwargs = mock_auto.call_args.kwargs
        assert kwargs["try_processor"] is True
        assert kwargs["try_tokenizer"] is True
        assert kwargs["try_image_processor"] is False
        assert kwargs["try_feature_extractor"] is False

    def test_strategy2_runs_when_nothing_filled(self) -> None:
        """Empty Strategy-1 result → Strategy 2 runs with every flag True."""
        with (
            patch(
                "winml.modelkit.inspect.resolver._resolve_processor_from_hub_configs",
                return_value=(None, None, None, None),
            ),
            # Block Strategy 0 (HF registry) by passing no model_type below
            patch(
                "winml.modelkit.inspect.resolver._resolve_processor_from_auto_classes",
                return_value=("P", "T", "I", "F"),
            ) as mock_auto,
        ):
            info = resolve_processor("some/model")

        assert mock_auto.call_count == 1
        kwargs = mock_auto.call_args.kwargs
        assert kwargs["try_processor"] is True
        assert kwargs["try_tokenizer"] is True
        assert kwargs["try_image_processor"] is True
        assert kwargs["try_feature_extractor"] is True
        assert info.processor_class == "P"
        assert info.feature_extractor_class == "F"
