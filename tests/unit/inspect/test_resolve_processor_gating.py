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

from unittest.mock import MagicMock, patch

from winml.modelkit.inspect.resolver import (
    _resolve_processor_from_auto_classes,
    resolve_processor,
)


def _all_filled_hub_result() -> tuple[str, str, str, str, bool, bool]:
    """Strategy 1 returns all four processor types + both config files present."""
    return (
        "BertProcessor",
        "BertTokenizer",
        "BertImageProcessor",
        "BertFeatureExtractor",
        True,
        True,
    )


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
        hub_result = (
            None,
            None,
            "ConvNextImageProcessor",
            "ConvNextFeatureExtractor",
            True,
            True,
        )

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
                return_value=(None, None, None, None, True, True),
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

    def test_missing_preprocessor_config_skips_image_and_feature(self) -> None:
        """preprocessor_config.json absent → skip AutoImageProcessor & AutoFeatureExtractor.

        Text-only models (RoBERTa, BERT, GPT, ...) don't ship a
        preprocessor_config.json. Without this gate, Strategy 2 spends
        ~2s confirming 404s for both AutoImageProcessor and
        AutoFeatureExtractor. The hub_configs helper now reports the
        file's existence so the caller can skip those lookups.
        """
        hub_result = (None, None, None, None, False, True)  # no preprocessor_config

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
            resolve_processor("text/model")

        assert mock_auto.call_count == 1
        kwargs = mock_auto.call_args.kwargs
        assert kwargs["try_processor"] is True
        assert kwargs["try_tokenizer"] is True
        assert kwargs["try_image_processor"] is False, (
            "Must skip AutoImageProcessor when preprocessor_config.json is absent"
        )
        assert kwargs["try_feature_extractor"] is False, (
            "Must skip AutoFeatureExtractor when preprocessor_config.json is absent"
        )


class TestAutoProcessorGatedOnTryProcessor:
    """When ``try_processor=False`` we skip AutoProcessor entirely.

    AutoProcessor.from_pretrained is the most expensive single Auto* call
    (~3.5s warm), so callers that already know ``processor_class`` from
    earlier strategies should not pay for it just to get sub-pieces.
    """

    def test_try_processor_false_skips_autoprocessor(self) -> None:
        """try_processor=False → AutoProcessor.from_pretrained is NOT called."""
        with (
            patch("transformers.AutoProcessor.from_pretrained") as mock_ap,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_at,
            patch("transformers.AutoImageProcessor.from_pretrained") as mock_aip,
            patch("transformers.AutoFeatureExtractor.from_pretrained") as mock_afe,
        ):
            # Make the sub-callers succeed so we can verify they ran instead
            mock_at.return_value = MagicMock(__class__=type("FakeTokenizer", (), {}))
            mock_aip.return_value = MagicMock(__class__=type("FakeImgProc", (), {}))
            mock_afe.return_value = MagicMock(__class__=type("FakeFeatExt", (), {}))

            _resolve_processor_from_auto_classes(
                "some/model",
                try_processor=False,
                try_tokenizer=True,
                try_image_processor=True,
                try_feature_extractor=True,
            )

        assert mock_ap.call_count == 0, (
            "AutoProcessor.from_pretrained must be skipped when try_processor=False"
        )
        # The standalone Auto* calls still run for the fields we need
        assert mock_at.call_count == 1
        assert mock_aip.call_count == 1
        assert mock_afe.call_count == 1

    def test_try_processor_true_still_calls_autoprocessor(self) -> None:
        """try_processor=True → AutoProcessor.from_pretrained runs as before."""
        # Build a fake processor that does NOT supply sub-pieces, so the
        # standalone Auto* calls below still fire for any field we need.
        fake_processor = MagicMock(spec=[])  # no .tokenizer / .image_processor / etc.
        fake_processor.__class__ = type("FakeProcessor", (), {})

        with (
            patch(
                "transformers.AutoProcessor.from_pretrained",
                return_value=fake_processor,
            ) as mock_ap,
            patch("transformers.AutoTokenizer.from_pretrained"),
            patch("transformers.AutoImageProcessor.from_pretrained"),
            patch("transformers.AutoFeatureExtractor.from_pretrained"),
        ):
            _resolve_processor_from_auto_classes(
                "some/model",
                try_processor=True,
                try_tokenizer=False,
                try_image_processor=False,
                try_feature_extractor=False,
            )

        assert mock_ap.call_count == 1


class TestAutoProcessorLeafClassDetection:
    """``AutoProcessor.from_pretrained`` may return a leaf processor.

    For text-only models (RoBERTa, BERT, ...) ``AutoProcessor`` returns
    the tokenizer directly — e.g. ``RobertaTokenizerFast``. Without
    recognising this we would re-load the tokenizer via the standalone
    ``AutoTokenizer.from_pretrained`` below at ~2s of extra cost.
    """

    @staticmethod
    def _make_leaf_instance(class_name: str) -> object:
        """Build an instance whose ``type(obj).__name__`` is ``class_name``.

        Plain instance — no ``.tokenizer`` / ``.image_processor`` /
        ``.feature_extractor`` attributes — so the leaf-class detection
        branch is what matches.
        """
        return type(class_name, (), {})()

    def test_autoprocessor_returns_tokenizer_fills_tokenizer_class(self) -> None:
        """When AutoProcessor returns a *Tokenizer*, tokenizer_class is populated
        and standalone AutoTokenizer is NOT called.
        """
        fake = self._make_leaf_instance("RobertaTokenizerFast")

        with (
            patch("transformers.AutoProcessor.from_pretrained", return_value=fake),
            patch("transformers.AutoTokenizer.from_pretrained") as mock_at,
            patch("transformers.AutoImageProcessor.from_pretrained"),
            patch("transformers.AutoFeatureExtractor.from_pretrained"),
        ):
            proc, tok, _img, _feat = _resolve_processor_from_auto_classes(
                "some/text-model",
                try_processor=True,
                try_tokenizer=True,
                try_image_processor=False,
                try_feature_extractor=False,
            )

        assert proc == "RobertaTokenizerFast"
        assert tok == "RobertaTokenizerFast"
        assert mock_at.call_count == 0, (
            "Standalone AutoTokenizer must be skipped when AutoProcessor "
            "already returned a *Tokenizer* leaf class"
        )

    def test_autoprocessor_returns_image_processor_fills_image_class(self) -> None:
        """AutoProcessor returning a *ImageProcessor* fills image_processor_class."""
        fake = self._make_leaf_instance("ConvNextImageProcessor")

        with (
            patch("transformers.AutoProcessor.from_pretrained", return_value=fake),
            patch("transformers.AutoImageProcessor.from_pretrained") as mock_aip,
        ):
            proc, _, img, _ = _resolve_processor_from_auto_classes(
                "some/vision-model",
                try_processor=True,
                try_tokenizer=False,
                try_image_processor=True,
                try_feature_extractor=False,
            )

        assert proc == "ConvNextImageProcessor"
        assert img == "ConvNextImageProcessor"
        assert mock_aip.call_count == 0

    def test_autoprocessor_returns_feature_extractor_fills_feature_class(self) -> None:
        """AutoProcessor returning a *FeatureExtractor* fills feature_extractor_class."""
        fake = self._make_leaf_instance("Wav2Vec2FeatureExtractor")

        with (
            patch("transformers.AutoProcessor.from_pretrained", return_value=fake),
            patch("transformers.AutoFeatureExtractor.from_pretrained") as mock_afe,
        ):
            proc, _, _, feat = _resolve_processor_from_auto_classes(
                "some/audio-model",
                try_processor=True,
                try_tokenizer=False,
                try_image_processor=False,
                try_feature_extractor=True,
            )

        assert proc == "Wav2Vec2FeatureExtractor"
        assert feat == "Wav2Vec2FeatureExtractor"
        assert mock_afe.call_count == 0

    def test_autoprocessor_with_wrapped_pieces_uses_attributes(self) -> None:
        """Multimodal AutoProcessor (real ProcessorMixin) wins over name suffix."""

        class CLIPTokenizer:
            pass

        class CLIPProcessor:
            def __init__(self) -> None:
                self.tokenizer = CLIPTokenizer()

        with (
            patch(
                "transformers.AutoProcessor.from_pretrained",
                return_value=CLIPProcessor(),
            ),
            patch("transformers.AutoTokenizer.from_pretrained") as mock_at,
        ):
            proc, tok, _, _ = _resolve_processor_from_auto_classes(
                "openai/clip-vit-base-patch32",
                try_processor=True,
                try_tokenizer=True,
                try_image_processor=False,
                try_feature_extractor=False,
            )

        assert proc == "CLIPProcessor"
        assert tok == "CLIPTokenizer"
        assert mock_at.call_count == 0
