# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for HF model class mapping that download real models.

Extracted from tests/unit/loader/test_hf_model_class_mapping.py.
These tests require network access. Use `pytest -m "not slow"` to skip them.
"""

import pytest

from winml.modelkit.loader import load_hf_model


@pytest.mark.slow
class TestHFModelClassMappingE2E:
    """End-to-end tests that actually load models."""

    def test_clip_image_feature_extraction_e2e(self):
        """E2E: Load CLIP with image-feature-extraction task."""
        model, _config, _task = load_hf_model(
            "openai/clip-vit-base-patch32",
            task="image-feature-extraction",
        )

        assert model.__class__.__name__ == "CLIPVisionModelWithProjection"

    def test_clip_feature_extraction_e2e(self):
        """E2E: Load CLIP with feature-extraction resolves to text model."""
        model, _config, task = load_hf_model(
            "openai/clip-vit-base-patch32",
            task="feature-extraction",
        )
        assert task == "feature-extraction"
        assert model.__class__.__name__ == "CLIPTextModelWithProjection"

    def test_nsp_e2e(self):
        """E2E: Load BERT with next-sentence-prediction task."""
        model, _config, _task = load_hf_model(
            "prajjwal1/bert-tiny",
            task="next-sentence-prediction",
        )

        # Should be BertForNextSentencePrediction (loaded via AutoModelForNextSentencePrediction)
        assert "NextSentencePrediction" in model.__class__.__name__
