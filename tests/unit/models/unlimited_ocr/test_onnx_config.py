# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the Unlimited-OCR (DeepSeek-OCR family) vision-tower ONNX config.

Unlimited-OCR is a ``trust_remote_code`` VLM whose full ``forward`` is a
generative pipeline (out of scope for ONNX export). This module registers only
the pure-tensor vision sub-graph (SAM -> CLIP -> projector) under the
``feature-extraction`` task. These tests are network-free: they validate the
registry wiring and the pinned 1024x1024 dummy-input contract without loading
the 6.67 GB checkpoint.
"""

from __future__ import annotations

from transformers import PretrainedConfig

# Import triggers ONNX config registration
import winml.modelkit.models  # noqa: F401
from winml.modelkit.export import generate_dummy_inputs
from winml.modelkit.models.hf.unlimited_ocr import (
    MODEL_CLASS_MAPPING,
    UnlimitedOCRVisionIOConfig,
    UnlimitedOCRVisionTowerWrapper,
)


# =============================================================================
# Test Constants
# =============================================================================

VISION_NUM_CHANNELS = 3
VISION_IMAGE_SIZE = 1024


# =============================================================================
# MODEL_CLASS_MAPPING — routing to the vision-tower wrapper
# =============================================================================


class TestUnlimitedOCRModelClassMapping:
    """The feature-extraction task routes to the vision-tower wrapper."""

    def test_registered_in_local_mapping(self):
        """Local sub-dict binds the wrapper to feature-extraction."""
        key = ("unlimited-ocr", "feature-extraction")
        assert key in MODEL_CLASS_MAPPING
        assert MODEL_CLASS_MAPPING[key] is UnlimitedOCRVisionTowerWrapper

    def test_aggregated_into_hf_mapping(self):
        """Entry is merged into the package-level aggregated mapping."""
        from winml.modelkit.models import HF_MODEL_CLASS_MAPPING

        key = ("unlimited-ocr", "feature-extraction")
        assert key in HF_MODEL_CLASS_MAPPING
        assert HF_MODEL_CLASS_MAPPING[key] is UnlimitedOCRVisionTowerWrapper

    def test_wrapper_exposes_from_pretrained(self):
        """Loader dispatches via a from_pretrained classmethod."""
        assert hasattr(UnlimitedOCRVisionTowerWrapper, "from_pretrained")


# =============================================================================
# OnnxConfig — pinned 1024x1024 dummy input + IO contract
# =============================================================================


class TestUnlimitedOCRVisionIOConfig:
    """The from-scratch OnnxConfig pins geometry and declares a single output."""

    def test_input_contract(self):
        config = UnlimitedOCRVisionIOConfig(PretrainedConfig(), task="feature-extraction")
        assert list(config.inputs.keys()) == ["pixel_values"]
        assert config.inputs["pixel_values"] == {0: "batch_size"}

    def test_output_contract(self):
        config = UnlimitedOCRVisionIOConfig(PretrainedConfig(), task="feature-extraction")
        assert list(config.outputs.keys()) == ["image_embeds"]
        assert config.outputs["image_embeds"] == {0: "batch_size"}

    def test_dummy_inputs_pinned_geometry(self):
        """generate_dummy_inputs emits a fixed [1, 3, 1024, 1024] tensor."""
        inputs = generate_dummy_inputs(
            "unlimited-ocr", "feature-extraction", PretrainedConfig()
        )
        assert set(inputs.keys()) == {"pixel_values"}
        pv = inputs["pixel_values"]
        assert tuple(pv.shape) == (1, VISION_NUM_CHANNELS, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)
