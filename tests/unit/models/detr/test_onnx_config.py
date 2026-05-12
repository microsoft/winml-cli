# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for DETR model configuration.

Tests cover:
- DETR_CONFIG (WinMLBuildConfig) conv fusion flags for ResNet BN folding
- ONNX export config override registration for DETR/Table Transformer
- pixel_mask input + dummy input generation for DETR-family exports
"""

from __future__ import annotations

import pytest
import torch
from transformers import DetrConfig, TableTransformerConfig

from winml.modelkit.export import generate_dummy_inputs
from winml.modelkit.export.io import _get_onnx_config  # Testing internal implementation
from winml.modelkit.models.hf.detr import DETR_CONFIG, DetrIOConfig, TableTransformerIOConfig


# =============================================================================
# DETR_CONFIG Tests
# =============================================================================


class TestDetrModelConfig:
    """Tests for DETR_CONFIG (WinMLBuildConfig)."""

    def test_optimization_config(self):
        """Verify conv fusion flags for ResNet backbone BN folding."""
        optim = DETR_CONFIG.optim

        # Conv fusions for BN fold absorption (not autoconf-discoverable)
        assert optim["conv_bn_fusion"] is True
        assert optim["conv_mul_fusion"] is True
        assert optim["conv_add_fusion"] is True


@pytest.fixture(scope="module")
def detr_config():
    """Minimal DetrConfig for testing ONNX I/O config."""
    return DetrConfig(
        num_channels=3,
        d_model=64,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
    )


@pytest.fixture(scope="module")
def table_transformer_config():
    """Minimal TableTransformerConfig for testing ONNX I/O config."""
    return TableTransformerConfig(
        num_channels=3,
        d_model=64,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
    )


class TestDetrFamilyIOConfig:
    """Tests for DETR-family ONNX config pixel_mask export behavior."""

    @pytest.mark.parametrize(
        "model_type,task,config_fixture,expected_class",
        [
            ("detr", "object-detection", "detr_config", DetrIOConfig),
            (
                "table-transformer",
                "object-detection",
                "table_transformer_config",
                TableTransformerIOConfig,
            ),
        ],
        ids=["detr", "table-transformer"],
    )
    def test_onnx_config_registered_with_pixel_mask(
        self, model_type, task, config_fixture, expected_class, request
    ):
        """DETR-family object-detection task resolves to pixel_mask-enabled config."""
        config = request.getfixturevalue(config_fixture)

        onnx_config = _get_onnx_config(model_type, task, config)

        assert isinstance(onnx_config, expected_class)
        assert "pixel_values" in onnx_config.inputs
        assert "pixel_mask" in onnx_config.inputs
        assert onnx_config.inputs["pixel_mask"] == {0: "batch_size", 1: "height", 2: "width"}

    @pytest.mark.parametrize(
        "model_type,config_fixture",
        [
            ("detr", "detr_config"),
            ("table-transformer", "table_transformer_config"),
        ],
        ids=["detr", "table-transformer"],
    )
    def test_generate_dummy_inputs_include_pixel_mask(self, model_type, config_fixture, request):
        """Dummy inputs include pixel_mask aligned with pixel_values height/width."""
        config = request.getfixturevalue(config_fixture)

        inputs = generate_dummy_inputs(
            model_type,
            "object-detection",
            config,
            batch_size=2,
            height=128,
            width=192,
        )

        assert "pixel_values" in inputs
        assert "pixel_mask" in inputs

        pixel_values = inputs["pixel_values"]
        pixel_mask = inputs["pixel_mask"]
        assert pixel_values.shape == (2, 3, 128, 192)
        assert pixel_mask.shape == (2, 128, 192)
        assert pixel_mask.dtype == torch.int64
        assert torch.all(pixel_mask == 1)
