# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for Depth-Anything ONNX export config and input generator.

Depth-Anything's ``backbone_config.image_size`` is the DINOv2 backbone
pretraining resolution (518). When users provide a non-default shape via
``--shape-config`` (e.g. to align ONNX input shape with a non-square dataset
after preprocessing), the explicit ``height`` / ``width`` kwargs must take
precedence. The ``_DepthAnythingVisionInputGenerator`` enforces this priority,
mirroring the pattern used by ``_SegformerVisionInputGenerator``.
"""

from __future__ import annotations

import pytest
from optimum.utils import NormalizedConfig
from transformers import DepthAnythingConfig

from winml.modelkit.export.io import _get_onnx_config  # Testing internal implementation
from winml.modelkit.models.hf.depth_anything import (
    DepthAnythingIOConfig,
    _DepthAnythingVisionInputGenerator,
)


# =============================================================================
# Test Constants
# =============================================================================

BATCH_SIZE = 1
BACKBONE_IMAGE_SIZE = 518  # DINOv2 default in DepthAnythingConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def depth_anything_config():
    """DepthAnythingConfig with default DINOv2 backbone (image_size=518)."""
    return DepthAnythingConfig()


@pytest.fixture()
def normalized_config(depth_anything_config):
    """NormalizedConfig wrapping DepthAnythingConfig.

    Resolves image_size and num_channels from the nested backbone_config
    via dotted-path access — the same pattern used by DepthAnythingIOConfig.
    """
    nc = NormalizedConfig.with_args(
        image_size="backbone_config.image_size",
        num_channels="backbone_config.num_channels",
        allow_new=True,
    )
    return nc(depth_anything_config)


# =============================================================================
# _DepthAnythingVisionInputGenerator — Override priority tests
# =============================================================================


class TestDepthAnythingVisionInputGenerator:
    """Tests for ``_DepthAnythingVisionInputGenerator`` override priority."""

    def test_explicit_height_width_overrides_backbone_image_size(
        self, normalized_config
    ):
        """User-provided height/width override backbone_config.image_size."""
        gen = _DepthAnythingVisionInputGenerator(
            task="depth-estimation",
            normalized_config=normalized_config,
            height=518,
            width=686,
        )

        assert gen.height == 518
        assert gen.width == 686
        assert gen.image_size == (518, 686)

    def test_default_kwargs_keep_backbone_image_size(self, normalized_config):
        """Without explicit height/width, backbone_config.image_size is used."""
        gen = _DepthAnythingVisionInputGenerator(
            task="depth-estimation",
            normalized_config=normalized_config,
        )

        assert gen.height == BACKBONE_IMAGE_SIZE
        assert gen.width == BACKBONE_IMAGE_SIZE

    def test_generated_tensor_uses_override_size(self, normalized_config):
        """``generate()`` produces a tensor with the user-provided shape."""
        gen = _DepthAnythingVisionInputGenerator(
            task="depth-estimation",
            normalized_config=normalized_config,
            batch_size=BATCH_SIZE,
            height=518,
            width=686,
        )

        tensor = gen.generate("pixel_values", framework="pt")
        assert tuple(tensor.shape) == (BATCH_SIZE, 3, 518, 686)


# =============================================================================
# DepthAnythingIOConfig — Registration tests
# =============================================================================


class TestDepthAnythingIOConfig:
    """Tests for DepthAnythingIOConfig ONNX export registration."""

    def test_onnx_config_registered(self, depth_anything_config):
        """DepthAnythingIOConfig is registered for depth-estimation."""
        config = _get_onnx_config(
            depth_anything_config.model_type,
            "depth-estimation",
            depth_anything_config,
        )
        assert isinstance(config, DepthAnythingIOConfig)

    def test_inputs_contain_pixel_values(self, depth_anything_config):
        """Inputs spec includes pixel_values with correct dynamic axes."""
        config = _get_onnx_config(
            depth_anything_config.model_type,
            "depth-estimation",
            depth_anything_config,
        )
        assert "pixel_values" in config.inputs
        assert config.inputs["pixel_values"][0] == "batch_size"

    def test_outputs_contain_predicted_depth(self, depth_anything_config):
        """Outputs spec includes predicted_depth."""
        config = _get_onnx_config(
            depth_anything_config.model_type,
            "depth-estimation",
            depth_anything_config,
        )
        assert "predicted_depth" in config.outputs

    def test_uses_overridable_generator(self):
        """The IO config registers the override-aware generator class."""
        assert (
            _DepthAnythingVisionInputGenerator
            in DepthAnythingIOConfig.DUMMY_INPUT_GENERATOR_CLASSES
        )
