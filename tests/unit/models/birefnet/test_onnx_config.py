# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for BiRefNet ONNX export config."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

# Import triggers ONNX config registration.
import winml.modelkit.models  # noqa: F401
from winml.modelkit.export import generate_dummy_inputs
from winml.modelkit.export.io import _get_onnx_config  # Testing internal implementation
from winml.modelkit.models.hf.birefnet import (
    BIREFNET_CONFIG,
    BIREFNET_DEFAULT_IMAGE_SIZE,
    BIREFNET_MODEL_TYPE,
    MODEL_CLASS_MAPPING,
    BiRefNetImageSegmentationWrapper,
    BiRefNetIOConfig,
)


class BiRefNetTestConfig(PretrainedConfig):
    """Minimal config matching ZhengPeng7/BiRefNet metadata."""

    model_type = BIREFNET_MODEL_TYPE

    def __init__(self) -> None:
        super().__init__()
        self.architectures = ["BiRefNet"]
        self.num_channels = 3


class TinyBiRefNet(torch.nn.Module):
    """Small stand-in for BiRefNet's pixel_values-only forward."""

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        return [pixel_values.mean(dim=1, keepdim=True)]


class TinyMultiScaleBiRefNet(torch.nn.Module):
    """Small stand-in for BiRefNet's multi-scale output list."""

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x[:, :1], x[:, :1] + 1]


@pytest.fixture(scope="module")
def birefnet_config() -> BiRefNetTestConfig:
    """Return a minimal BiRefNet config."""
    return BiRefNetTestConfig()


class TestBiRefNetIOConfig:
    """BiRefNet ONNX export registration and I/O specs."""

    def test_onnx_config_registered(self, birefnet_config: BiRefNetTestConfig) -> None:
        """BiRefNetIOConfig is registered for image-segmentation."""
        config = _get_onnx_config(
            birefnet_config.model_type,
            "image-segmentation",
            birefnet_config,
        )

        assert isinstance(config, BiRefNetIOConfig)

    def test_inputs_contain_pixel_values(self, birefnet_config: BiRefNetTestConfig) -> None:
        """Input spec includes pixel_values with spatial axes."""
        config = _get_onnx_config(
            birefnet_config.model_type,
            "image-segmentation",
            birefnet_config,
        )

        assert config.inputs == {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}
        }

    def test_outputs_contain_pred_masks(self, birefnet_config: BiRefNetTestConfig) -> None:
        """Output spec exposes the final mask as pred_masks."""
        config = _get_onnx_config(
            birefnet_config.model_type,
            "image-segmentation",
            birefnet_config,
        )

        assert "pred_masks" in config.outputs

    def test_dummy_inputs_use_documented_resolution(
        self, birefnet_config: BiRefNetTestConfig
    ) -> None:
        """Dummy inputs default to the model card's 1024x1024 inference size."""
        inputs = generate_dummy_inputs(
            birefnet_config.model_type,
            "image-segmentation",
            birefnet_config,
            batch_size=1,
        )

        assert inputs["pixel_values"].shape == (
            1,
            3,
            BIREFNET_DEFAULT_IMAGE_SIZE,
            BIREFNET_DEFAULT_IMAGE_SIZE,
        )

    def test_shape_override_is_preserved(self, birefnet_config: BiRefNetTestConfig) -> None:
        """Explicit shape overrides can shrink export smoke tests."""
        inputs = generate_dummy_inputs(
            birefnet_config.model_type,
            "image-segmentation",
            birefnet_config,
            batch_size=1,
            height=128,
            width=160,
        )

        assert inputs["pixel_values"].shape == (1, 3, 128, 160)

    def test_dummy_inputs_can_forward(self, birefnet_config: BiRefNetTestConfig) -> None:
        """Generated dummy inputs match BiRefNet's pixel_values-only forward."""
        model = TinyBiRefNet().eval()
        inputs = generate_dummy_inputs(
            birefnet_config.model_type,
            "image-segmentation",
            birefnet_config,
            batch_size=1,
            height=32,
            width=32,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs[-1].shape == (1, 1, 32, 32)


class TestBiRefNetModelClassMapping:
    """BiRefNet task routing."""

    def test_maps_to_image_segmentation_auto_class(self) -> None:
        """image-segmentation routes to the BiRefNet export wrapper."""
        assert MODEL_CLASS_MAPPING[(BIREFNET_MODEL_TYPE, "image-segmentation")] is (
            BiRefNetImageSegmentationWrapper
        )

    def test_declares_default_task_sentinel(self) -> None:
        """The sentinel makes image-segmentation the default BiRefNet task."""
        assert MODEL_CLASS_MAPPING[(BIREFNET_MODEL_TYPE, None)] is BiRefNetImageSegmentationWrapper

    def test_model_type_matches_hf_config_normalization(self) -> None:
        """HF model_type normalization matches the registry key."""
        hf_config = SimpleNamespace(model_type="SegformerForSemanticSegmentation")

        assert hf_config.model_type.lower().replace("_", "-") == BIREFNET_MODEL_TYPE

    def test_mixed_case_hf_model_type_resolves_registered_config(self) -> None:
        """The HF-published mixed-case model_type resolves to the registered config."""
        hf_config = BiRefNetTestConfig()
        hf_config.model_type = "SegformerForSemanticSegmentation"

        config = _get_onnx_config(
            hf_config.model_type,
            "image-segmentation",
            hf_config,
        )

        assert isinstance(config, BiRefNetIOConfig)


class TestBiRefNetWrapper:
    """BiRefNet export wrapper behavior."""

    def test_wrapper_uses_pixel_values_and_final_scale_output(
        self, birefnet_config: BiRefNetTestConfig
    ) -> None:
        """Wrapper exposes pixel_values and returns the last scale prediction."""
        model = BiRefNetImageSegmentationWrapper(TinyMultiScaleBiRefNet(), birefnet_config)
        pixel_values = torch.zeros(1, 3, 8, 8)

        output = model(pixel_values)

        assert output.shape == (1, 1, 8, 8)
        assert torch.all(output == 1)


class TestBiRefNetBuildConfig:
    """BiRefNet build defaults."""

    def test_uses_dynamo_and_opset_19_for_deform_conv(self) -> None:
        """BiRefNet needs dynamo export and ONNX DeformConv from opset 19."""
        assert BIREFNET_CONFIG.export is not None
        assert BIREFNET_CONFIG.export.dynamo is True
        assert BIREFNET_CONFIG.export.opset_version == 19
