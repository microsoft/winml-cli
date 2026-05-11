# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for Real-ESRGAN PreTrainedModel config and construction."""

from __future__ import annotations

import pytest
import torch
from transformers.modeling_outputs import ImageSuperResolutionOutput

from winml.modelkit.models.hf.esrgan import (
    ESRGANConfig,
    ESRGANForImageSuperResolution,
)


# =============================================================================
# Minimal model params for fast tests
# =============================================================================
FAST_NUM_BLOCK = 2


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def default_config() -> ESRGANConfig:
    """Config with all default values."""
    return ESRGANConfig()


@pytest.fixture(scope="module")
def fast_config() -> ESRGANConfig:
    """Config with minimal blocks for fast instantiation."""
    return ESRGANConfig(num_block=FAST_NUM_BLOCK)


# =============================================================================
# TestESRGANConfig
# =============================================================================


class TestESRGANConfig:
    """Tests for ESRGANConfig (PretrainedConfig subclass)."""

    def test_model_type(self, default_config: ESRGANConfig) -> None:
        """model_type is uppercase 'ESRGAN' to enable HF's case-sensitive
        substring fallback against repo names like 'ai-forever/Real-ESRGAN'."""
        assert default_config.model_type == "ESRGAN"

    def test_default_params(self, default_config: ESRGANConfig) -> None:
        """Default config has expected RRDBNet hyperparameters."""
        assert default_config.num_in_ch == 3
        assert default_config.num_out_ch == 3
        assert default_config.num_feat == 64
        assert default_config.num_block == 23
        assert default_config.num_grow_ch == 32
        assert default_config.scale == 4
        assert default_config.weight_file_format == "RealESRGAN_x{scale}.pth"

    def test_custom_params(self) -> None:
        """Custom overrides propagate correctly."""
        cfg = ESRGANConfig(scale=2, num_block=6)
        assert cfg.scale == 2
        assert cfg.num_block == 6
        # Other defaults unchanged
        assert cfg.num_in_ch == 3
        assert cfg.num_feat == 64

    def test_weight_file_format_renders_default_filename(self) -> None:
        """Default template + scale produces the upstream ``RealESRGAN_xN.pth`` filename."""
        for scale in (2, 4, 8):
            cfg = ESRGANConfig(scale=scale)
            assert cfg.weight_file_format.format(scale=cfg.scale) == (f"RealESRGAN_x{scale}.pth")

    def test_weight_file_format_override(self) -> None:
        """A caller-supplied template is honored and produces the rendered filename."""
        cfg = ESRGANConfig(scale=4, weight_file_format="fork_x{scale}_v2.bin")
        assert cfg.weight_file_format == "fork_x{scale}_v2.bin"
        assert cfg.weight_file_format.format(scale=cfg.scale) == "fork_x4_v2.bin"

    def test_config_serialization_roundtrip(self, tmp_path) -> None:
        """save_pretrained -> from_pretrained preserves all values."""
        original = ESRGANConfig(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=32,
            num_block=4,
            num_grow_ch=16,
            scale=2,
            weight_file_format="custom_{scale}x.bin",
        )
        original.save_pretrained(str(tmp_path))
        loaded = ESRGANConfig.from_pretrained(str(tmp_path))

        assert loaded.model_type == original.model_type
        assert loaded.num_in_ch == original.num_in_ch
        assert loaded.num_out_ch == original.num_out_ch
        assert loaded.num_feat == original.num_feat
        assert loaded.num_block == original.num_block
        assert loaded.num_grow_ch == original.num_grow_ch
        assert loaded.scale == original.scale
        assert loaded.weight_file_format == original.weight_file_format


# =============================================================================
# TestESRGANModelConstruction
# =============================================================================


class TestESRGANModelConstruction:
    """Tests for ESRGANForImageSuperResolution construction and forward."""

    def test_model_creates_from_config(self, fast_config: ESRGANConfig) -> None:
        """Model instantiates without error from config."""
        model = ESRGANForImageSuperResolution(fast_config)
        assert model is not None
        assert model.config.num_block == FAST_NUM_BLOCK

    @pytest.mark.parametrize("scale", [2, 4, 8])
    def test_output_shape_matches_scale(self, scale: int) -> None:
        """Output spatial dims = input spatial dims * scale."""
        cfg = ESRGANConfig(num_block=FAST_NUM_BLOCK, scale=scale)
        model = ESRGANForImageSuperResolution(cfg)
        model.eval()

        h, w = 16, 16
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            out = model(pixel_values=x)

        assert out.reconstruction.shape == (1, 3, h * scale, w * scale)

    def test_output_is_image_super_resolution_output(self, fast_config: ESRGANConfig) -> None:
        """Forward returns ImageSuperResolutionOutput."""
        model = ESRGANForImageSuperResolution(fast_config)
        model.eval()

        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = model(pixel_values=x)

        assert isinstance(out, ImageSuperResolutionOutput)
        assert out.reconstruction is not None

    def test_return_dict_false(self, fast_config: ESRGANConfig) -> None:
        """return_dict=False returns a tuple."""
        model = ESRGANForImageSuperResolution(fast_config)
        model.eval()

        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out = model(pixel_values=x, return_dict=False)

        assert isinstance(out, tuple)
        assert len(out) == 1
        assert out[0].shape[1] == 3  # channels

    def test_save_and_load_pretrained(self, fast_config: ESRGANConfig, tmp_path) -> None:
        """save_pretrained -> from_pretrained roundtrip on a local dir.

        Verifies the overridden ``from_pretrained`` still delegates to
        :class:`PreTrainedModel` for local directories with a ``config.json``.
        """
        model = ESRGANForImageSuperResolution(fast_config)
        model.eval()

        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            original_out = model(pixel_values=x)

        save_dir = str(tmp_path / "esrgan_model")
        model.save_pretrained(save_dir)
        loaded = ESRGANForImageSuperResolution.from_pretrained(save_dir)
        loaded.eval()

        with torch.no_grad():
            loaded_out = loaded(pixel_values=x)

        assert torch.allclose(
            original_out.reconstruction,
            loaded_out.reconstruction,
            atol=1e-6,
        )
