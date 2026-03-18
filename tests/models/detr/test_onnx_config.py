"""Tests for DETR model configuration.

Tests for DETR_CONFIG (WinMLBuildConfig) which provides conv fusion flags
for ResNet backbone BN folding. DetrIOConfig was removed in favor of
_populate_image_size_from_preprocessor which reads image size from
preprocessor_config.json.
"""

from __future__ import annotations

from winml.modelkit.models.hf.detr import DETR_CONFIG


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
