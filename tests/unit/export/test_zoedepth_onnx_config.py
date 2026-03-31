# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for ZoeDepth ONNX config registration and I/O specs.

Verifies that ZoeDepthIOConfig is correctly registered with Optimum's
TasksManager for depth-estimation, and that I/O specs match the expected
tensors:

- Inputs: pixel_values
- Outputs: predicted_depth (always), domain_logits (multi-bin only)

domain_logits is included only when the model has multiple bin configurations
(e.g. NYU + KITTI).  A single-domain model omits it.

See also: modelkit/models/hf/zoedepth.py
"""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager

# Trigger OnnxConfig registration with TasksManager
import winml.modelkit.models  # noqa: F401
from winml.modelkit.export.io import resolve_io_specs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zoedepth_multi_bin_config():
    """ZoeDepthConfig with two bin configurations (NYU + KITTI style)."""
    from transformers import BeitConfig, ZoeDepthConfig

    backbone = BeitConfig(
        image_size=32,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
    )
    return ZoeDepthConfig(
        backbone_config=backbone,
        bin_configurations=[
            {"n_bins": 64, "min_depth": 0.001, "max_depth": 10.0},
            {"n_bins": 64, "min_depth": 0.001, "max_depth": 80.0},
        ],
    )


@pytest.fixture(scope="module")
def zoedepth_single_bin_config():
    """ZoeDepthConfig with a single bin configuration (NYU-only style)."""
    from transformers import BeitConfig, ZoeDepthConfig

    backbone = BeitConfig(
        image_size=32,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
    )
    return ZoeDepthConfig(
        backbone_config=backbone,
        bin_configurations=[
            {"n_bins": 64, "min_depth": 0.001, "max_depth": 10.0},
        ],
    )


# =============================================================================
# Registration verification
# =============================================================================


class TestZoeDepthRegistration:
    """Verify ZoeDepthIOConfig is registered for depth-estimation."""

    def test_zoedepth_config_registered(self) -> None:
        """ZoeDepthIOConfig must be registered for depth-estimation."""
        config_constructor = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type="zoedepth",
            task="depth-estimation",
            library_name="transformers",
        )
        actual_class_name = config_constructor.func.__name__
        assert actual_class_name == "ZoeDepthIOConfig", (
            f"Expected ZoeDepthIOConfig for zoedepth/depth-estimation, "
            f"got {actual_class_name}. Registration may have failed."
        )


# =============================================================================
# I/O verification
# =============================================================================


class TestZoeDepthIOSpecs:
    """Verify I/O specs for ZoeDepth depth estimation."""

    def test_inputs_contain_pixel_values(self, zoedepth_multi_bin_config) -> None:
        """Inputs must include pixel_values."""
        specs = resolve_io_specs("zoedepth", "depth-estimation", zoedepth_multi_bin_config)
        assert "pixel_values" in specs["input_names"]

    def test_outputs_contain_predicted_depth_multi_bin(self, zoedepth_multi_bin_config) -> None:
        """Multi-bin outputs must include predicted_depth."""
        specs = resolve_io_specs("zoedepth", "depth-estimation", zoedepth_multi_bin_config)
        assert "predicted_depth" in specs["output_names"]

    def test_outputs_contain_domain_logits_multi_bin(self, zoedepth_multi_bin_config) -> None:
        """Multi-bin config must include domain_logits for head selection."""
        specs = resolve_io_specs("zoedepth", "depth-estimation", zoedepth_multi_bin_config)
        assert "domain_logits" in specs["output_names"], (
            "domain_logits must be present when multiple bin configurations exist."
        )

    def test_outputs_no_domain_logits_single_bin(self, zoedepth_single_bin_config) -> None:
        """Single-bin config must NOT include domain_logits."""
        specs = resolve_io_specs("zoedepth", "depth-estimation", zoedepth_single_bin_config)
        assert "domain_logits" not in specs["output_names"], (
            "domain_logits must not be present for a single bin configuration."
        )
