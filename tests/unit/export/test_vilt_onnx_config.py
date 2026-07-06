# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for ViLT VQA ONNX config registration and I/O specs.

Verifies that ViltVqaOnnxConfig is correctly registered with Optimum's
TasksManager for visual-question-answering, and that I/O specs match the
expected tensors:

- Inputs: input_ids, attention_mask, token_type_ids, pixel_values
- Outputs: logits (over the fixed answer vocabulary)

``pixel_mask`` is intentionally NOT an ONNX input: the export-time
ModelPatcher replaces ViLT's non-traceable ``visual_embed`` with a
statically-shaped version that synthesizes an all-ones token mask
internally, so the exported graph is called with only the 4 declared
inputs.  This module locks that contract in.

``visual-question-answering`` has no default AutoModel routing for ViLT,
so the (model_type, task) tuple is bound directly to the head-bearing HF
class via MODEL_CLASS_MAPPING.

See also: modelkit/models/hf/vilt.py
"""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager

# Trigger OnnxConfig registration with TasksManager
import winml.modelkit.models  # noqa: F401
from winml.modelkit.export import resolve_io_specs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vilt_config():
    """Minimal ViltConfig matching the exported dandelin/vilt-b32 topology.

    image_size/patch_size/num_channels drive the vision path; the hidden
    dims are shrunk to keep the (weightless) dummy-input generation fast.
    """
    from transformers import ViltConfig

    return ViltConfig(
        image_size=384,
        patch_size=32,
        num_channels=3,
        max_position_embeddings=40,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=37,
        num_labels=8,
    )


# =============================================================================
# Registration verification
# =============================================================================


class TestViltRegistration:
    """Verify ViltVqaOnnxConfig is registered for visual-question-answering."""

    def test_vilt_config_registered(self) -> None:
        """ViltVqaOnnxConfig must be registered for visual-question-answering."""
        config_constructor = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type="vilt",
            task="visual-question-answering",
            library_name="transformers",
        )
        actual_class_name = config_constructor.func.__name__
        assert actual_class_name == "ViltVqaOnnxConfig", (
            f"Expected ViltVqaOnnxConfig for vilt/visual-question-answering, "
            f"got {actual_class_name}. Registration may have failed."
        )


# =============================================================================
# I/O verification
# =============================================================================


class TestViltIOSpecs:
    """Verify I/O specs for ViLT visual question answering."""

    def test_declares_exactly_four_inputs(self, vilt_config) -> None:
        """Inputs must be exactly the 4 declared text+vision tensors."""
        specs = resolve_io_specs("vilt", "visual-question-answering", vilt_config)
        assert set(specs["input_names"]) == {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "pixel_values",
        }

    def test_pixel_mask_is_dropped(self, vilt_config) -> None:
        """pixel_mask must NOT be an exported input (patched visual_embed synthesizes it)."""
        specs = resolve_io_specs("vilt", "visual-question-answering", vilt_config)
        assert "pixel_mask" not in specs["input_names"], (
            "pixel_mask must not be exported: the patched visual_embed builds an "
            "all-ones mask internally, and exporting it would break sess.run."
        )

    def test_outputs_contain_logits(self, vilt_config) -> None:
        """Output must be the single classification logits tensor."""
        specs = resolve_io_specs("vilt", "visual-question-answering", vilt_config)
        assert specs["output_names"] == ["logits"]

    def test_pixel_values_shape_is_static_384(self, vilt_config) -> None:
        """pixel_values H,W must be pinned to the config image_size (static)."""
        specs = resolve_io_specs("vilt", "visual-question-answering", vilt_config)
        idx = specs["input_names"].index("pixel_values")
        _, channels, height, width = specs["input_shapes"][idx]
        assert (channels, height, width) == (3, 384, 384)


# =============================================================================
# Model class mapping verification
# =============================================================================


class TestViltModelClassMapping:
    """Verify the (model_type, task) tuple binds to the head-bearing HF class."""

    def test_mapping_binds_vqa_head(self) -> None:
        """(vilt, visual-question-answering) must map to ViltForQuestionAnswering."""
        from transformers import ViltForQuestionAnswering

        from winml.modelkit.models.hf.vilt import MODEL_CLASS_MAPPING

        assert (
            MODEL_CLASS_MAPPING[("vilt", "visual-question-answering")]
            is ViltForQuestionAnswering
        )
