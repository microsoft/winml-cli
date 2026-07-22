# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for MGP-STR image-to-text ONNX config registration and I/O specs.

MGP-STR is a scene-text-recognition model whose vendor ``MgpstrOnnxConfig``
(Optimum) is registered only under ``feature-extraction``. This module
verifies the contribution's task-label alias — ``MgpstrImage2TextOnnxConfig``
registered under ``image-to-text`` — and locks in the inherited I/O contract:

- Input: pixel_values
- Outputs: char_logits, bpe_logits, wp_logits (3 granularity heads)

It also verifies the MODEL_CLASS_MAPPING binding to
``MgpstrForSceneTextRecognition`` (MGP-STR is NOT a Vision2Seq model, so the
loader must not fall back to AutoModelForVision2Seq).

See also: modelkit/models/hf/mgp_str.py
"""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager

from winml.modelkit.export import resolve_io_specs
from winml.modelkit.models.hf.mgp_str import (
    MODEL_CLASS_MAPPING,
    MgpstrImage2TextOnnxConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mgp_str_config():
    """Return a small MGP-STR config with the production I/O topology."""
    from transformers import MgpstrConfig

    return MgpstrConfig(
        image_size=[32, 128],
        patch_size=4,
        num_channels=3,
        hidden_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        mlp_ratio=2,
        max_token_length=27,
        num_character_labels=38,
        num_bpe_labels=50257,
        num_wordpiece_labels=30522,
    )


# =============================================================================
# Registration verification
# =============================================================================


class TestMgpStrRegistration:
    """Verify MgpstrImage2TextOnnxConfig registration and inheritance."""

    def test_mgp_str_config_registered(self) -> None:
        """MgpstrImage2TextOnnxConfig must be registered for image-to-text."""
        config_constructor = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type="mgp-str",
            task="image-to-text",
            library_name="transformers",
        )
        actual_class_name = config_constructor.func.__name__
        assert actual_class_name == "MgpstrImage2TextOnnxConfig", (
            f"Expected MgpstrImage2TextOnnxConfig for mgp-str/image-to-text, "
            f"got {actual_class_name}. Registration may have failed."
        )

    def test_vendor_model_patcher_is_preserved(self) -> None:
        """Subclassing the vendor config must retain its export-time patcher."""
        from optimum.exporters.onnx.model_configs import MgpstrOnnxConfig

        assert MgpstrImage2TextOnnxConfig._MODEL_PATCHER is MgpstrOnnxConfig._MODEL_PATCHER


# =============================================================================
# I/O verification
# =============================================================================


class TestMgpStrIOSpecs:
    """Verify I/O specs for MGP-STR scene text recognition."""

    def test_input_is_pixel_values(self, mgp_str_config) -> None:
        """Input must be the single pixel_values tensor."""
        specs = resolve_io_specs("mgp-str", "image-to-text", mgp_str_config)
        assert specs["input_names"] == ["pixel_values"]

    def test_outputs_are_three_heads(self, mgp_str_config) -> None:
        """Outputs must be the three granularity logit heads."""
        specs = resolve_io_specs("mgp-str", "image-to-text", mgp_str_config)
        assert set(specs["output_names"]) == {
            "char_logits",
            "bpe_logits",
            "wp_logits",
        }


# =============================================================================
# Model class mapping verification
# =============================================================================


class TestMgpStrModelClassMapping:
    """Verify the model-type/task tuple binds to the head-bearing HF class."""

    def test_mapping_binds_scene_text_head(self) -> None:
        """MGP-STR image-to-text must map to its recognition head."""
        from transformers import MgpstrForSceneTextRecognition

        assert (
            MODEL_CLASS_MAPPING[("mgp-str", "image-to-text")]
            is MgpstrForSceneTextRecognition
        )
        assert set(MODEL_CLASS_MAPPING) == {("mgp-str", "image-to-text")}
