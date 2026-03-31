# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for BLIP ONNX config registration and I/O specs.

Verifies that BlipCaptioningIOConfig is correctly registered with Optimum's
TasksManager for both image-to-text and image-text-to-text tasks, and that
I/O specs match the expected tensors:

- Inputs: pixel_values, input_ids, attention_mask
- Output: logits
- Sequence length from text_config.max_position_embeddings

See also: modelkit/models/hf/blip.py
"""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager

# Trigger OnnxConfig registration with TasksManager
import winml.modelkit.models  # noqa: F401
from winml.modelkit.export.io import generate_dummy_inputs, resolve_io_specs


@pytest.fixture(scope="module")
def blip_config():
    """Minimal BlipConfig for testing (image_size=32, max_position_embeddings=32).

    BLIP has nested sub-configs (vision_config, text_config).
    """
    from transformers import BlipConfig

    return BlipConfig(
        vision_config={
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 128,
        },
        text_config={
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 32,
        },
    )


# =============================================================================
# Registration verification
# =============================================================================


class TestBlipRegistration:
    """Verify BlipCaptioningIOConfig is registered for both tasks."""

    @pytest.mark.parametrize(
        "task",
        ["image-to-text", "image-text-to-text"],
        ids=["i2t", "it2t"],
    )
    def test_blip_config_registered(self, task: str) -> None:
        """BlipCaptioningIOConfig must be registered for both tasks."""
        config_constructor = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type="blip",
            task=task,
            library_name="transformers",
        )
        actual_class_name = config_constructor.func.__name__
        assert actual_class_name == "BlipCaptioningIOConfig", (
            f"Expected BlipCaptioningIOConfig for blip/{task}, "
            f"got {actual_class_name}. Registration may have failed."
        )


# =============================================================================
# I/O verification
# =============================================================================


class TestBlipIOSpecs:
    """Both BLIP tasks share the same ONNX graph and I/O specs.

    ONNX export traces forward() which always requires input_ids for the
    decoder. The difference between image-to-text and image-text-to-text
    is only in what the caller feeds at inference time.
    """

    @pytest.mark.parametrize(
        "task",
        ["image-to-text", "image-text-to-text"],
        ids=["i2t", "it2t"],
    )
    def test_inputs_have_vision_and_text(self, task: str, blip_config) -> None:
        """Both tasks require pixel_values + input_ids + attention_mask."""
        specs = resolve_io_specs("blip", task, blip_config)
        assert "pixel_values" in specs["input_names"]
        assert "input_ids" in specs["input_names"]
        assert "attention_mask" in specs["input_names"]

    @pytest.mark.parametrize(
        "task",
        ["image-to-text", "image-text-to-text"],
        ids=["i2t", "it2t"],
    )
    def test_text_inputs_use_max_position_embeddings(self, task: str, blip_config) -> None:
        """Text input seq_len must come from text_config.max_position_embeddings."""
        inputs = generate_dummy_inputs("blip", task, blip_config)
        seq_len = inputs["input_ids"].shape[1]
        expected = blip_config.text_config.max_position_embeddings
        assert seq_len == expected, f"Expected seq_len={expected}, got {seq_len}."

    @pytest.mark.parametrize(
        "task",
        ["image-to-text", "image-text-to-text"],
        ids=["i2t", "it2t"],
    )
    def test_output_is_logits(self, task: str, blip_config) -> None:
        """Both tasks must output logits."""
        specs = resolve_io_specs("blip", task, blip_config)
        assert specs["output_names"] == ["logits"], (
            f"Expected ['logits'], got {specs['output_names']}"
        )
