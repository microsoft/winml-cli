# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for MGP-STR image-to-text export registration."""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager

from winml.modelkit.export import resolve_io_specs
from winml.modelkit.models.hf.mgp_str import MODEL_CLASS_MAPPING


@pytest.fixture(scope="module")
def mgp_str_config():
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


def test_mgp_str_config_registered() -> None:
    config_constructor = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model_type="mgp-str",
        task="image-to-text",
        library_name="transformers",
    )
    assert config_constructor.func.__name__ == "MgpstrImage2TextOnnxConfig"


def test_mgp_str_io_specs(mgp_str_config) -> None:
    specs = resolve_io_specs("mgp-str", "image-to-text", mgp_str_config)
    assert specs["input_names"] == ["pixel_values"]
    assert specs["output_names"] == ["char_logits", "bpe_logits", "wp_logits"]


def test_mgp_str_model_class_mapping() -> None:
    from transformers import MgpstrForSceneTextRecognition

    assert (
        MODEL_CLASS_MAPPING[("mgp-str", "image-to-text")]
        is MgpstrForSceneTextRecognition
    )
