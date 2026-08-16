# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for MGP-STR image-to-text registration and export I/O."""

from optimum.exporters.tasks import TasksManager
from transformers import MgpstrConfig, MgpstrForSceneTextRecognition

import winml.modelkit.models.hf  # noqa: F401
from winml.modelkit.export import resolve_io_specs
from winml.modelkit.loader.resolution import TaskSource, resolve_task


def test_stale_architecture_resolves_to_scene_text_recognition() -> None:
    config = MgpstrConfig()
    config.architectures = ["MGPSTRModel"]

    resolved = resolve_task(config)

    assert resolved.task == "image-to-text"
    assert resolved.optimum_task == "image-to-text"
    assert resolved.model_class is MgpstrForSceneTextRecognition
    assert resolved.source == TaskSource.SENTINEL_DEFAULT


def test_image_to_text_onnx_config_is_registered() -> None:
    constructor = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model_type="mgp-str",
        task="image-to-text",
        library_name="transformers",
    )

    assert constructor.func.__name__ == "MgpstrImageToTextIOConfig"


def test_image_to_text_io_preserves_three_head_order() -> None:
    specs = resolve_io_specs("mgp-str", "image-to-text", MgpstrConfig())

    assert specs["input_names"] == ["pixel_values"]
    assert specs["output_names"] == ["char_logits", "bpe_logits", "wp_logits"]
