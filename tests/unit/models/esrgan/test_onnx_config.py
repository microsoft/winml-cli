# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ESRGAN ONNX export config registration."""

from __future__ import annotations

from optimum.exporters.tasks import TasksManager

# Trigger registration via import side effects
import winml.modelkit.models.hf as _hf  # noqa: F401
from winml.modelkit.models.hf.esrgan import ESRGANConfig, ESRGANIOConfig


class TestESRGANOnnxConfigRegistration:
    """Verify ESRGAN OnnxConfig is reachable through Optimum's TasksManager."""

    def test_config_registered_for_image_to_image(self) -> None:
        config_constructor = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type="ESRGAN",
            task="image-to-image",
            library_name="transformers",
        )
        actual_class_name = config_constructor.func.__name__
        assert actual_class_name == "ESRGANIOConfig"

    def test_inputs_have_pixel_values(self) -> None:
        config = ESRGANConfig()
        io_config = ESRGANIOConfig(config)
        inputs = io_config.inputs
        assert "pixel_values" in inputs
        # Dynamic axes: batch (0), height (2), width (3)
        assert 0 in inputs["pixel_values"]
        assert 2 in inputs["pixel_values"]
        assert 3 in inputs["pixel_values"]

    def test_outputs_have_reconstruction(self) -> None:
        config = ESRGANConfig()
        io_config = ESRGANIOConfig(config)
        outputs = io_config.outputs
        assert "reconstruction" in outputs
        assert 0 in outputs["reconstruction"]

    def test_image_to_image_in_supported_tasks(self) -> None:
        """`get_supported_tasks_for_model_type` lists image-to-image for ESRGAN."""
        supported = TasksManager.get_supported_tasks_for_model_type(
            "ESRGAN", exporter="onnx", library_name="transformers",
        )
        tasks = list(supported.keys()) if isinstance(supported, dict) else list(supported)
        assert "image-to-image" in tasks
