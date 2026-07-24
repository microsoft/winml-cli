# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for Swin2SR model support registration.

Swin2SR should resolve to image-to-image by default and expose ONNX export
registrations through WinML's local registry import path.
"""

from __future__ import annotations

from transformers import Swin2SRConfig

import winml.modelkit.models  # noqa: F401  # trigger registrations
from winml.modelkit.export.io import _get_onnx_config
from winml.modelkit.loader import get_supported_tasks, resolve_task
from winml.modelkit.models.hf import MODEL_CLASS_MAPPING
from winml.modelkit.models.hf.swin2sr import (
    MODEL_CLASS_MAPPING as SWIN2SR_MAPPING,
    Swin2SRIOConfig,
)


class TestSwin2SRSupport:
    """Swin2SR is discoverable and resolvable as image-to-image."""

    def test_get_supported_tasks_includes_image_to_image(self):
        """swin2sr task list includes image-to-image after local registrations."""
        tasks = get_supported_tasks("swin2sr")
        assert "image-to-image" in tasks

    def test_default_resolution_is_image_to_image(self):
        """Task auto-detection defaults Swin2SR to image-to-image."""
        config = Swin2SRConfig()
        config.architectures = ["Swin2SRForImageSuperResolution"]

        resolution = resolve_task(config)

        assert resolution.task == "image-to-image"
        assert resolution.model_class.__name__ == "AutoModelForImageToImage"

    def test_onnx_config_registration(self):
        """ONNX config lookup for swin2sr/image-to-image resolves to local shim."""
        onnx_config = _get_onnx_config("swin2sr", "image-to-image", Swin2SRConfig())
        assert isinstance(onnx_config, Swin2SRIOConfig)

    def test_mapping_is_merged_and_has_default_sentinel(self):
        """Model-class mapping includes swin2sr task key and default sentinel."""
        assert SWIN2SR_MAPPING.items() <= MODEL_CLASS_MAPPING.items()
        assert ("swin2sr", "image-to-image") in MODEL_CLASS_MAPPING
        assert ("swin2sr", None) in MODEL_CLASS_MAPPING
        assert (
            MODEL_CLASS_MAPPING[("swin2sr", None)]
            is MODEL_CLASS_MAPPING[("swin2sr", "image-to-image")]
        )
