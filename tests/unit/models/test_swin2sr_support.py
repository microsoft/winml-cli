# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Offline tests for vendor-provided Swin2SR support."""

from transformers import Swin2SRConfig

from winml.modelkit.export.io import _get_onnx_config, ensure_hf_models_registered
from winml.modelkit.loader import get_supported_tasks, resolve_task


def test_swin2sr_vendor_registration_supports_image_to_image() -> None:
    ensure_hf_models_registered()
    assert "image-to-image" in get_supported_tasks("swin2sr")

    onnx_config = _get_onnx_config("swin2sr", "image-to-image", Swin2SRConfig())
    assert type(onnx_config).__name__ == "Swin2srOnnxConfig"


def test_swin2sr_default_task_resolves_to_image_to_image() -> None:
    config = Swin2SRConfig(architectures=["Swin2SRForImageSuperResolution"])

    resolution = resolve_task(config)

    assert resolution.task == "image-to-image"
    assert resolution.model_class.__name__ == "AutoModelForImageToImage"
