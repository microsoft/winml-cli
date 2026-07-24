# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Swin2SR HuggingFace model registration.

Swin2SR checkpoints default to image super-resolution (image-to-image). Optimum
ships a built-in ``Swin2srOnnxConfig``, but registering it here ensures WinML's
``get_supported_tasks("swin2sr")`` is populated as soon as
``winml.modelkit.models`` is imported, even if
``optimum.exporters.onnx.model_configs`` has not been imported yet.
"""

from __future__ import annotations

from optimum.exporters.onnx.model_configs import Swin2srOnnxConfig
from transformers import AutoModelForImageToImage

from ...export import register_onnx_overwrite


# (model_type, task) -> HuggingFace model class
#
# The (swin2sr, None) sentinel declares image-to-image as the default task for
# task auto-detection when --task is omitted.
MODEL_CLASS_MAPPING: dict[tuple[str, str | None], type] = {
    ("swin2sr", "image-to-image"): AutoModelForImageToImage,
    ("swin2sr", None): AutoModelForImageToImage,
}


@register_onnx_overwrite("swin2sr", "feature-extraction", library_name="transformers")
@register_onnx_overwrite("swin2sr", "image-to-image", library_name="transformers")
class Swin2SRIOConfig(Swin2srOnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """Local registration shim for Swin2SR ONNX export tasks."""

