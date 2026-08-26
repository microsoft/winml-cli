# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""MGP-STR image-to-text export and model-class registration."""

from __future__ import annotations

from optimum.exporters.onnx.model_configs import MgpstrOnnxConfig
from transformers import MgpstrForSceneTextRecognition

from ...export import register_onnx_overwrite


@register_onnx_overwrite("mgp-str", "image-to-text", library_name="transformers")
class MgpstrImageToTextIOConfig(MgpstrOnnxConfig):  # type: ignore[misc]
    """Register the vendor MGP-STR image-to-text ONNX contract."""


MODEL_CLASS_MAPPING: dict[tuple[str, str | None], type] = {
    ("mgp-str", "image-to-text"): MgpstrForSceneTextRecognition,
    ("mgp-str", None): MgpstrForSceneTextRecognition,
}
