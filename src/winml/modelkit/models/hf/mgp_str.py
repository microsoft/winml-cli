# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Configure MGP-STR for image-to-text export."""

from __future__ import annotations

from optimum.exporters.onnx.model_configs import MgpstrOnnxConfig
from transformers import MgpstrForSceneTextRecognition

from ...export import register_onnx_overwrite


@register_onnx_overwrite("mgp-str", "image-to-text", library_name="transformers")
class MgpstrImage2TextOnnxConfig(MgpstrOnnxConfig):  # type: ignore[misc]
    """Register the vendor MGP-STR export config for image-to-text."""


MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("mgp-str", "image-to-text"): MgpstrForSceneTextRecognition,
}
