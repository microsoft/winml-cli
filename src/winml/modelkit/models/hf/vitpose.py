# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ViTPose HuggingFace Model Configuration.

ViTPose is a top-down human pose (keypoint-detection) model: a plain ViT
backbone with a lightweight decoder that regresses keypoint heatmaps inside a
given person box.

This module provides:
- MODEL_CLASS_MAPPING: routes keypoint-detection to VitPoseForPoseEstimation.

Why ViTPose needs class mapping:
Optimum already registers the ONNX export config (VitPoseOnnxConfig) for the
"vitpose" model type, so export works once the model is loaded. However,
Optimum's TasksManager has no task-to-class entry for "keypoint-detection",
and transformers' AutoModelForKeypointDetection only recognizes SuperPoint —
not ViTPose. Without this mapping the resolver cannot load the model class for
the keypoint-detection task. The "plus" checkpoints (MoE backbone) load through
the same class; their expert index is fixed at export time by Optimum's
VitPoseModelPatcher, so no extra input is needed.
"""

from __future__ import annotations

from transformers import VitPoseForPoseEstimation


# (model_type, task) -> HuggingFace model class
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("vitpose", "keypoint-detection"): VitPoseForPoseEstimation,
}
