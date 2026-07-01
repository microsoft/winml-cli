# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ViTPose HuggingFace Model Configuration.

ViTPose is a top-down human pose (keypoint-detection) model: a plain ViT
backbone with a lightweight decoder that regresses keypoint heatmaps inside a
given person box.

This module provides:
- MODEL_CLASS_MAPPING: routes keypoint-detection to VitPoseForPoseEstimation,
  and declares it the default task via a (vitpose, None) sentinel.

Why ViTPose needs class mapping:
Optimum already registers the ONNX export config (VitPoseOnnxConfig) for the
"vitpose" model type, so export works once the model is loaded. However,
Optimum's TasksManager has no task-to-class entry for "keypoint-detection",
and transformers' AutoModelForKeypointDetection only recognizes SuperPoint —
not ViTPose. Without this mapping the resolver cannot load the model class for
the keypoint-detection task. The "plus" checkpoints (MoE backbone) load through
the same class; their expert index is fixed at export time by Optimum's
VitPoseModelPatcher, so no extra input is needed.

Why the (vitpose, None) sentinel:
TasksManager cannot infer a task from the ViTPose architecture, so without a
declared default the resolver falls back to an unrelated task and config/build
fail unless the user passes --task keypoint-detection. The sentinel encodes
keypoint-detection as the canonical default (the resolver reverse-looks-up the
task sharing the sentinel's class), making --task optional. Mirrors SAM, which
declares mask-generation the same way.
"""

from __future__ import annotations

from transformers import VitPoseForPoseEstimation


# (model_type, task) -> HuggingFace model class
#
# The (vitpose, None) sentinel declares keypoint-detection as the default task
# applied during auto-detection (when the user does not pass --task). Its value
# is the default *class*; the resolver reverse-looks-up the task name from the
# matching (vitpose, keypoint-detection) -> same class entry.
MODEL_CLASS_MAPPING: dict[tuple[str, str | None], type] = {
    ("vitpose", "keypoint-detection"): VitPoseForPoseEstimation,
    ("vitpose", None): VitPoseForPoseEstimation,
}
