# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ViTPose keypoint-detection model-class resolution.

Optimum registers the ViTPose ONNX export config but has no
task-to-class entry for ``keypoint-detection``, and transformers'
``AutoModelForKeypointDetection`` only recognises SuperPoint. The
``("vitpose", "keypoint-detection")`` entry in ``MODEL_CLASS_MAPPING``
bridges that gap so the resolver can load ``VitPoseForPoseEstimation``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from winml.modelkit.loader import resolve_task
from winml.modelkit.models.hf import MODEL_CLASS_MAPPING
from winml.modelkit.models.hf.vitpose import MODEL_CLASS_MAPPING as VITPOSE_MAPPING


class TestVitPoseMapping:
    """ViTPose keypoint-detection routes to VitPoseForPoseEstimation."""

    def test_mapping_entry_registered(self):
        """The aggregated mapping exposes the vitpose keypoint-detection entry."""
        assert ("vitpose", "keypoint-detection") in MODEL_CLASS_MAPPING
        assert (
            MODEL_CLASS_MAPPING[("vitpose", "keypoint-detection")].__name__
            == "VitPoseForPoseEstimation"
        )

    def test_module_mapping_merged_into_aggregate(self):
        """The module-level mapping is included in the aggregated mapping."""
        assert VITPOSE_MAPPING.items() <= MODEL_CLASS_MAPPING.items()

    def test_explicit_task_resolves_vitpose_class(self):
        """An explicit keypoint-detection task resolves VitPoseForPoseEstimation."""
        config = MagicMock()
        config.model_type = "vitpose"
        config.architectures = ["VitPoseForPoseEstimation"]
        config._name_or_path = "usyd-community/vitpose-base-simple"

        resolution = resolve_task(config, task="keypoint-detection")

        assert resolution.task == "keypoint-detection"
        assert resolution.model_class.__name__ == "VitPoseForPoseEstimation"

