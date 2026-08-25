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

from transformers import VitPoseConfig

from winml.modelkit.export import resolve_io_specs
from winml.modelkit.loader import resolve_task
from winml.modelkit.models.hf import MODEL_CLASS_MAPPING
from winml.modelkit.models.hf.vitpose import MODEL_CLASS_MAPPING as VITPOSE_MAPPING
from winml.modelkit.models.hf.vitpose import VitPoseIOConfig


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

    def test_sentinel_resolves_default_task_without_explicit_task(self):
        """With no --task, the (vitpose, None) sentinel defaults to keypoint-detection."""
        config = MagicMock()
        config.model_type = "vitpose"
        config.architectures = ["VitPoseForPoseEstimation"]
        config._name_or_path = "usyd-community/vitpose-base-simple"

        resolution = resolve_task(config)

        assert resolution.task == "keypoint-detection"
        assert resolution.model_class.__name__ == "VitPoseForPoseEstimation"

    def test_sentinel_registered_in_mapping(self):
        """The (vitpose, None) sentinel shares the keypoint-detection class."""
        assert ("vitpose", None) in MODEL_CLASS_MAPPING
        assert (
            MODEL_CLASS_MAPPING[("vitpose", None)]
            is MODEL_CLASS_MAPPING[("vitpose", "keypoint-detection")]
        )


class TestVitPoseIOConfig:
    def test_resolves_dummy_input_without_loading_processor(self):
        config = VitPoseConfig(
            backbone_config={
                "model_type": "vitpose_backbone",
                "image_size": [256, 192],
                "num_channels": 3,
            },
            num_labels=17,
            scale_factor=4,
            use_simple_decoder=True,
        )

        specs = resolve_io_specs(
            "vitpose",
            "keypoint-detection",
            config,
            height=256,
            width=192,
        )

        assert specs["input_names"] == ["pixel_values"]
        assert specs["output_names"] == ["heatmaps"]
        assert specs["input_shapes"] == [(1, 3, 256, 192)]
        assert specs["input_dtypes"] == ["float32"]
        assert VitPoseIOConfig.__name__ == "VitPoseIOConfig"
