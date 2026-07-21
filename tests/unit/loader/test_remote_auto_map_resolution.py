# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from unittest.mock import MagicMock, patch

from transformers import AutoModelForImageSegmentation, PretrainedConfig

from winml.modelkit.loader import load_hf_model
from winml.modelkit.loader.config import resolve_loader_config
from winml.modelkit.loader.resolution import (
    TaskSource,
    _resolve_remote_auto_model_class,
    resolve_task,
)


class RemoteImageSegmentationConfig(PretrainedConfig):
    model_type = "custom-image-segmentation"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.architectures = ["RemoteSegmentationModel"]
        self.auto_map = {
            "AutoConfig": "configuration.RemoteConfig",
            "AutoModelForImageSegmentation": "modeling.RemoteSegmentationModel",
        }


def test_resolve_task_uses_matching_remote_auto_map_metadata() -> None:
    resolution = resolve_task(RemoteImageSegmentationConfig())

    assert resolution.task == "image-segmentation"
    assert resolution.model_class is AutoModelForImageSegmentation
    assert resolution.source == TaskSource.TASKS_MANAGER


def test_explicit_task_uses_matching_remote_auto_map_metadata() -> None:
    resolution = resolve_task(
        RemoteImageSegmentationConfig(),
        task="image-segmentation",
    )

    assert resolution.model_class is AutoModelForImageSegmentation
    assert resolution.source == TaskSource.USER_TASK


def test_remote_auto_map_requires_architecture_match() -> None:
    config = RemoteImageSegmentationConfig()
    config.architectures = ["DifferentModel"]

    assert _resolve_remote_auto_model_class(config) is None


def test_loader_config_preserves_remote_auto_class_and_trust() -> None:
    loader, _, resolved_class, resolution = resolve_loader_config(
        "org/custom-model",
        trust_remote_code=True,
        hf_config=RemoteImageSegmentationConfig(),
    )

    assert loader.task == "image-segmentation"
    assert loader.model_class == "AutoModelForImageSegmentation"
    assert loader.model_type == "custom-image-segmentation"
    assert loader.trust_remote_code is True
    assert resolved_class is AutoModelForImageSegmentation
    assert resolution.source == TaskSource.TASKS_MANAGER


def test_load_hf_model_calls_remote_auto_class_with_trust() -> None:
    config = RemoteImageSegmentationConfig()
    model = MagicMock()
    model.parameters.return_value = []
    with (
        patch("winml.modelkit.loader.hf.AutoConfig.from_pretrained", return_value=config),
        patch.object(
            AutoModelForImageSegmentation,
            "from_pretrained",
            return_value=model,
        ) as from_pretrained,
    ):
        loaded, _, task = load_hf_model(
            "org/custom-model",
            trust_remote_code=True,
        )

    assert loaded is model
    assert task == "image-segmentation"
    from_pretrained.assert_called_once_with(
        "org/custom-model",
        trust_remote_code=True,
    )
