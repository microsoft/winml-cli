# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for get_supported_tasks function."""

import pytest

import winml.modelkit.models  # noqa: F401 - trigger registrations
from winml.modelkit.loader import get_supported_tasks


class TestGetSupportedTasks:
    """Tests for get_supported_tasks(model_type).

    No network access needed — queries TasksManager directly.
    """

    @pytest.mark.parametrize(
        "model_type,expected_task",
        [
            ("bert", "fill-mask"),
            ("resnet", "image-classification"),
            ("gpt2", "text-generation"),
            ("vit", "image-classification"),
            ("t5", "text2text-generation"),
            ("detr", "object-detection"),
            ("segformer", "image-segmentation"),
            ("whisper", "automatic-speech-recognition"),
            ("clip", "zero-shot-image-classification"),
            ("llama", "text-generation"),
        ],
        ids=[
            "bert",
            "resnet",
            "gpt2",
            "vit",
            "t5",
            "detr",
            "segformer",
            "whisper",
            "clip",
            "llama",
        ],
    )
    def test_known_model_type_contains_expected_task(self, model_type, expected_task):
        """Known model types must include their primary task."""
        tasks = get_supported_tasks(model_type)
        assert expected_task in tasks

    @pytest.mark.parametrize(
        "model_type,min_count",
        [
            ("bert", 5),
            ("gpt2", 3),
            ("resnet", 2),
            ("clip", 2),
        ],
        ids=["bert", "gpt2", "resnet", "clip"],
    )
    def test_multi_task_models_return_multiple(self, model_type, min_count):
        """Models supporting multiple tasks return all of them."""
        tasks = get_supported_tasks(model_type)
        assert len(tasks) >= min_count

    def test_all_tasks_include_feature_extraction(self):
        """Most models support feature-extraction as a baseline task."""
        for model_type in ["bert", "resnet", "vit", "gpt2"]:
            tasks = get_supported_tasks(model_type)
            assert "feature-extraction" in tasks, f"{model_type} missing feature-extraction"

    def test_invalid_model_type_returns_empty_list(self):
        """Unknown model type returns empty list (graceful degradation)."""
        tasks = get_supported_tasks("nonexistent_model_xyz")
        assert tasks == []

    def test_returns_list_of_strings(self):
        """Return type is always list[str]."""
        tasks = get_supported_tasks("bert")
        assert isinstance(tasks, list)
        assert all(isinstance(t, str) for t in tasks)

    def test_empty_string_returns_empty_list(self):
        """Empty string model_type returns empty list."""
        tasks = get_supported_tasks("")
        assert tasks == []

    def test_library_name_diffusers(self):
        """Diffusers library models return tasks."""
        tasks = get_supported_tasks("unet-2d-condition", library_name="diffusers")
        assert len(tasks) > 0

    def test_library_name_timm(self):
        """TIMM library models return tasks."""
        tasks = get_supported_tasks("default-timm-config", library_name="timm")
        assert len(tasks) > 0

    def test_wrong_library_returns_empty(self):
        """Model type with wrong library returns empty list."""
        # bert is transformers, not diffusers
        tasks = get_supported_tasks("bert", library_name="diffusers")
        assert tasks == []
