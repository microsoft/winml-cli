# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for auto-detect task/class resolution via ``resolve_task``.

Tests the resolution strategy:
1. Specializations (HF_MODEL_CLASS_MAPPING / sentinel) - highest priority
2. TasksManager.get_model_class_for_task() - honor if successful
3. Fallback to architecture class from config.architectures

Uses BLIP as primary test case for TasksManager fallback scenario.

Note on missing/invalid architectures: the legacy auto-detect orchestrator
raised ``ValueError`` when ``config.architectures`` was missing/empty/unimportable
and no wrapped-library route applied. The unified ``resolve_task`` never raises on
the auto-detect path — it falls back to the last-resort ``HF_TASK_DEFAULT`` instead,
so those cases now assert ``TaskSource.HF_TASK_DEFAULT``.
"""

from unittest.mock import MagicMock, patch

from winml.modelkit.loader.resolution import TaskSource, resolve_task
from winml.modelkit.loader.task import (
    WRAPPED_LIBRARY_MODEL_TYPES,
    resolve_optimum_library,
)


class TestAutoDetectMissingArchitectures:
    """Missing/invalid architectures fall back to the last-resort HF_TASK_DEFAULT.

    (The legacy orchestrator raised ValueError here; ``resolve_task`` has a
    last-resort default and never raises on the auto-detect path.)
    """

    def test_missing_architectures_falls_back_to_hf_task_default(self):
        """config.architectures=None with an unknown model_type -> HF_TASK_DEFAULT."""
        config = MagicMock()
        config.architectures = None
        config.model_type = None
        config._name_or_path = ""

        r = resolve_task(config)

        assert r.source == TaskSource.HF_TASK_DEFAULT

    def test_empty_architectures_falls_back_to_hf_task_default(self):
        """config.architectures=[] with an unknown model_type -> HF_TASK_DEFAULT."""
        config = MagicMock()
        config.architectures = []
        config.model_type = None
        config._name_or_path = ""

        r = resolve_task(config)

        assert r.source == TaskSource.HF_TASK_DEFAULT

    def test_unimportable_architecture_falls_back_to_hf_task_default(self):
        """An architecture name that cannot be imported from transformers -> HF_TASK_DEFAULT."""
        config = MagicMock()
        config.architectures = ["NonExistentClass"]
        config.model_type = "some-model"
        config._name_or_path = ""

        r = resolve_task(config)

        assert r.source == TaskSource.HF_TASK_DEFAULT


class TestTasksManagerFallback:
    """Tests for TasksManager fallback behavior using mocks.

    These tests patch optimum.exporters.tasks.TasksManager to simulate
    various scenarios including fallback when TasksManager fails.
    """

    def test_fallback_to_arch_class_when_tasksmanager_fails(self):
        """Test fallback to arch_model_class when TasksManager.get_model_class_for_task fails.

        Simulates scenario where TasksManager can infer the task but
        get_model_class_for_task raises an exception.
        """
        from transformers import BlipForConditionalGeneration

        config = MagicMock()
        config.architectures = ["BlipForConditionalGeneration"]
        config.model_type = "blip"
        config._name_or_path = ""

        with patch("optimum.exporters.tasks.TasksManager.get_model_class_for_task") as mock_get:
            # Simulate TasksManager failure
            mock_get.side_effect = Exception("Model not supported")

            r = resolve_task(config)

        assert r.task == "image-text-to-text"
        # Should fallback to architecture class
        assert r.model_class == BlipForConditionalGeneration


class TestModelTaskDefaultsOverride:
    """Tests for per-model-type default-task auto-detection override.

    Some model families (e.g., SAM/SAM2) have an architecture class whose
    default TasksManager mapping ("feature-extraction") differs from the
    canonical export target ("mask-generation"). The default is encoded as a
    MODEL_CLASS_MAPPING[(model_type, None)] sentinel entry that biases
    auto-detection toward the right export configuration when --task is
    not provided.
    """

    def test_sam2_video_defaults_to_mask_generation(self):
        """Sam2Model on sam2_video config auto-detects to mask-generation."""
        # Trigger HF model registrations (loads SAM sentinel entries)
        import winml.modelkit.models.hf  # noqa: F401
        from winml.modelkit.models.hf.sam import SAM2MaskGeneration

        config = MagicMock()
        config.architectures = ["Sam2Model"]
        config.model_type = "sam2_video"
        config._name_or_path = ""

        r = resolve_task(config)

        assert r.task == "mask-generation"
        assert r.source == TaskSource.SENTINEL_DEFAULT
        assert r.model_class is SAM2MaskGeneration

    def test_sam_defaults_to_mask_generation(self):
        """SamModel on sam config auto-detects to mask-generation."""
        import winml.modelkit.models.hf  # noqa: F401
        from winml.modelkit.models.hf.sam import SAMMaskGeneration

        config = MagicMock()
        config.architectures = ["SamModel"]
        config.model_type = "sam"
        config._name_or_path = ""

        r = resolve_task(config)

        assert r.task == "mask-generation"
        assert r.source == TaskSource.SENTINEL_DEFAULT
        assert r.model_class is SAMMaskGeneration

    def test_model_type_underscore_normalized(self):
        """sam2_video (underscore) matches sam2-video (hyphen) in MODEL_CLASS_MAPPING."""
        import winml.modelkit.models.hf  # noqa: F401

        config = MagicMock()
        config.architectures = ["Sam2Model"]
        config.model_type = "sam2_video"
        config._name_or_path = ""

        r = resolve_task(config)
        assert r.task == "mask-generation"

    def test_no_override_for_unrelated_model(self):
        """Models without a (model_type, None) sentinel keep TasksManager-inferred task."""
        from transformers import ResNetForImageClassification

        config = MagicMock()
        config.architectures = ["ResNetForImageClassification"]
        config.model_type = "resnet"
        config._name_or_path = ""

        r = resolve_task(config)

        assert r.task == "image-classification"
        assert r.source == TaskSource.TASKS_MANAGER
        # TasksManager returns AutoModelForImageClassification, not the arch class
        assert r.model_class is not ResNetForImageClassification or r.task == "image-classification"


class TestResolveOptimumLibrary:
    """Unit tests for the resolve_optimum_library wrapped-library router."""

    def test_timm_wrapper_routes_to_timm(self):
        """timm_wrapper under the default library routes to Optimum's 'timm'."""
        assert resolve_optimum_library("timm_wrapper", "transformers") == "timm"

    def test_unmapped_model_type_unchanged(self):
        """A normal transformers model_type is not rerouted."""
        assert resolve_optimum_library("bert", "transformers") == "transformers"

    def test_none_model_type_unchanged(self):
        assert resolve_optimum_library(None, "transformers") == "transformers"

    def test_explicit_library_is_respected(self):
        """An explicit (non-default) library always wins over the wrapper routing."""
        assert resolve_optimum_library("timm_wrapper", "timm") == "timm"
        assert resolve_optimum_library("timm_wrapper", "diffusers") == "diffusers"


class TestWrappedLibraryArchitecturesFallback:
    """Auto-detection for wrapper model_types that carry no `architectures`.

    timm checkpoints load through transformers' TimmWrapper as TimmWrapperConfig
    (architectures=None); the loader resolves them via WRAPPED_LIBRARY_MODEL_TYPES
    instead of falling back to the generic default.
    """

    def test_timm_wrapper_resolves_without_architectures(self):
        config = MagicMock()
        config.architectures = None
        config.model_type = "timm_wrapper"
        config._name_or_path = ""

        r = resolve_task(config)

        # Task is derived from Optimum's task list for the timm library, not hardcoded.
        assert WRAPPED_LIBRARY_MODEL_TYPES["timm_wrapper"] == "timm"
        assert r.task == "image-classification"
        assert r.source == TaskSource.WRAPPED_LIBRARY
        # A generic Auto* class is used; it dispatches to TimmWrapper at load time.
        assert r.model_class.__name__ == "AutoModelForImageClassification"

    def test_missing_architectures_without_wrapper_falls_back_to_hf_task_default(self):
        """Unknown model_type with no architectures and no wrapped-library route.

        The legacy orchestrator raised ValueError; ``resolve_task`` instead falls
        back to the last-resort HF_TASK_DEFAULT.
        """
        config = MagicMock()
        config.architectures = None
        config.model_type = "totally-unknown-model-xyz"
        config._name_or_path = ""

        r = resolve_task(config)

        assert r.source == TaskSource.HF_TASK_DEFAULT
