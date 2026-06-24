# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for resolve_loader_config.

Tests the new high-level resolution functions that encapsulate
hf_config loading, model_type override, task auto-detection,
and I/O sub-config resolution for multimodal models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.loader import (
    WinMLLoaderConfig,
    resolve_loader_config,
)


def _make_resolution(task: str, model_class: MagicMock) -> MagicMock:
    """Return a minimal TaskResolution-like mock."""
    from winml.modelkit.loader.resolution import TaskResolution, TaskSource

    return TaskResolution(
        task=task,
        optimum_task=task,
        model_class=model_class,
        source=TaskSource.TASKS_MANAGER,
        composite=None,
    )


def _make_resolution_wrapped(task: str, model_class: MagicMock) -> MagicMock:
    """Return a TaskResolution-like mock with WRAPPED_LIBRARY source."""
    from winml.modelkit.loader.resolution import TaskResolution, TaskSource

    return TaskResolution(
        task=task,
        optimum_task=task,
        model_class=model_class,
        source=TaskSource.WRAPPED_LIBRARY,
        composite=None,
    )


# =============================================================================
# TestResolveLoaderConfig
# =============================================================================


class TestResolveLoaderConfig:
    """Tests for resolve_loader_config function."""

    def test_returns_tuple_of_four(self) -> None:
        """resolve_loader_config returns (WinMLLoaderConfig, hf_config, class, TaskResolution)."""
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_class = MagicMock(spec=[])
        mock_class.__name__ = "BertForMaskedLM"
        mock_class.config_class = None

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("fill-mask", mock_class),
            ),
        ):
            loader_config, hf_config, resolved_class, resolution = resolve_loader_config(
                "bert-base-uncased", task="fill-mask"
            )

        assert isinstance(loader_config, WinMLLoaderConfig)
        assert loader_config.task == "fill-mask"
        assert loader_config.model_class == "BertForMaskedLM"
        assert loader_config.model_type == "bert"
        assert hf_config is mock_config
        assert resolved_class is mock_class
        assert resolution is not None

    def test_neither_model_id_nor_model_type_raises(self) -> None:
        """Neither model_id nor model_type raises ValueError."""
        with pytest.raises(ValueError, match="At least one of"):
            resolve_loader_config()

    def test_hf_config_kwarg_skips_autoconfig_fetch(self) -> None:
        """When the caller supplies hf_config, step 1's AutoConfig.from_pretrained
        is skipped — used by inspect to avoid double-fetching the config.
        """
        provided_config = MagicMock()
        provided_config.model_type = "bert"
        mock_class = MagicMock(spec=[])
        mock_class.__name__ = "BertForMaskedLM"
        mock_class.config_class = None

        with (
            patch("transformers.AutoConfig.from_pretrained") as mock_from_pretrained,
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("fill-mask", mock_class),
            ),
        ):
            loader_config, hf_config, _, _resolution = resolve_loader_config(
                "bert-base-uncased",
                task="fill-mask",
                hf_config=provided_config,
            )

        assert mock_from_pretrained.call_count == 0, (
            "Pre-loaded hf_config must short-circuit AutoConfig.from_pretrained"
        )
        assert hf_config is provided_config
        assert loader_config.model_type == "bert"

    def test_model_type_only_creates_default_config(self) -> None:
        """model_type without model_id uses create_hf_config_from_model_type."""
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_class = MagicMock(spec=[])
        mock_class.__name__ = "BertForMaskedLM"
        mock_class.config_class = None

        with (
            patch(
                "transformers.AutoConfig.for_model",
                return_value=mock_config,
            ) as mock_create,
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("feature-extraction", mock_class),
            ),
        ):
            loader_config, _, _, _resolution = resolve_loader_config(model_type="bert")

        mock_create.assert_called_once_with("bert")
        assert loader_config.task == "feature-extraction"

    def test_explicit_model_type_overrides_hf_config(self) -> None:
        """An explicit model_type (with a model_id) overrides the resolved type.

        Needed so a variant model_type such as ``qwen3_transformer_only`` selects
        the variant rather than the architecture's native type. The override only
        applies when a model_id is present and the requested type differs.
        """
        mock_config = MagicMock()
        mock_config.model_type = "original_type"
        mock_class = MagicMock(spec=[])
        mock_class.__name__ = "SomeModel"
        mock_class.config_class = None

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("text-generation", mock_class),
            ),
        ):
            loader_config, hf_config, _, _resolution = resolve_loader_config(
                "some-model", model_type="gpt2", task="text-generation"
            )

        # The explicit model_type wins over the architecture's native type.
        assert hf_config.model_type == "gpt2"
        # loader_config.model_type reflects the overridden type.
        assert loader_config.model_type == "gpt2"

    def test_auto_detect_task_from_model_type(self) -> None:
        """model_type without task auto-detects first supported task."""
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_class = MagicMock(spec=[])
        mock_class.__name__ = "BertForMaskedLM"
        mock_class.config_class = None

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("feature-extraction", mock_class),
            ) as mock_resolve,
        ):
            loader_config, _, _, _resolution = resolve_loader_config(
                "some-model", model_type="bert"
            )

        assert loader_config.task == "feature-extraction"
        # Verify resolve_task was called with the hf_config (no pre-set task)
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.kwargs.get("task") is None

    def test_no_supported_tasks_raises(self) -> None:
        """model_type with no supported tasks raises ValueError — delegated to resolve_task."""
        mock_config = MagicMock()
        mock_config.model_type = "nonexistent"

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                side_effect=ValueError("No supported tasks found for model_type 'nonexistent'"),
            ),
            pytest.raises(ValueError, match="No supported tasks found"),
        ):
            resolve_loader_config("some-model", model_type="nonexistent")

    def test_model_type_none_no_hf_config_raises(self) -> None:
        """hf_config without model_type raises ValueError."""
        mock_config = MagicMock(spec=[])  # No model_type attribute

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            pytest.raises(ValueError, match="does not have 'model_type'"),
        ):
            resolve_loader_config("some-model")

    def test_trust_remote_code_propagated(self) -> None:
        """trust_remote_code is set in the returned WinMLLoaderConfig."""
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_class = MagicMock(spec=[])
        mock_class.__name__ = "BertForMaskedLM"
        mock_class.config_class = None

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("fill-mask", mock_class),
            ),
        ):
            loader_config, _, _, _resolution = resolve_loader_config(
                "some-model", task="fill-mask", trust_remote_code=True
            )

        assert loader_config.trust_remote_code is True

    def test_explicit_task_passed_through(self) -> None:
        """Explicit task is forwarded to resolve_task."""
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_class = MagicMock(spec=[])
        mock_class.__name__ = "BertForSequenceClassification"
        mock_class.config_class = None

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("text-classification", mock_class),
            ) as mock_resolve,
        ):
            resolve_loader_config("bert-base-uncased", task="text-classification")

        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.kwargs["task"] == "text-classification"

    def test_explicit_model_class_passed_through(self) -> None:
        """Explicit model_class is forwarded to resolve_task."""
        mock_config = MagicMock()
        mock_config.model_type = "bert"
        mock_class = MagicMock(spec=[])
        mock_class.__name__ = "BertForMaskedLM"
        mock_class.config_class = None

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock_config,
            ),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("fill-mask", mock_class),
            ) as mock_resolve,
        ):
            resolve_loader_config(
                "bert-base-uncased",
                task="fill-mask",
                model_class="BertForMaskedLM",
            )

        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.kwargs["model_class"] == "BertForMaskedLM"


# =============================================================================
# TestSubConfigConsolidation - verify resolve_loader_config consolidates
# hf_config and model_type for multimodal sub-models (pure mocks)
# =============================================================================


def _make_mock_class(name: str, *, base_config_key: str = "") -> MagicMock:
    """Create a mock model class with config_class and base_config_key."""
    mock_config_cls = MagicMock()
    mock_config_cls.base_config_key = base_config_key

    mock_cls = MagicMock(spec=[])
    mock_cls.__name__ = name
    mock_cls.config_class = mock_config_cls
    return mock_cls


class TestSubConfigConsolidation:
    """Tests for sub-config consolidation inside resolve_loader_config."""

    def test_no_base_config_key_keeps_parent(self) -> None:
        """When base_config_key is empty, hf_config is returned unchanged."""
        parent = MagicMock()
        parent.model_type = "bert"
        mock_cls = _make_mock_class("BertForMaskedLM", base_config_key="")

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=parent),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("fill-mask", mock_cls),
            ),
        ):
            loader_config, hf_config, _, _resolution = resolve_loader_config(
                "bert-base-uncased", task="fill-mask"
            )

        assert hf_config is parent
        assert loader_config.model_type == "bert"

    def test_base_config_key_extracts_sub_config(self) -> None:
        """When base_config_key is set, sub-config is extracted from parent."""
        sub_config = MagicMock()
        sub_config.model_type = "clip_text_model"
        parent = MagicMock()
        parent.model_type = "clip"
        parent.text_config = sub_config

        mock_cls = _make_mock_class("CLIPTextModelWithProjection", base_config_key="text_config")

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=parent),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("feature-extraction", mock_cls),
            ),
        ):
            loader_config, hf_config, _, _resolution = resolve_loader_config(
                "openai/clip-vit-base-patch32", task="feature-extraction"
            )

        assert hf_config is sub_config
        assert hf_config is not parent
        assert loader_config.model_type == "clip_text_model"

    def test_base_config_key_vision(self) -> None:
        """Vision sub-config extracted via base_config_key='vision_config'."""
        sub_config = MagicMock()
        sub_config.model_type = "clip_vision_model"
        parent = MagicMock()
        parent.model_type = "clip"
        parent.vision_config = sub_config

        mock_cls = _make_mock_class(
            "CLIPVisionModelWithProjection", base_config_key="vision_config"
        )

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=parent),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("image-feature-extraction", mock_cls),
            ),
        ):
            loader_config, hf_config, _, _resolution = resolve_loader_config(
                "openai/clip-vit-base-patch32", task="image-feature-extraction"
            )

        assert hf_config is sub_config
        assert loader_config.model_type == "clip_vision_model"

    def test_base_config_key_missing_attr_keeps_parent(self) -> None:
        """When base_config_key is set but parent lacks the attr, keep parent."""
        parent = MagicMock(spec=["model_type"])
        parent.model_type = "custom"

        mock_cls = _make_mock_class("CustomModel", base_config_key="text_config")

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=parent),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("feature-extraction", mock_cls),
            ),
        ):
            loader_config, hf_config, _, _resolution = resolve_loader_config(
                "some-model", task="feature-extraction"
            )

        assert hf_config is parent
        assert loader_config.model_type == "custom"

    def test_config_class_none_keeps_parent(self) -> None:
        """When resolved_class has no config_class, keep parent unchanged."""
        parent = MagicMock()
        parent.model_type = "bert"
        mock_cls = MagicMock(spec=[])
        mock_cls.__name__ = "AutoModel"
        mock_cls.config_class = None

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=parent),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("feature-extraction", mock_cls),
            ),
        ):
            loader_config, hf_config, _, _resolution = resolve_loader_config(
                "bert-base-uncased", task="feature-extraction"
            )

        assert hf_config is parent
        assert loader_config.model_type == "bert"

    def test_base_config_key_list_keeps_parent(self) -> None:
        """When base_config_key is a list (CLVP edge case), keep parent."""
        parent = MagicMock()
        parent.model_type = "clvp"

        mock_config_cls = MagicMock()
        mock_config_cls.base_config_key = ["speech_config", "decoder_config"]
        mock_cls = MagicMock(spec=[])
        mock_cls.__name__ = "ClvpEncoder"
        mock_cls.config_class = mock_config_cls

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=parent),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("feature-extraction", mock_cls),
            ),
        ):
            loader_config, hf_config, _, _resolution = resolve_loader_config(
                "some-model", task="feature-extraction"
            )

        # list base_config_key is not a string — should fall through safely
        assert hf_config is parent
        assert loader_config.model_type == "clvp"


# =============================================================================
# TestResolveLoaderConfigInputOutput - verify input→output contracts
# =============================================================================


class TestResolveLoaderConfigInputOutput:
    """Test input/output contract of resolve_loader_config for each input combo."""

    def test_output_structure(self) -> None:
        """Return value is a 4-tuple of (WinMLLoaderConfig, PretrainedConfig, type,
        TaskResolution)."""
        parent = MagicMock()
        parent.model_type = "bert"
        mock_cls = MagicMock(spec=[])
        mock_cls.__name__ = "BertForMaskedLM"
        mock_cls.config_class = None

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=parent),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("fill-mask", mock_cls),
            ),
        ):
            result = resolve_loader_config("bert-base-uncased", task="fill-mask")

        # Must be a 4-tuple
        assert isinstance(result, tuple)
        assert len(result) == 4

        loader_config, hf_config, resolved_class, resolution = result

        # Item 0: WinMLLoaderConfig with all fields populated
        assert isinstance(loader_config, WinMLLoaderConfig)
        assert isinstance(loader_config.task, str)
        assert isinstance(loader_config.model_class, str)
        assert isinstance(loader_config.model_type, str)

        # Item 1: hf_config (PretrainedConfig-like, has model_type)
        assert hasattr(hf_config, "model_type")

        # Item 2: resolved_class (a type/callable with __name__)
        assert hasattr(resolved_class, "__name__")

        # Item 3: TaskResolution
        assert resolution is not None

    def test_output_loader_config_fields_match_inputs(self) -> None:
        """WinMLLoaderConfig fields reflect resolved values, not raw inputs."""
        sub = MagicMock()
        sub.model_type = "clip_text_model"
        parent = MagicMock()
        parent.model_type = "clip"
        parent.text_config = sub

        mock_cls = _make_mock_class("CLIPTextModelWithProjection", base_config_key="text_config")

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=parent),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("feature-extraction", mock_cls),
            ),
        ):
            loader_config, hf_config, resolved_class, _resolution = resolve_loader_config(
                "openai/clip-vit-base-patch32", task="feature-extraction"
            )

        # loader_config.model_type is the RESOLVED type, not the input
        assert loader_config.model_type == "clip_text_model"
        # hf_config is the RESOLVED sub-config, not the parent
        assert hf_config.model_type == "clip_text_model"
        assert hf_config is sub
        # resolved_class is the actual class, accessible by __name__
        assert resolved_class.__name__ == "CLIPTextModelWithProjection"

    def test_model_id_only(self) -> None:
        """model_id only → task from architectures, model_type from hf_config."""
        parent = MagicMock()
        parent.model_type = "resnet"
        mock_cls = MagicMock(spec=[])
        mock_cls.__name__ = "ResNetForImageClassification"
        mock_cls.config_class = None

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=parent),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("image-classification", mock_cls),
            ),
        ):
            loader_config, hf_config, resolved_class, _resolution = resolve_loader_config(
                "microsoft/resnet-50"
            )

        assert loader_config.task == "image-classification"
        assert loader_config.model_class == "ResNetForImageClassification"
        assert loader_config.model_type == "resnet"
        assert hf_config is parent
        assert resolved_class is mock_cls

    def test_model_id_plus_task(self) -> None:
        """model_id + task → user task forwarded, model_class resolved."""
        parent = MagicMock()
        parent.model_type = "bert"
        mock_cls = MagicMock(spec=[])
        mock_cls.__name__ = "BertForSequenceClassification"
        mock_cls.config_class = None

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=parent),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("text-classification", mock_cls),
            ) as mock_resolve,
        ):
            loader_config, _, _, _resolution = resolve_loader_config(
                "bert-base-uncased", task="text-classification"
            )

        assert loader_config.task == "text-classification"
        assert mock_resolve.call_args.kwargs["task"] == "text-classification"

    def test_model_id_plus_task_plus_model_class(self) -> None:
        """model_id + task + model_class → all forwarded."""
        parent = MagicMock()
        parent.model_type = "clip"
        sub = MagicMock()
        sub.model_type = "clip_text_model"
        parent.text_config = sub

        mock_cls = _make_mock_class("CLIPTextModelWithProjection", base_config_key="text_config")

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=parent),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("feature-extraction", mock_cls),
            ) as mock_resolve,
        ):
            loader_config, hf_config, _, _resolution = resolve_loader_config(
                "openai/clip-vit-base-patch32",
                task="feature-extraction",
                model_class="CLIPTextModelWithProjection",
            )

        assert mock_resolve.call_args.kwargs["model_class"] == "CLIPTextModelWithProjection"
        assert loader_config.model_type == "clip_text_model"
        assert hf_config is sub

    def test_model_type_only(self) -> None:
        """model_type only → default config, task auto-detected."""
        parent = MagicMock()
        parent.model_type = "bert"
        mock_cls = MagicMock(spec=[])
        mock_cls.__name__ = "AutoModel"
        mock_cls.config_class = None

        with (
            patch(
                "transformers.AutoConfig.for_model",
                return_value=parent,
            ) as mock_create,
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("feature-extraction", mock_cls),
            ),
        ):
            loader_config, hf_config, _, _resolution = resolve_loader_config(model_type="bert")

        mock_create.assert_called_once_with("bert")
        assert loader_config.task == "feature-extraction"
        assert loader_config.model_type == "bert"
        assert hf_config is parent

    def test_model_type_plus_task(self) -> None:
        """model_type + task → task not auto-detected, forwarded directly."""
        parent = MagicMock()
        parent.model_type = "bert"
        mock_cls = MagicMock(spec=[])
        mock_cls.__name__ = "AutoModelForMaskedLM"
        mock_cls.config_class = None

        with (
            patch(
                "transformers.AutoConfig.for_model",
                return_value=parent,
            ),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("fill-mask", mock_cls),
            ) as mock_resolve,
        ):
            loader_config, _, _, _resolution = resolve_loader_config(
                model_type="bert", task="fill-mask"
            )

        assert loader_config.task == "fill-mask"
        # task was provided, so it should be forwarded to resolve_task
        assert mock_resolve.call_args.kwargs["task"] == "fill-mask"

    def test_model_class_only(self) -> None:
        """model_class only → config from config_class(), task auto-detected."""
        mock_hf_config = MagicMock()
        mock_hf_config.model_type = "bert"

        mock_transformers_cls = MagicMock()
        mock_transformers_cls.__name__ = "BertForMaskedLM"

        mock_resolved_cls = MagicMock(spec=[])
        mock_resolved_cls.__name__ = "BertForMaskedLM"
        mock_resolved_cls.config_class = None

        with (
            patch(
                "winml.modelkit.loader.config._create_hf_config_from_model_class",
                return_value=mock_hf_config,
            ) as mock_create,
            patch("transformers.BertForMaskedLM", mock_transformers_cls, create=True),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=_make_resolution("fill-mask", mock_resolved_cls),
            ) as mock_resolve,
        ):
            loader_config, _hf_config, _, _resolution = resolve_loader_config(
                model_class="BertForMaskedLM"
            )

        # create_hf_config_from_model_class called with the imported class
        mock_create.assert_called_once()
        # task auto-detected via resolve_task Case 3
        assert loader_config.task == "fill-mask"
        assert loader_config.model_class == "BertForMaskedLM"
        assert loader_config.model_type == "bert"
        # model_class forwarded to resolve_task
        assert mock_resolve.call_args.kwargs["model_class"] == "BertForMaskedLM"

    def test_model_class_invalid_raises(self) -> None:
        """model_class with unknown class name raises ValueError."""
        with pytest.raises(ValueError, match="not found in any of"):
            resolve_loader_config(model_class="NonExistentModelClass12345")


def test_model_type_only_no_architectures_resolves_first_supported_task():
    from winml.modelkit.loader import resolve_loader_config

    loader_config, _hf_config, _resolved_class, resolution = resolve_loader_config(
        model_id=None, model_type="bert"
    )
    assert resolution.source.value == "wrapped-library"
    assert loader_config.task in ("feature-extraction", "fill-mask")
