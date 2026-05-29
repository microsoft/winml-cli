# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for `HTPExporter._get_optimum_patcher` task-synonym handling.

Optimum's ``TasksManager.get_exporter_config_constructor`` only accepts
canonical task names. When ``_get_optimum_patcher`` is invoked with a
HuggingFace pipeline alias (e.g. ``image-feature-extraction``), the lookup
raises and the patcher silently falls back to ``contextlib.nullcontext()``,
producing ONNX exports without the Transformers >= 4.53 tracing patches.

This test pins the contract that ``_get_optimum_patcher`` normalises the
task argument via ``_map_task_synonym`` before the TasksManager lookup.

Regression for https://github.com/microsoft/winml-cli/issues/777.
"""

from __future__ import annotations

from unittest.mock import patch

import torch.nn as nn

from winml.modelkit.export.htp import HTPExporter


class _FakeConfig:
    """Minimal HF-style config exposing the model_type the patcher checks."""

    model_type = "dinov2"


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()


class TestGetOptimumPatcherTaskSynonyms:
    """_get_optimum_patcher must normalise HF-alias tasks before TasksManager lookup."""

    def test_hf_alias_image_feature_extraction_is_normalized(self) -> None:
        """Calling the patcher with 'image-feature-extraction' must pass the canonical
        'feature-extraction' to ``TasksManager.get_exporter_config_constructor``.

        We patch the TasksManager call and capture the ``task`` kwarg. The
        spy raises ``KeyError`` to short-circuit the rest of the patcher
        (no need to construct a real OnnxConfig); the patcher returns
        ``nullcontext()`` on KeyError, which is fine — we assert on the
        captured task argument.
        """
        captured: dict[str, object] = {}

        def spy(*args: object, **kwargs: object) -> None:
            captured["task"] = kwargs.get("task")
            raise KeyError("test sentinel — short-circuit after capture")

        with patch(
            "optimum.exporters.tasks.TasksManager.get_exporter_config_constructor",
            side_effect=spy,
        ):
            HTPExporter._get_optimum_patcher(_FakeModel(), task="image-feature-extraction")

        assert captured.get("task") == "feature-extraction", (
            f"Expected normalised task 'feature-extraction' to reach "
            f"TasksManager.get_exporter_config_constructor, got {captured.get('task')!r}. "
            "_get_optimum_patcher must call _map_task_synonym on the task argument."
        )

    def test_canonical_task_passes_through_unchanged(self) -> None:
        """Control: canonical 'feature-extraction' passes through unchanged."""
        captured: dict[str, object] = {}

        def spy(*args: object, **kwargs: object) -> None:
            captured["task"] = kwargs.get("task")
            raise KeyError("test sentinel")

        with patch(
            "optimum.exporters.tasks.TasksManager.get_exporter_config_constructor",
            side_effect=spy,
        ):
            HTPExporter._get_optimum_patcher(_FakeModel(), task="feature-extraction")

        assert captured.get("task") == "feature-extraction"
