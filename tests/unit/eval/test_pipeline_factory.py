# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests that evaluators use the shared capability-aware pipeline factory."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from winml.modelkit.eval import WinMLEvaluationConfig
from winml.modelkit.eval.base_evaluator import WinMLEvaluator
from winml.modelkit.inference.pipeline import PipelineCapability


def test_evaluator_delegates_capability_aware_pipeline_creation(monkeypatch) -> None:
    model = SimpleNamespace(
        pipeline_capabilities=frozenset(
            {PipelineCapability.COMBINED_IMAGE_TEXT_PROCESSOR}
        )
    )
    config = WinMLEvaluationConfig(model_id="local-model", task="image-to-text")
    evaluator = object.__new__(WinMLEvaluator)
    evaluator.model = model
    evaluator.config = config
    calls: list[tuple[str, Any, str | None]] = []
    expected_pipeline = object()

    def create_pipeline(task: str, pipeline_model: Any, model_id: str | None) -> object:
        calls.append((task, pipeline_model, model_id))
        return expected_pipeline

    monkeypatch.setattr("winml.modelkit.inference.pipeline.create_pipeline", create_pipeline)

    assert evaluator.prepare_pipeline() is expected_pipeline
    assert calls == [("image-to-text", model, "local-model")]


def test_pipeline_class_ignores_synthesized_mock_capabilities() -> None:
    """Models without a declared capability contract use the default pipeline."""
    from winml.modelkit.inference.pipeline import _pipeline_class_for

    assert _pipeline_class_for(MagicMock()) is None


def test_pipeline_class_rejects_declared_invalid_capabilities() -> None:
    """An explicitly declared contract must remain strictly validated."""
    from winml.modelkit.inference.pipeline import _pipeline_class_for

    with pytest.raises(TypeError, match="must be a frozenset"):
        _pipeline_class_for(SimpleNamespace(pipeline_capabilities=set()))  # type: ignore[arg-type]
