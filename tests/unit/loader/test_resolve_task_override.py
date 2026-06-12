# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ``loader.task._resolve_task_override``.

The single model-type / model-id task-override lookup. It unifies the three
mechanisms that used to live in three places (the ``detect_task`` short-circuit,
the ``(model_type, None)`` sentinel reverse-lookup in
``_detect_task_and_class_from_config``, and the model-id default), so every
detection entry point resolves the same canonical default task. Data-driven
against the real ``MODEL_CLASS_MAPPING`` / ``MODEL_TASK_MAPPING``.
"""

from __future__ import annotations

import pytest

from winml.modelkit.loader.task import _resolve_task_override


@pytest.mark.parametrize(
    "model_type, expected",
    [
        # Single real task + (type, None) sentinel -> that task.
        ("sam", "mask-generation"),
        # Multi-task + (type, None) sentinel -> sentinel's canonical target
        # (reverse-looked-up), NOT a fall-through. This is the sam2 fix.
        ("sam2", "mask-generation"),
        ("sam2-video", "mask-generation"),
        # A single (model_type, task) entry with NO sentinel is a class-fix, not a default
        # declaration -> no override (the architecture head decides). segformer's only entry
        # exists to fix the image-segmentation class, so it must not force that task.
        ("segformer", None),
        # Multi-task, no sentinel -> ambiguous, no override (architecture head decides).
        ("bart", None),
        ("clip", None),
        # Unregistered model_type -> no override.
        ("bert", None),
    ],
)
def test_model_type_override(model_type: str, expected: str | None) -> None:
    assert _resolve_task_override(model_type, None) == expected


def test_model_id_override_takes_priority() -> None:
    """A configured model-id default (prajjwal1/bert-tiny -> feature-extraction)
    wins over any model-type resolution."""
    assert _resolve_task_override("bert", "prajjwal1/bert-tiny") == "feature-extraction"


def test_model_id_override_is_case_insensitive() -> None:
    """get_default_task_for_model_id normalizes case/whitespace."""
    assert _resolve_task_override("bert", "  PRAJJWAL1/BERT-TINY  ") == "feature-extraction"


def test_no_override_for_plain_model_id() -> None:
    """A model_id with no configured default and an unregistered model_type -> None."""
    assert _resolve_task_override("bert", "bert-base-uncased") is None
