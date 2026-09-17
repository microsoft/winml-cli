# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for NLI-style sequence-classifier task resolution."""

from transformers import AutoConfig

from winml.modelkit.loader.resolution import TaskSource, resolve_task


def _cfg(*, id2label: dict[int, str]):
    cfg = AutoConfig.for_model("roberta")
    cfg.architectures = ["RobertaForSequenceClassification"]
    cfg.id2label = id2label
    cfg._name_or_path = "local-nli-probe"
    return cfg


def test_nli_sequence_classifier_surfaces_zero_shot_classification():
    result = resolve_task(
        _cfg(
            id2label={
                0: "contradiction",
                1: "entailment",
                2: "neutral",
            }
        )
    )

    assert result.source == TaskSource.TASKS_MANAGER
    assert result.task == "zero-shot-classification"
    assert result.optimum_task == "text-classification"


def test_non_nli_sequence_classifier_remains_text_classification():
    result = resolve_task(
        _cfg(
            id2label={
                0: "negative",
                1: "positive",
            }
        )
    )

    assert result.source == TaskSource.TASKS_MANAGER
    assert result.task == "text-classification"
    assert result.optimum_task == "text-classification"
