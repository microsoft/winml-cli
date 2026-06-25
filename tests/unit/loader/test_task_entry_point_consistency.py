# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Cross-entry-point task-resolution consistency.

The unified ``_resolve_task_override`` is consulted by every task-resolution entry
point, so a model resolves to the same task whether reached via ``inspect``/``eval``/
``build -m`` (the unified ``resolve_task``) or ``config``/``build --model-type``
(``resolve_loader_config`` step 2).

Offline: synthetic configs (``Config(architectures=[...])``) and
``AutoConfig.for_model`` — no network.
"""

from __future__ import annotations

import pytest
from transformers import ASTConfig, BartConfig, Sam2Config, SegformerConfig, ViTConfig

from winml.modelkit.loader import resolve_loader_config
from winml.modelkit.loader.resolution import TaskSource, resolve_task


def test_sam2_resolves_to_mask_generation_on_every_entry_point() -> None:
    """The (sam2, None) sentinel's canonical target (mask-generation) is applied by the
    unified override on every entry point — config-based resolve_task and
    config/build-by-model_type — so they no longer disagree (was feature-extraction via
    inspect and --model-type, mask-generation only via -m)."""
    cfg = Sam2Config(architectures=["Sam2Model"])
    assert resolve_task(cfg).task == "mask-generation"
    loader_config, _, _, _ = resolve_loader_config(model_type="sam2")
    assert loader_config.task == "mask-generation"


@pytest.mark.parametrize(
    "config, expected",
    [
        # Audio backbone: the AST fix — feature-extraction, not the old image misroute.
        (ASTConfig(architectures=["ASTModel"]), "feature-extraction"),
        # Vision backbone: modality upgrade via main_input_name=pixel_values.
        (ViTConfig(architectures=["ViTModel"]), "image-feature-extraction"),
        # Multi-task type, head decides (no override).
        (BartConfig(architectures=["BartForSequenceClassification"]), "text-classification"),
        # Single-entry-no-sentinel type: the class-fix entry must NOT force its task; a
        # classification checkpoint resolves head-aware to image-classification, not the
        # image-segmentation class-fix entry (regression guard).
        (
            SegformerConfig(architectures=["SegformerForImageClassification"]),
            "image-classification",
        ),
    ],
)
def test_resolve_task_head_aware_for_checkpoint_configs(config: object, expected: str) -> None:
    """The head-aware auto-detect path of resolve_task resolves a real checkpoint
    config to the architecture-head task."""
    assert resolve_task(config).task == expected


def test_entry_points_agree_on_seq2seq_source():
    from transformers import AutoConfig

    cfg = AutoConfig.for_model("bart")
    cfg.architectures = ["BartForConditionalGeneration"]
    cfg.is_encoder_decoder = True
    assert resolve_task(cfg).source == TaskSource.TASKS_MANAGER
