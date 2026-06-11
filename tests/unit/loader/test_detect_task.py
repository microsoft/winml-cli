# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ``loader.task.detect_task`` — the single modality-aware detector.

These tests pin the D2 vision-modality upgrade, which is applied to the RETURNED
task only. The internal ``_detect_task_from_config`` (used for model-class
resolution) is mocked so the dispatch reaches the TasksManager branch
deterministically without network.
"""

from __future__ import annotations

from unittest.mock import patch

from winml.modelkit.loader import detect_task, resolve_task_and_model_class
from winml.modelkit.loader.task import _upgrade_fill_mask_for_seq2seq


class _FakeConfig:
    """Minimal stand-in for a HF PretrainedConfig used by detect_task."""

    def __init__(
        self,
        model_type: str,
        *,
        image_size: int | None = None,
        patch_size: int | None = None,
        architectures: list[str] | None = None,
        is_encoder_decoder: bool = False,
    ) -> None:
        self.model_type = model_type
        self.architectures = architectures or ["FakeModel"]
        self.is_encoder_decoder = is_encoder_decoder
        if image_size is not None:
            self.image_size = image_size
        if patch_size is not None:
            self.patch_size = patch_size

    def to_dict(self) -> dict:
        d: dict = {"model_type": self.model_type}
        if hasattr(self, "image_size"):
            d["image_size"] = self.image_size
        if hasattr(self, "patch_size"):
            d["patch_size"] = self.patch_size
        return d


_DETECT = "winml.modelkit.loader.task._detect_task_from_config"


def test_upgrade_fill_mask_for_seq2seq_upgrades_encoder_decoder() -> None:
    """Optimum mislabels encoder-decoder generation heads (e.g. BartForConditionalGeneration)
    as fill-mask; an is_encoder_decoder config reported as fill-mask is a seq2seq generator."""
    cfg = _FakeConfig("bart", is_encoder_decoder=True)
    assert _upgrade_fill_mask_for_seq2seq("fill-mask", cfg) == "text2text-generation"


def test_upgrade_fill_mask_for_seq2seq_leaves_encoder_only_masked_lm() -> None:
    """A real masked-LM (BERT/RoBERTa) is encoder-only -> fill-mask stays fill-mask."""
    cfg = _FakeConfig("bert", is_encoder_decoder=False)
    assert _upgrade_fill_mask_for_seq2seq("fill-mask", cfg) == "fill-mask"


def test_upgrade_fill_mask_for_seq2seq_only_touches_fill_mask() -> None:
    """Tasks other than fill-mask are never rewritten, even for encoder-decoder configs."""
    cfg = _FakeConfig("bart", is_encoder_decoder=True)
    assert _upgrade_fill_mask_for_seq2seq("text-classification", cfg) == "text-classification"


def test_upgrade_fill_mask_for_seq2seq_ignores_config_without_flag() -> None:
    """A config that does not carry is_encoder_decoder at all is never upgraded
    (getattr defaults to False), so a partial/duck-typed config is never silently
    rewritten — as the docstring promises."""

    class _NoFlagConfig:
        pass

    assert _upgrade_fill_mask_for_seq2seq("fill-mask", _NoFlagConfig()) == "fill-mask"


def test_d2_upgrades_vision_feature_extraction_on_returned_task() -> None:
    """Top-level image_size/patch_size + feature-extraction -> image-feature-extraction."""
    cfg = _FakeConfig("faketype", image_size=518, patch_size=14)
    with patch(_DETECT, return_value="feature-extraction") as m:
        task, source = detect_task(cfg)
    assert task == "image-feature-extraction"
    assert source == "TasksManager"
    # Internal Optimum-canonical detection is still consulted (pre-D2).
    m.assert_called_once()


def test_d2_does_not_fire_for_text_feature_extraction() -> None:
    """No top-level image_size/patch_size -> feature-extraction stays as-is (text)."""
    cfg = _FakeConfig("faketype")  # no vision fields
    with patch(_DETECT, return_value="feature-extraction"):
        task, source = detect_task(cfg)
    assert task == "feature-extraction"
    assert source == "TasksManager"


def test_d2_only_upgrades_feature_extraction() -> None:
    """A vision config whose task is NOT feature-extraction is left untouched."""
    cfg = _FakeConfig("faketype", image_size=224, patch_size=16)
    with patch(_DETECT, return_value="image-classification"):
        task, _ = detect_task(cfg)
    assert task == "image-classification"


def test_detect_task_falls_back_to_hf_task_defaults() -> None:
    """When TasksManager detection raises ValueError, fall back to HF_TASK_DEFAULTS."""
    cfg = _FakeConfig("faketype")
    with patch(_DETECT, side_effect=ValueError("no task")):
        _, source = detect_task(cfg)
    assert source == "HF_TASK_DEFAULTS"


def test_detect_task_does_not_short_circuit_for_ambiguous_model_type() -> None:
    """A model_type with >1 distinct task in MODEL_CLASS_MAPPING (e.g. bart:
    feature-extraction + text2text-generation) cannot be disambiguated by
    model_type alone, so detect_task must fall through to architecture-aware
    detection instead of short-circuiting to the first key (feature-extraction)."""
    cfg = _FakeConfig("bart", architectures=["BartForSequenceClassification"])
    with patch(_DETECT, return_value="text-classification") as m:
        task, source = detect_task(cfg)
    assert task == "text-classification"
    assert source == "TasksManager"
    m.assert_called_once()


def test_detect_task_uses_single_real_task_despite_none_sentinel() -> None:
    """A model_type with a None default-class sentinel plus exactly one real task
    (sam: (sam, None) + (sam, mask-generation)) short-circuits to that real task
    rather than falling through on the None sentinel."""
    cfg = _FakeConfig("sam")
    with patch(_DETECT) as m:
        task, source = detect_task(cfg)
    assert (task, source) == ("mask-generation", "HF_MODEL_CLASS_MAPPING")
    m.assert_not_called()


def test_detect_task_falls_through_for_multi_task_model_type_sam2() -> None:
    """sam2 maps to multiple real tasks (image-segmentation, feature-extraction,
    image-feature-extraction, mask-generation) in MODEL_CLASS_MAPPING, so the
    (sam2, None) sentinel cannot disambiguate and detection must fall through to
    architecture-aware detection rather than short-circuiting."""
    cfg = _FakeConfig("sam2")
    with patch(_DETECT, return_value="feature-extraction") as m:
        task, source = detect_task(cfg)
    assert (task, source) == ("feature-extraction", "TasksManager")
    m.assert_called_once()


def test_detect_task_short_circuits_for_unambiguous_model_type() -> None:
    """A model_type with a single task entry (segformer -> image-segmentation)
    still short-circuits via MODEL_CLASS_MAPPING without consulting TasksManager."""
    cfg = _FakeConfig("segformer")
    with patch(_DETECT) as m:
        task, source = detect_task(cfg)
    assert (task, source) == ("image-segmentation", "HF_MODEL_CLASS_MAPPING")
    m.assert_not_called()


def test_resolve_case1_surfaces_modality_aware_task() -> None:
    """Orchestrator Case 1 (auto-detect) surfaces the D2-upgraded task; the model
    class is unchanged (resolved from the pre-upgrade Optimum task)."""
    cfg = _FakeConfig("faketype", image_size=518)
    sentinel_cls = type("Sentinel", (), {})
    with patch(
        "winml.modelkit.loader.task._detect_task_and_class_from_config",
        return_value=("feature-extraction", sentinel_cls),
    ):
        task, cls = resolve_task_and_model_class(cfg)
    assert task == "image-feature-extraction"
    assert cls is sentinel_cls
