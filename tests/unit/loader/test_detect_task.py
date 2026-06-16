# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ``resolve_task`` — the single modality-aware resolver.

These tests pin the D2 vision-modality upgrade, which is applied to the SURFACED
task only. The internal ``_infer_task_from_architecture`` (used for model-class
resolution) is mocked so the dispatch reaches the TasksManager branch
deterministically without network.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from winml.modelkit.loader.resolution import (
    TaskSource,
    _resolve_task_modality,
    _upgrade_fill_mask_for_seq2seq,
    resolve_task,
)


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
        name_or_path: str = "",
    ) -> None:
        self.model_type = model_type
        self.architectures = architectures or ["FakeModel"]
        self.is_encoder_decoder = is_encoder_decoder
        self._name_or_path = name_or_path
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


# Stage 1c (TasksManager branch) of ``resolve_task`` calls this to infer the
# Optimum-canonical task from ``config.architectures``; patch it to drive the
# dispatch deterministically without network.
_INFER = "winml.modelkit.loader.resolution._infer_task_from_architecture"
# Stage 3 (modality upgrade) reads the architecture class's ``main_input_name``
# via this helper in ``resolution.py``.
_RESOLVE_CLASS = "winml.modelkit.loader.resolution._resolve_model_class_from_config"


def _fake_arch_class(main_input_name: str) -> type:
    """A stand-in architecture class exposing only ``main_input_name`` (the modality signal)."""
    return type("FakeArch", (), {"main_input_name": main_input_name})


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


def test_resolve_task_upgrades_pixel_values_feature_extraction_to_image() -> None:
    """resolve_task wires the modality upgrade: a feature-extraction backbone whose
    architecture takes pixel_values surfaces as image-feature-extraction."""
    cfg = _FakeConfig("faketype")
    with (
        patch(_INFER, return_value="feature-extraction") as m,
        patch(_RESOLVE_CLASS, return_value=_fake_arch_class("pixel_values")),
    ):
        r = resolve_task(cfg)
    assert r.task == "image-feature-extraction"
    assert r.optimum_task == "feature-extraction"
    assert r.source == TaskSource.TASKS_MANAGER
    # Internal Optimum-canonical detection is still consulted (pre-D2).
    m.assert_called_once()


@pytest.mark.parametrize(
    "main_input_name, expected",
    [
        ("pixel_values", "image-feature-extraction"),  # vision backbone (ViT/DINOv2)
        ("input_ids", "feature-extraction"),  # text encoder (BERT/CLIP-text)
        ("input_values", "feature-extraction"),  # audio (wav2vec2/AST) — no downstream yet
        ("input_features", "feature-extraction"),  # audio (whisper-style) — no downstream yet
    ],
)
def test_resolve_task_modality_by_main_input(main_input_name: str, expected: str) -> None:
    """feature-extraction is upgraded by the architecture class's main_input_name; only
    pixel_values has a modality-aware downstream today, so text/audio stay as-is."""
    cfg = _FakeConfig("faketype")
    with patch(_RESOLVE_CLASS, return_value=_fake_arch_class(main_input_name)):
        assert _resolve_task_modality(cfg, "feature-extraction") == expected


def test_resolve_task_modality_only_touches_feature_extraction() -> None:
    """A non-feature-extraction task is returned unchanged without resolving a class."""
    cfg = _FakeConfig("faketype")
    with patch(_RESOLVE_CLASS, side_effect=AssertionError("must not resolve a class")):
        assert _resolve_task_modality(cfg, "image-classification") == "image-classification"


def test_resolve_task_modality_noop_when_class_unresolvable() -> None:
    """When config.architectures cannot be resolved, feature-extraction is left as-is."""
    cfg = _FakeConfig("faketype")
    with patch(_RESOLVE_CLASS, side_effect=ValueError("no architectures")):
        assert _resolve_task_modality(cfg, "feature-extraction") == "feature-extraction"


def test_resolve_task_falls_back_to_hf_task_default() -> None:
    """When TasksManager detection raises ValueError, fall back to HF_TASK_DEFAULTS."""
    cfg = _FakeConfig("faketype")
    with patch(_INFER, side_effect=ValueError("no task")):
        r = resolve_task(cfg)
    assert r.source == TaskSource.HF_TASK_DEFAULT


def test_resolve_task_does_not_short_circuit_for_ambiguous_model_type() -> None:
    """A model_type with >1 distinct task in MODEL_CLASS_MAPPING (e.g. bart:
    feature-extraction + text2text-generation) cannot be disambiguated by
    model_type alone, so resolve_task must fall through to architecture-aware
    detection instead of short-circuiting to the first key (feature-extraction)."""
    cfg = _FakeConfig("bart", architectures=["BartForSequenceClassification"])
    with patch(_INFER, return_value="text-classification") as m:
        r = resolve_task(cfg)
    assert r.task == "text-classification"
    assert r.source == TaskSource.TASKS_MANAGER
    m.assert_called_once()


def test_resolve_task_uses_single_real_task_despite_none_sentinel() -> None:
    """A model_type with a None default-class sentinel plus exactly one real task
    (sam: (sam, None) + (sam, mask-generation)) short-circuits to that real task
    rather than falling through on the None sentinel."""
    cfg = _FakeConfig("sam")
    with patch(_INFER) as m:
        r = resolve_task(cfg)
    assert (r.task, r.source) == ("mask-generation", TaskSource.SENTINEL_DEFAULT)
    m.assert_not_called()


def test_resolve_task_applies_sentinel_for_multi_task_model_type_sam2() -> None:
    """sam2 maps to multiple real tasks but also carries a (sam2, None) sentinel whose
    canonical export target is the mask-generation decoder. The unified override applies
    that sentinel on the detect path too (matching the build path), so detection resolves
    to mask-generation without consulting the architecture head — inspect now predicts the
    artifact build actually produces."""
    cfg = _FakeConfig("sam2")
    with patch(_INFER) as m:
        r = resolve_task(cfg)
    assert (r.task, r.source) == ("mask-generation", TaskSource.SENTINEL_DEFAULT)
    m.assert_not_called()


def test_resolve_task_applies_model_id_override() -> None:
    """A configured model-id default (prajjwal1/bert-tiny -> feature-extraction) now fires
    on the detect path too (previously build-only), so inspect agrees with build."""
    cfg = _FakeConfig("bert", name_or_path="prajjwal1/bert-tiny")
    with patch(_INFER) as m:
        r = resolve_task(cfg)
    assert r.task == "feature-extraction"
    assert r.source == TaskSource.MODEL_ID_DEFAULT
    m.assert_not_called()


def test_resolve_task_no_override_for_single_entry_without_sentinel() -> None:
    """segformer's only MODEL_CLASS_MAPPING entry (image-segmentation) is a class-fix, NOT
    a (model_type, None) sentinel, so it is not treated as a default-task override:
    detection falls through to the architecture head instead of short-circuiting. A
    fine-tuned classification checkpoint therefore keeps image-classification rather than
    being forced to image-segmentation."""
    cfg = _FakeConfig("segformer")
    with patch(_INFER, return_value="image-classification") as m:
        r = resolve_task(cfg)
    assert (r.task, r.source) == ("image-classification", TaskSource.TASKS_MANAGER)
    m.assert_called_once()


def test_resolve_task_case1_surfaces_modality_aware_task() -> None:
    """Auto-detect (Case 1) surfaces the modality-aware task. Modality comes from the
    architecture class's main_input_name (pixel_values), independent of the class
    resolved for loading — which is exactly why modality must read the arch class
    (config.architectures), not the resolved/Auto class."""
    cfg = _FakeConfig("faketype")
    with (
        patch(_INFER, return_value="feature-extraction"),
        patch(_RESOLVE_CLASS, return_value=_fake_arch_class("pixel_values")),
    ):
        r = resolve_task(cfg)
    assert r.task == "image-feature-extraction"
    assert r.optimum_task == "feature-extraction"
