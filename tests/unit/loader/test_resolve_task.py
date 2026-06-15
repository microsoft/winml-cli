# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the unified ``resolve_task`` resolver.

Offline / config-only: every case builds its config with
``AutoConfig.for_model`` so no network access is required.
"""

import pytest
from transformers import AutoConfig

from winml.modelkit.loader.resolution import TaskSource, resolve_task
from winml.modelkit.loader.task import to_optimum_task


def _cfg(model_type, architectures=None, **kw):
    cfg = AutoConfig.for_model(model_type)
    if architectures is not None:
        cfg.architectures = architectures
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_invariant_optimum_task_is_collapse_of_task():
    r = resolve_task(_cfg("vit", ["ViTForImageClassification"]))
    assert r.optimum_task == to_optimum_task(r.task)


def test_autodetect_text_classification_via_tasks_manager():
    r = resolve_task(_cfg("bert", ["BertForSequenceClassification"]))
    assert r.task == "text-classification"
    assert r.source == TaskSource.TASKS_MANAGER


def test_autodetect_modality_upgrade_for_vision_feature_extraction():
    r = resolve_task(_cfg("vit", ["ViTModel"]))
    assert r.task == "image-feature-extraction"
    assert r.optimum_task == "feature-extraction"


def test_seq2seq_fill_mask_is_upgraded_and_source_is_tasks_manager():
    r = resolve_task(_cfg("bart", ["BartForConditionalGeneration"], is_encoder_decoder=True))
    assert r.task == "text2text-generation"
    assert r.source == TaskSource.TASKS_MANAGER
    assert r.composite == {"encoder": "feature-extraction", "decoder": "text2text-generation"}


def test_user_task_preserved_verbatim_no_modality_upgrade():
    r = resolve_task(_cfg("vit", ["ViTModel"]), task="feature-extraction")
    assert r.task == "feature-extraction"
    assert r.source == TaskSource.USER_TASK
    assert r.composite is None


def test_no_architectures_uses_first_supported_task():
    cfg = AutoConfig.for_model("bert")
    assert not getattr(cfg, "architectures", None)
    r = resolve_task(cfg)
    assert r.source == TaskSource.WRAPPED_LIBRARY
    assert r.task in ("feature-extraction", "fill-mask")


def test_model_id_default_source():
    cfg = _cfg("bert")
    cfg._name_or_path = "prajjwal1/bert-tiny"
    r = resolve_task(cfg)
    assert r.task == "feature-extraction"
    assert r.source == TaskSource.MODEL_ID_DEFAULT


def test_user_class_unknown_raises_friendly_error():
    cfg = _cfg("bert", ["BertModel"])
    with pytest.raises(ValueError, match="not found for task"):
        resolve_task(cfg, model_class="NotARealClass")


def test_user_task_unsupported_raises_friendly_error():
    cfg = _cfg("bert", ["BertModel"])
    with pytest.raises(ValueError, match="not supported by TasksManager"):
        resolve_task(cfg, task="not-a-real-task")


def test_unimportable_architecture_falls_back_to_hf_task_default():
    cfg = _cfg("bert", ["TotallyNotAClass"])
    r = resolve_task(cfg)
    assert r.source == TaskSource.HF_TASK_DEFAULT


# --- ported from the deleted test_detect_task_from_config.py -----------------
# These pin the architecture-class -> TasksManager task inference that the
# (now-removed) ``_detect_task_from_config`` covered, asserted against the
# unified ``resolve_task`` auto-detect path.


def test_known_vision_architecture_resolves_via_tasks_manager():
    """A known vision architecture class infers its TasksManager task."""
    r = resolve_task(_cfg("resnet", ["ResNetForImageClassification"]))
    assert r.task == "image-classification"
    assert r.source == TaskSource.TASKS_MANAGER


def test_encoder_only_masked_lm_resolves_to_fill_mask():
    """BertForMaskedLM (encoder-only) stays fill-mask — not upgraded to seq2seq."""
    r = resolve_task(_cfg("bert", ["BertForMaskedLM"]))
    assert r.task == "fill-mask"
    assert r.source == TaskSource.TASKS_MANAGER


def test_uses_first_architecture_only():
    """architectures[0] drives detection when multiple architectures are present."""
    r = resolve_task(_cfg("resnet", ["ResNetForImageClassification", "SomeOtherClass"]))
    assert r.task == "image-classification"
    assert r.source == TaskSource.TASKS_MANAGER


def test_missing_architectures_with_unknown_model_type_falls_back_to_hf_task_default():
    """No architectures AND an unknown model_type (no wrapped-library route, no
    ONNX-exportable task): the legacy ``_detect_task_from_config`` raised ValueError;
    ``resolve_task`` instead falls back to the last-resort HF_TASK_DEFAULT."""
    cfg = AutoConfig.for_model("bert")
    cfg.architectures = None
    cfg.model_type = "totally-unknown-model-xyz"
    cfg._name_or_path = ""
    r = resolve_task(cfg)
    assert r.source == TaskSource.HF_TASK_DEFAULT


def test_user_class_inferred_task_is_modality_aware():
    """USER_CLASS without --task infers the task from the architecture, so it is surfaced
    modality-aware (consistent with the detection path) — a ViT backbone resolves to
    image-feature-extraction, not the modality-blind feature-extraction."""
    r = resolve_task(_cfg("vit", ["ViTModel"]), model_class="ViTModel")
    assert r.source == TaskSource.USER_CLASS
    assert r.task == "image-feature-extraction"
    assert r.optimum_task == "feature-extraction"


def test_user_class_with_explicit_task_preserved_verbatim():
    """USER_CLASS WITH an explicit --task preserves the user's task (no modality upgrade)."""
    r = resolve_task(
        _cfg("vit", ["ViTModel"]), model_class="ViTModel", task="feature-extraction"
    )
    assert r.source == TaskSource.USER_CLASS
    assert r.task == "feature-extraction"
