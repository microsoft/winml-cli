# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the unified ``resolve_task`` resolver.

Offline / config-only: every case builds its config with
``AutoConfig.for_model`` so no network access is required.
"""

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
