# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for ``resolve_task`` that download real HF configs.

These tests require network access (model-config download / cache).
Marked ``slow`` to match the other integration loader tests; deselect with
``pytest -m "not slow"``.
"""

import pytest
from transformers import AutoConfig, CLIPTextModelWithProjection

from winml.modelkit.loader.resolution import TaskSource, resolve_task


@pytest.mark.slow
def test_sentinel_default_for_sam():
    r = resolve_task(AutoConfig.from_pretrained("facebook/sam-vit-base"))
    assert r.task == "mask-generation"
    assert r.source == TaskSource.SENTINEL_DEFAULT


@pytest.mark.slow
def test_user_class_override():
    cfg = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
    r = resolve_task(cfg, model_class="CLIPTextModelWithProjection")
    assert r.model_class is CLIPTextModelWithProjection
    assert r.source == TaskSource.USER_CLASS
