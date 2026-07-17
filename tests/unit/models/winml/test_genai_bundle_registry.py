# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the genai-bundle recipe registry.

``register_genai_bundle`` is the single write point for
``GENAI_BUNDLE_REGISTRY``; ``resolve_genai_bundle`` is the read side that
imports the model packages populating it.  These tests pin the registered
Qwen3 recipe's shape and the registry's guard rails without any model download.
"""

from __future__ import annotations

import pytest

from winml.modelkit.models.winml import (
    GENAI_BUNDLE_REGISTRY,
    GenaiBundleRecipe,
    GenaiTarget,
    GenaiTransformerSpec,
    register_genai_bundle,
    resolve_genai_bundle,
)


def test_resolve_qwen3_returns_recipe():
    recipe = resolve_genai_bundle("qwen3")
    assert recipe is not None
    assert recipe.family == "qwen3"
    assert recipe.transformer.model_type == "qwen3_transformer_only"
    assert recipe.transformer.task == "text-generation"
    assert recipe.transformer.context_sub_model == "decoder_prefill"
    assert recipe.transformer.iterator_sub_model == "decoder_gen"
    assert {c.role for c in recipe.companions} == {"embeddings", "lm_head"}
    assert callable(recipe.assemble)
    assert len(recipe.transformer_onnx_passes) >= 1


def test_resolve_unregistered_returns_none():
    assert resolve_genai_bundle("bert") is None


def test_resolve_none_returns_none():
    assert resolve_genai_bundle(None) is None


def test_registry_is_populated_and_contains_qwen3():
    # resolve_* triggers registry population as an import side effect.
    resolve_genai_bundle("qwen3")
    assert "qwen3" in GENAI_BUNDLE_REGISTRY


def test_duplicate_registration_raises():
    existing = resolve_genai_bundle("qwen3")
    assert existing is not None
    dup = GenaiBundleRecipe(
        family="qwen3",
        transformer=GenaiTransformerSpec(
            model_type="x",
            task="text-generation",
            precision="w8a16",
            context_sub_model="a",
            iterator_sub_model="b",
        ),
        companions=(),
        assemble=lambda *_a, **_k: None,
        supported_targets=(GenaiTarget(ep="qnn", device="npu"),),
    )
    with pytest.raises(ValueError, match="already registered"):
        register_genai_bundle(dup)
    # The guard precedes insertion, so the real recipe is left untouched.
    assert resolve_genai_bundle("qwen3") is existing


def test_register_rejects_empty_supported_targets():
    """A recipe with no supported targets is a registration-time error."""
    recipe = GenaiBundleRecipe(
        family="no-targets-fam",
        transformer=GenaiTransformerSpec(
            model_type="x",
            task="text-generation",
            precision="w8a16",
            context_sub_model="a",
            iterator_sub_model="b",
        ),
        companions=(),
        assemble=lambda *_a, **_k: None,
        supported_targets=(),
    )
    with pytest.raises(ValueError, match="no supported_targets"):
        register_genai_bundle(recipe)
    assert "no-targets-fam" not in GENAI_BUNDLE_REGISTRY
